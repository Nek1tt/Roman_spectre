"""
main.py — Entry point for Raman v10 training pipeline.

Usage
-----
# Full training with Optuna tuning:
    python main.py --data_root ./

# Control number of Optuna trials:
    python main.py --data_root ./ --optuna_trials_ridge 30 --optuna_trials_cnn 20

# Disable Optuna:
    python main.py --data_root ./ --optuna_trials_ridge 0 --optuna_trials_cnn 0

# Inference with a saved ML model:
    python main.py --load_model outputs/best_model_center1500.pkl \
                   --predict_dir /path/to/spectra

# Inference with a saved CNN model:
    python main.py --load_cnn outputs/cnn_weights_center1500.pt \
                   --cnn_meta outputs/cnn_meta_center1500.pkl \
                   --predict_dir /path/to/spectra
"""

import argparse
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# GPU detection (kept here to avoid circular imports)
# ---------------------------------------------------------------------------

def detect_gpu(force: bool = False) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "available": False, "name": "–",
        "xgb_device": "cpu", "xgb_tree": "hist", "lgbm_device": "cpu",
    }
    if force:
        return {**info, "available": True, "name": "forced",
                "xgb_device": "cuda", "lgbm_device": "gpu"}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            name = r.stdout.strip().split("\n")[0]
            return {**info, "available": True, "name": name,
                    "xgb_device": "cuda", "lgbm_device": "gpu"}
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data_root",        default=None)
    p.add_argument("--n_grid",           type=int,   default=256)
    p.add_argument("--norm",             default="snv",
                   choices=["snv", "peak_phe", "area", "minmax"])
    p.add_argument("--use_als",          action="store_true")
    p.add_argument("--fuse_bands",       action="store_true")
    p.add_argument("--permutation_test", action="store_true")
    p.add_argument("--n_permutations",   type=int,   default=200)
    p.add_argument("--save_plots",       action="store_true")
    p.add_argument("--n_jobs",           type=int,   default=-1)
    p.add_argument("--use_gpu",          action="store_true")
    p.add_argument("--force_cpu",        action="store_true")
    # CNN
    p.add_argument("--skip_cnn",         action="store_true")
    p.add_argument("--skip_ml",          action="store_true")
    p.add_argument("--cnn_epochs",       type=int,   default=80)
    p.add_argument("--cnn_batch",        type=int,   default=256)
    p.add_argument("--cnn_lr",           type=float, default=1e-3)
    p.add_argument("--cnn_weight_decay", type=float, default=1e-4)
    p.add_argument("--cnn_dropout",      type=float, default=0.4)
    p.add_argument("--cnn_patience",     type=int,   default=15)
    # Optuna
    p.add_argument("--optuna_trials_ridge", type=int, default=20,
                   help="Optuna trials for Ridge alpha. 0 = disable.")
    p.add_argument("--optuna_trials_cnn",   type=int, default=15,
                   help="Optuna trials for CNN (lr, wd, dropout). 0 = disable.")
    p.add_argument("--optuna_cnn_epochs",   type=int, default=10,
                   help="Epochs per Optuna CNN trial (quick eval).")
    # Inference
    p.add_argument("--load_model",   default=None)
    p.add_argument("--load_cnn",     default=None)
    p.add_argument("--cnn_meta",     default=None)
    p.add_argument("--predict_dir",  default=None)
    return p


def main() -> None:
    args    = build_parser().parse_args()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # ── Inference mode ───────────────────────────────────────────────────────
    if args.load_model or args.load_cnn:
        if not args.predict_dir:
            print("ERROR: --predict_dir required for inference mode")
            return
        from inference import run_inference
        run_inference(args)
        return

    if not args.data_root:
        print("ERROR: --data_root required")
        return

    # ── GPU setup ────────────────────────────────────────────────────────────
    gpu = detect_gpu(force=args.use_gpu)
    if args.force_cpu:
        gpu = {
            "available": False, "name": "–",
            "xgb_device": "cpu", "xgb_tree": "hist", "lgbm_device": "cpu",
        }

    print("=" * 65)
    print(f"GPU     : {'✅ ' + gpu['name'] if gpu['available'] else 'CPU'}")
    print(f"n_grid  : {args.n_grid},  norm: {args.norm}")
    cnn_status = (
        "OFF (--skip_cnn)" if args.skip_cnn
        else f"ON  epochs={args.cnn_epochs} batch={args.cnn_batch} "
             f"lr={args.cnn_lr} [2ch ResNet+SE]"
    )
    print(f"CNN     : {cnn_status}")
    print(f"ML      : {'OFF (--skip_ml)' if args.skip_ml else 'ON'}")
    print(f"Fusion  : {args.fuse_bands},  perm_test: {args.permutation_test}")
    print(f"Optuna  : Ridge trials={args.optuna_trials_ridge}  "
          f"CNN trials={args.optuna_trials_cnn} "
          f"(cnn_search_epochs={args.optuna_cnn_epochs})")
    print("=" * 65)

    # ── Training mode ────────────────────────────────────────────────────────
    from data_loading import load_dataset_maps
    from pipeline import run_pipeline, run_fusion

    all_maps = load_dataset_maps(args.data_root, n_grid=args.n_grid)
    results: Dict[str, Optional[Dict]] = {}
    for center, maps in all_maps.items():
        if maps:
            tag          = f"center{center}"
            results[tag] = run_pipeline(maps, tag, args, gpu, out_dir)

    if args.fuse_bands and not args.skip_ml:
        df_f = run_fusion(results, gpu, args, out_dir)
        if df_f is not None:
            results["fused"] = {
                "logo_df": df_f, "best_acc": df_f["LOGO_mean"].max()
            }

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📋 FINAL SUMMARY")
    print("=" * 65)
    print(f"  {'Source':<16} {'Best model':<16} {'LOGO':>10}  {'p-value':>9}")
    print("  " + "-" * 57)
    for tag, r in results.items():
        if r is None:
            continue
        df = r.get("logo_df", r.get("all_metrics"))
        if df is None:
            continue
        best = df.loc[df["LOGO_mean"].idxmax()]
        acc  = best["LOGO_mean"]
        pv   = r.get("p_value")
        pv_s = f"{pv:.3f}" if pv is not None else "N/A"
        flag = "↑" if acc > 1 / 3 else "↓"
        print(f"  {tag:<16} {best['Model']:<16} {acc:>8.3f} {flag}  {pv_s:>9}")
    print(f"\n  Random baseline : {1/3:.3f}")
    print(f"  Models → outputs/")
    print("  ✅ Done.")


if __name__ == "__main__":
    main()

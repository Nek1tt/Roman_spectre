"""
inference.py — CLI inference: load a saved ML or CNN model and predict on
a folder of .txt spectrum files.

Called from main.py when --load_model or --load_cnn is specified.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from features import featurize_single_raw_spectrum
from preprocessing import preprocess_spectrum


# ---------------------------------------------------------------------------
# File reader
# ---------------------------------------------------------------------------

def _load_single_spectrum_file(
    fpath: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read a 2- or 4-column spectrum .txt file → (wave, intensity)."""
    try:
        df = pd.read_csv(
            fpath, sep=r"\s+", comment="#", header=None, dtype=np.float64
        ).dropna()
    except Exception as e:
        print(f"  [WARN] Cannot read {fpath.name}: {e}")
        return None

    if df.shape[1] == 2:
        wave, intn = df.iloc[:, 0].values, df.iloc[:, 1].values
    elif df.shape[1] >= 4:
        df.columns = ["X", "Y", "Wave", "Intensity"] + list(
            range(df.shape[1] - 4)
        )
        first_xy = df[["X", "Y"]].drop_duplicates().iloc[0]
        mask = (df["X"] == first_xy["X"]) & (df["Y"] == first_xy["Y"])
        sub  = df[mask].sort_values("Wave")
        wave, intn = sub["Wave"].values, sub["Intensity"].values
    else:
        print(f"  [WARN] Unexpected columns in {fpath.name}: {df.shape[1]}")
        return None

    if len(wave) < 10:
        print(f"  [WARN] Too few points in {fpath.name}: {len(wave)}")
        return None

    order = np.argsort(wave)
    return wave[order], intn[order]


# ---------------------------------------------------------------------------
# Main inference entry-point
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    predict_dir = Path(args.predict_dir)
    txt_files   = sorted(predict_dir.glob("*.txt"))
    if not txt_files:
        print(f"  [WARN] No .txt files found in {predict_dir}")
        return

    print(f"\n{'='*65}")
    print(f"🔍 INFERENCE MODE — {len(txt_files)} file(s) in {predict_dir}")
    print(f"{'='*65}")

    results_rows: List[Dict] = []

    # ── ML inference ────────────────────────────────────────────────────────
    if args.load_model:
        payload = joblib.load(args.load_model)
        model   = payload["model"]
        le      = payload["label_encoder"]
        grid    = payload["grid"]
        bands   = payload["bands"]
        ctag    = payload["center_tag"]
        norm    = payload.get("norm", "snv")
        use_als = payload.get("use_als", False)

        print(f"  ML model: {type(model).__name__}  "
              f"LOGO acc={payload['logo_acc']:.3f}")
        print(f"  Classes : {list(le.classes_)}")
        print(f"  center  : {ctag},  norm={norm},  use_als={use_als}\n")

        for fpath in txt_files:
            raw = _load_single_spectrum_file(fpath)
            if raw is None:
                continue
            wave, intn = raw
            feats    = featurize_single_raw_spectrum(
                wave, intn, grid, bands, ctag, use_als=use_als, norm=norm
            )
            feats_2d = feats.reshape(1, -1)
            pred_idx = model.predict(feats_2d)[0]
            pred_cls = le.inverse_transform([pred_idx])[0]
            try:
                proba     = model.predict_proba(feats_2d)[0]
                proba_str = "  ".join(
                    f"{cls}={p:.3f}" for cls, p in zip(le.classes_, proba)
                )
            except AttributeError:
                proba     = np.array([])
                proba_str = "N/A"

            print(f"  {fpath.name:<40}  →  {pred_cls:<10}  [{proba_str}]")
            results_rows.append({
                "file":      fpath.name,
                "predicted": pred_cls,
                **{f"p_{cls}": float(p)
                   for cls, p in zip(le.classes_, proba)},
            })

    # ── CNN inference ────────────────────────────────────────────────────────
    elif args.load_cnn:
        if not args.cnn_meta:
            print("ERROR: --cnn_meta required when using --load_cnn")
            return

        from cnn_model import CNNTrainer, try_import_torch
        torch_mods = try_import_torch()
        if torch_mods[0] is None:
            print("ERROR: PyTorch required for CNN inference")
            return
        torch, nn, *_ = torch_mods

        meta    = joblib.load(args.cnn_meta)
        le      = meta["label_encoder"]
        grid    = meta["grid"]
        bands   = meta["bands"]
        ctag    = meta["center_tag"]
        norm    = meta.get("norm", "snv")
        use_als = meta.get("use_als", False)
        n_grid  = meta["n_grid"]
        n_cls   = meta["n_classes"]
        dropout = meta.get("dropout", 0.4)

        device  = (torch.device("cuda") if torch.cuda.is_available()
                   else torch.device("cpu"))
        trainer = CNNTrainer(
            n_grid=n_grid, n_classes=n_cls, device=device, dropout=dropout
        )
        trainer.load(args.load_cnn, nn)

        print(f"  CNN model loaded  n_grid={n_grid},  n_classes={n_cls},  "
              f"dropout={dropout:.3f}")
        print(f"  Classes : {list(le.classes_)}")
        print(f"  center  : {ctag},  norm={norm},  use_als={use_als}\n")

        for fpath in txt_files:
            raw = _load_single_spectrum_file(fpath)
            if raw is None:
                continue
            wave, intn   = raw
            spec_interp  = np.interp(grid, wave, intn)
            spec_proc, d2 = preprocess_spectrum(
                spec_interp, grid, use_als=use_als, norm=norm
            )
            spec_2ch = np.stack([spec_proc, d2], axis=0)

            pred_idx, proba = trainer.predict_single_spectrum(spec_2ch)
            pred_cls        = le.inverse_transform([pred_idx])[0]
            proba_str       = "  ".join(
                f"{cls}={p:.3f}" for cls, p in zip(le.classes_, proba)
            )

            print(f"  {fpath.name:<40}  →  {pred_cls:<10}  [{proba_str}]")
            results_rows.append({
                "file":      fpath.name,
                "predicted": pred_cls,
                **{f"p_{cls}": float(p)
                   for cls, p in zip(le.classes_, proba)},
            })

    else:
        print("ERROR: specify --load_model or --load_cnn")
        return

    if results_rows:
        out_df   = pd.DataFrame(results_rows)
        out_path = Path("outputs") / "predictions.csv"
        Path("outputs").mkdir(exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"\n  💾 Predictions saved → {out_path}")
        print(f"\n  Distribution of predictions:")
        print(out_df["predicted"].value_counts().to_string())

    print("\n  ✅ Inference done.")

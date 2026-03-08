"""
pipeline.py — Top-level training pipeline and band-fusion logic.

Public API
----------
run_pipeline(maps, center_tag, args, gpu, out_dir) → dict | None
run_fusion(results, gpu, args, out_dir)             → DataFrame | None
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from Roman_spectre.constants import BAND_RANGES
from Roman_spectre.features import build_pixel_feature_matrix
from Roman_spectre.ml_models import OptunaRidgeClf, get_ml_models
from Roman_spectre.evaluation import (
    permutation_test, run_gss, run_logo_cnn, run_logo_ml, run_sgkf,
)
from Roman_spectre.preprocessing import preprocess_map_pixels
from Roman_spectre.visualisation import (
    plot_confusion_matrix, plot_cv_all, plot_feature_importance,
    plot_pca, plot_shap_ml, plot_cnn_saliency,
)


# ---------------------------------------------------------------------------
# Single-centre pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    maps: List[Dict],
    center_tag: str,
    args: argparse.Namespace,
    gpu: Dict[str, Any],
    out_dir: Path,
) -> Optional[Dict]:
    if not maps:
        print(f"  [SKIP] No maps for {center_tag}")
        return None

    print(f"\n{'='*65}")
    print(f"🔬 PIPELINE: {center_tag}  ({len(maps)} maps)")
    print(f"{'='*65}")

    data = build_pixel_feature_matrix(
        maps, center_tag, use_als=args.use_als,
        norm=args.norm, n_jobs=args.n_jobs,
    )
    if data is None:
        return None

    X, y, aids = data["X"], data["y"], data["aids"]
    grid       = data["grid"]
    feat_names = data["feat_names"]

    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)
    classes = le.classes_
    le_grp  = LabelEncoder()
    groups  = le_grp.fit_transform(aids)
    print(f"  Animals: {list(le_grp.classes_)}")

    if args.save_plots:
        plot_pca(X, y, aids, f"PCA features – {center_tag}",
                 out_dir, args.save_plots, f"pca_{center_tag}.png")

    all_logo_rows: List[pd.DataFrame] = []
    gss_df = gkf_df = None

    if not args.skip_ml:
        ml_models = get_ml_models(
            gpu,
            optuna_ridge_trials=args.optuna_trials_ridge,
            ridge_groups=groups,
        )
        print(f"\n📊 GroupShuffleSplit — {center_tag}")
        gss_df = run_gss(ml_models, X, y_enc, groups, classes)

        print(f"\n📊 StratifiedGroupKFold — {center_tag}")
        gkf_df = run_sgkf(ml_models, X, y_enc, groups, classes)

        print(f"\n📊 LOGO — {center_tag}  [ML]")
        logo_ml = run_logo_ml(ml_models, X, y_enc, groups, classes,
                              label=center_tag)
        all_logo_rows.append(logo_ml)

    if not args.skip_cnn:
        from Roman_spectre.cnn_model import try_import_torch
        torch_mods = try_import_torch()
        if torch_mods[0] is not None:
            torch = torch_mods[0]
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"\n  🖥️  CNN device: CUDA ({torch.cuda.get_device_name(0)})")
            else:
                device = torch.device("cpu")
                print(f"\n  🖥️  CNN device: CPU")

            print(f"\n📊 LOGO — {center_tag}  [CNN 2-ch ResNet]")
            logo_cnn = run_logo_cnn(maps, grid, le, device, args)
            if logo_cnn is not None:
                all_logo_rows.append(logo_cnn)
        else:
            print("  [SKIP] CNN: PyTorch not available")

    if not all_logo_rows:
        return None

    logo_df   = pd.concat(all_logo_rows, ignore_index=True)
    best_name = logo_df.loc[logo_df["LOGO_mean"].idxmax(), "Model"]
    best_row  = logo_df[logo_df["Model"] == best_name].iloc[0]
    best_acc  = best_row["LOGO_mean"]
    print(f"\n🏆 Best: {best_name}  LOGO={best_acc:.3f}")

    plot_cv_all(logo_df, gss_df, gkf_df,
                out_dir, args.save_plots, suffix=f"_{center_tag}")
    plot_confusion_matrix(
        best_row["y_true"], best_row["y_pred"],
        classes, f"LOGO {best_name} {center_tag}",
        out_dir, args.save_plots,
    )

    p_value = None
    if args.permutation_test and not args.skip_ml:
        best_ml_name = (logo_df[logo_df["type"] == "ML"]
                        .loc[logo_df[logo_df["type"] == "ML"]["LOGO_mean"].idxmax(),
                             "Model"])
        ml_models_fresh = get_ml_models(
            gpu, optuna_ridge_trials=0
        )
        p_value = permutation_test(
            ml_models_fresh[best_ml_name], X, y_enc, groups,
            logo_df.loc[logo_df["Model"] == best_ml_name,
                        "LOGO_mean"].values[0],
            n_perm=args.n_permutations,
        )

    if not args.skip_ml:
        best_ml_name  = (logo_df[logo_df["type"] == "ML"]
                         .loc[logo_df[logo_df["type"] == "ML"]["LOGO_mean"].idxmax(),
                              "Model"])
        ml_models_fin = get_ml_models(
            gpu,
            optuna_ridge_trials=args.optuna_trials_ridge,
            ridge_groups=groups,
        )
        ml_models_fin[best_ml_name].fit(X, y_enc)
        plot_feature_importance(
            ml_models_fin[best_ml_name], feat_names, top_n=30,
            title=f"{best_ml_name} {center_tag}",
            out_dir=out_dir, save_plots=args.save_plots,
        )

        if args.save_plots and best_ml_name in ("XGBoost", "LightGBM", "HistGB"):
            plot_shap_ml(
                ml_models_fin[best_ml_name], X, feat_names, classes,
                title=f"SHAP {best_ml_name} {center_tag}",
                out_dir=out_dir, save_plots=args.save_plots,
            )

        payload = {
            "model":         ml_models_fin[best_ml_name],
            "label_encoder": le,
            "feat_names":    feat_names,
            "center_tag":    center_tag,
            "grid":          grid,
            "bands":         data["bands"],
            "logo_acc":      best_acc,
            "p_value":       p_value,
            "norm":          args.norm,
            "use_als":       args.use_als,
            "wave_range":    BAND_RANGES.get(
                int(center_tag.replace("center", "")), (None, None)
            ),
        }
        save_path = out_dir / f"best_model_{center_tag}.pkl"
        joblib.dump(payload, save_path)
        print(f"  💾 ML model saved → {save_path}")

    if not args.skip_cnn and "CNN-1D-Res" in logo_df["Model"].values:
        from Roman_spectre.cnn_model import CNNTrainer, try_import_torch
        torch_mods = try_import_torch()
        if torch_mods[0] is not None:
            torch  = torch_mods[0]
            device = (torch.device("cuda") if torch.cuda.is_available()
                      else torch.device("cpu"))
            trainer_final = CNNTrainer(
                n_grid=len(grid), n_classes=len(classes), device=device,
                epochs=args.cnn_epochs, batch_size=args.cnn_batch,
                lr=args.cnn_lr, weight_decay=args.cnn_weight_decay,
                dropout=args.cnn_dropout,
                patience=args.cnn_patience,
                optuna_n_trials=args.optuna_trials_cnn,
                optuna_epochs=args.optuna_cnn_epochs,
            )
            all_pix: List[np.ndarray] = []
            all_y_l: List[int]        = []
            for rec in maps:
                pix_raw      = np.array([np.interp(grid, rec["grid"], px)
                                          for px in rec["pixels"]])
                pix_proc, pix_d2 = preprocess_map_pixels(
                    pix_raw, grid, use_als=args.use_als,
                    norm=args.norm, n_jobs=args.n_jobs,
                )
                pix_2ch = np.stack([pix_proc, pix_d2], axis=1).astype(
                    np.float32
                )
                all_pix.append(pix_2ch)
                all_y_l.extend(
                    [le.transform([rec["label"]])[0]] * len(pix_proc)
                )
            X_all_2ch = np.vstack(all_pix)
            y_all     = np.array(all_y_l)
            trainer_final.fit(X_all_2ch, y_all)

            cnn_path = out_dir / f"cnn_weights_{center_tag}.pt"
            trainer_final.save(str(cnn_path))
            print(f"  💾 CNN weights saved → {cnn_path}")

            cnn_meta = {
                "n_grid":             len(grid),
                "n_classes":          len(classes),
                "label_encoder":      le,
                "grid":               grid,
                "bands":              data["bands"],
                "center_tag":         center_tag,
                "norm":               args.norm,
                "use_als":            args.use_als,
                "dropout":            trainer_final.dropout,
                "best_optuna_params": trainer_final.best_params,
                "wave_range":         BAND_RANGES.get(
                    int(center_tag.replace("center", "")), (None, None)
                ),
            }
            cnn_meta_path = out_dir / f"cnn_meta_{center_tag}.pkl"
            joblib.dump(cnn_meta, cnn_meta_path)
            print(f"  💾 CNN meta saved  → {cnn_meta_path}")

            if args.save_plots:
                plot_cnn_saliency(
                    trainer_final, X_all_2ch, y_all, grid, classes,
                    title=f"CNN Saliency {center_tag}",
                    out_dir=out_dir, save_plots=args.save_plots,
                )

    print(f"\n📋 SUMMARY  {center_tag}")
    summary = logo_df[["Model", "LOGO_mean", "LOGO_std"]].copy()
    if gss_df is not None:
        summary = summary.merge(gss_df[["Model", "GSS_mean", "GSS_std"]],
                                on="Model", how="left")
    if gkf_df is not None:
        summary = summary.merge(gkf_df[["Model", "GKF_mean", "GKF_std"]],
                                on="Model", how="left")

    print(f"  {'Model':<16} {'LOGO':>12}  {'GSS':>12}  {'SGKF':>12}")
    print("  " + "-" * 57)
    for _, row in summary.sort_values("LOGO_mean", ascending=False).iterrows():
        mark = "  ◀" if row["Model"] == best_name else ""
        gss  = (f"{row['GSS_mean']:.3f}±{row['GSS_std']:.3f}"
                if "GSS_mean" in row and not pd.isna(row.get("GSS_mean"))
                else "N/A")
        gkf  = (f"{row['GKF_mean']:.3f}±{row['GKF_std']:.3f}"
                if "GKF_mean" in row and not pd.isna(row.get("GKF_mean"))
                else "N/A")
        print(f"  {row['Model']:<16} "
              f"{row['LOGO_mean']:>6.3f}±{row['LOGO_std']:.3f}  "
              f"{gss:>12}  {gkf:>12}{mark}")
    if p_value is not None:
        print(f"  p-value = {p_value:.3f}")

    return {
        "logo_df": logo_df, "gss_df": gss_df, "gkf_df": gkf_df,
        "X": X, "y": y, "aids": aids, "feat_names": feat_names,
        "le": le, "groups": groups, "best_acc": best_acc,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Band fusion
# ---------------------------------------------------------------------------

def run_fusion(
    results: Dict,
    gpu: Dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """Fuse centre-1500 and centre-2900 feature matrices and run ML LOGO."""
    r1 = results.get("center1500")
    r2 = results.get("center2900")
    if r1 is None or r2 is None:
        return None

    print(f"\n{'='*65}")
    print(f"🔀 BAND FUSION: center1500 + center2900")
    print(f"{'='*65}")

    aid1 = r1["aids"]; X1 = r1["X"]; y1 = r1["y"]
    aid2 = r2["aids"]; X2 = r2["X"]
    common = sorted(set(aid1) & set(aid2))
    print(f"  Common animals: {len(common)}")

    rows:  List[np.ndarray] = []
    y_f:   List[str]        = []
    aid_f: List[str]        = []
    for animal in common:
        m1 = X1[aid1 == animal].mean(axis=0)
        m2 = X2[aid2 == animal].mean(axis=0)
        rows.append(np.concatenate([m1, m2]))
        y_f.append(y1[aid1 == animal][0])
        aid_f.append(animal)

    X_f   = np.array(rows, dtype=np.float32)
    y_f   = np.array(y_f)
    aid_f = np.array(aid_f)
    names_f = (
        [f"c1500_{n}" for n in r1["feat_names"]] +
        [f"c2900_{n}" for n in r2["feat_names"]]
    )

    le_f      = LabelEncoder()
    y_f_enc   = le_f.fit_transform(y_f)
    groups_f  = LabelEncoder().fit_transform(aid_f)
    print(f"  Fused matrix: {X_f.shape[0]} × {X_f.shape[1]}")

    ml_models = get_ml_models(
        gpu,
        optuna_ridge_trials=args.optuna_trials_ridge,
        ridge_groups=groups_f,
    )
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import accuracy_score, classification_report
    logo           = LeaveOneGroupOut()
    fusion_results: List[Dict] = []

    for name, model in ml_models.items():
        if isinstance(model, OptunaRidgeClf):
            model.cv_groups = groups_f
        fold_scores: List[float] = []
        ft: List[int] = []
        fp: List[int] = []
        for tr, te in logo.split(X_f, y_f_enc, groups_f):
            if isinstance(model, OptunaRidgeClf):
                model.cv_groups = groups_f[tr]
            model.fit(X_f[tr], y_f_enc[tr])
            preds = model.predict(X_f[te])
            fold_scores.append(accuracy_score(y_f_enc[te], preds))
            ft.extend(y_f_enc[te].tolist())
            fp.extend(preds.tolist())
        scores = np.array(fold_scores)
        print(f"\n  {name}: LOGO={scores.mean():.3f}±{scores.std():.3f}  "
              f"{[f'{s:.2f}' for s in scores]}")
        print(classification_report(ft, fp,
                                     target_names=le_f.classes_, digits=3))
        fusion_results.append({
            "Model":     name,
            "LOGO_mean": scores.mean(),
            "LOGO_std":  scores.std(),
            "y_true":    ft,
            "y_pred":    fp,
        })

    df_f      = pd.DataFrame(fusion_results)
    best_name = df_f.loc[df_f["LOGO_mean"].idxmax(), "Model"]
    best_row  = df_f[df_f["Model"] == best_name].iloc[0]
    best_acc  = best_row["LOGO_mean"]
    print(f"\n🏆 Fusion best: {best_name}  LOGO={best_acc:.3f}")

    plot_confusion_matrix(
        best_row["y_true"], best_row["y_pred"],
        le_f.classes_, f"Fusion {best_name}",
        out_dir, args.save_plots,
    )
    plot_cv_all(df_f, None, None, out_dir, args.save_plots, suffix="_fused")

    if args.permutation_test:
        permutation_test(
            ml_models[best_name], X_f, y_f_enc, groups_f,
            best_acc, n_perm=args.n_permutations,
        )

    ml_models[best_name].fit(X_f, y_f_enc)

    if args.save_plots and best_name in ("XGBoost", "LightGBM", "HistGB"):
        plot_shap_ml(
            ml_models[best_name], X_f, names_f, le_f.classes_,
            title=f"SHAP {best_name} fused",
            out_dir=out_dir, save_plots=args.save_plots,
        )

    joblib.dump(
        {
            "model":         ml_models[best_name],
            "label_encoder": le_f,
            "feat_names":    names_f,
            "logo_acc":      best_acc,
            "norm":          args.norm,
        },
        out_dir / "best_model_fused.pkl",
    )
    print(f"  💾 Saved → {out_dir / 'best_model_fused.pkl'}")
    return df_f

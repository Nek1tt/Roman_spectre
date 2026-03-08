"""
evaluation.py — Cross-validation routines and permutation significance test.

Public API
----------
run_logo_ml(models, X, y_enc, groups, classes)  → DataFrame
run_logo_cnn(maps, grid, le, device, args)       → DataFrame | None
run_gss(models, X, y_enc, groups, classes)       → DataFrame
run_sgkf(models, X, y_enc, groups, classes)      → DataFrame
permutation_test(model, X, y_enc, groups, ...)   → float (p-value)
"""

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GroupShuffleSplit,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from Roman_spectre.ml_models import OptunaRidgeClf
from Roman_spectre.preprocessing import preprocess_map_pixels


# ---------------------------------------------------------------------------
# ML cross-validation
# ---------------------------------------------------------------------------

def run_logo_ml(
    models: Dict,
    X: np.ndarray,
    y_enc: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    label: str = "",
) -> pd.DataFrame:
    """Leave-One-Group-Out CV for all ML models."""
    logo    = LeaveOneGroupOut()
    results = []
    for name, model in models.items():
        if isinstance(model, OptunaRidgeClf):
            model.cv_groups = groups

        fold_scores: List[float] = []
        fold_true:   List[int]   = []
        fold_pred:   List[int]   = []

        for tr, te in logo.split(X, y_enc, groups):
            if isinstance(model, OptunaRidgeClf):
                model.cv_groups = groups[tr]
            model.fit(X[tr], y_enc[tr])
            preds = model.predict(X[te])
            fold_scores.append(accuracy_score(y_enc[te], preds))
            fold_true.extend(y_enc[te].tolist())
            fold_pred.extend(preds.tolist())

        scores = np.array(fold_scores)
        alpha_info = (f"  [best alpha={model.best_alpha_:.2f}]"
                      if isinstance(model, OptunaRidgeClf) else "")
        print(f"\n  [{label}] {name}{alpha_info}")
        print(f"    LOGO: {scores.mean():.3f} ± {scores.std():.3f}  "
              f"per-fold: {[f'{s:.2f}' for s in scores]}")
        print(classification_report(fold_true, fold_pred,
                                    target_names=classes, digits=3))
        results.append({
            "Model":     name,
            "type":      "ML",
            "LOGO_mean": scores.mean(),
            "LOGO_std":  scores.std(),
            "LOGO_min":  scores.min(),
            "LOGO_max":  scores.max(),
            "y_true":    fold_true,
            "y_pred":    fold_pred,
        })
    return pd.DataFrame(results)


def run_gss(
    models: Dict,
    X: np.ndarray,
    y_enc: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    n_splits: int = 10,
) -> pd.DataFrame:
    """GroupShuffleSplit CV."""
    gss     = GroupShuffleSplit(n_splits=n_splits, test_size=0.25,
                                random_state=42)
    results = []
    for name, model in models.items():
        scores:     List[float] = []
        y_true_all: List[int]   = []
        y_pred_all: List[int]   = []
        if isinstance(model, OptunaRidgeClf):
            model.cv_groups = groups
        for tr, te in gss.split(X, y_enc, groups):
            if isinstance(model, OptunaRidgeClf):
                model.cv_groups = groups[tr]
            model.fit(X[tr], y_enc[tr])
            preds = model.predict(X[te])
            scores.append(accuracy_score(y_enc[te], preds))
            y_true_all.extend(y_enc[te].tolist())
            y_pred_all.extend(preds.tolist())
        scores_arr = np.array(scores)
        print(f"  {name}: GSS={scores_arr.mean():.3f}±{scores_arr.std():.3f}")
        print(classification_report(y_true_all, y_pred_all,
                                    target_names=classes, digits=3))
        results.append({
            "Model":    name,
            "GSS_mean": scores_arr.mean(),
            "GSS_std":  scores_arr.std(),
        })
    return pd.DataFrame(results)


def run_sgkf(
    models: Dict,
    X: np.ndarray,
    y_enc: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    n_splits: int = 4,
) -> pd.DataFrame:
    """StratifiedGroupKFold CV."""
    n_splits = min(n_splits, len(np.unique(groups)))
    sgkf     = StratifiedGroupKFold(n_splits=n_splits)
    results:  List[Dict] = []
    for name, model in models.items():
        scores:     List[float] = []
        y_true_all: List[int]   = []
        y_pred_all: List[int]   = []
        if isinstance(model, OptunaRidgeClf):
            model.cv_groups = groups
        for tr, te in sgkf.split(X, y_enc, groups):
            if isinstance(model, OptunaRidgeClf):
                model.cv_groups = groups[tr]
            model.fit(X[tr], y_enc[tr])
            preds = model.predict(X[te])
            scores.append(accuracy_score(y_enc[te], preds))
            y_true_all.extend(y_enc[te].tolist())
            y_pred_all.extend(preds.tolist())
        scores_arr = np.array(scores)
        print(f"  {name}: SGKF={scores_arr.mean():.3f}±{scores_arr.std():.3f}")
        print(classification_report(y_true_all, y_pred_all,
                                    target_names=classes, digits=3))
        results.append({
            "Model":    name,
            "GKF_mean": scores_arr.mean(),
            "GKF_std":  scores_arr.std(),
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CNN LOGO
# ---------------------------------------------------------------------------

def run_logo_cnn(
    maps: List[Dict],
    grid: np.ndarray,
    le: LabelEncoder,
    device,
    args: argparse.Namespace,
) -> Optional[pd.DataFrame]:
    """Leave-One-Group-Out CV for the 2-channel CNN."""
    from Roman_spectre.cnn_model import CNNTrainer, try_import_torch

    torch_mods = try_import_torch()
    if torch_mods[0] is None:
        return None

    torch, nn, optim_mod, DataLoader, TensorDataset = torch_mods
    n_classes = len(le.classes_)
    animals   = sorted(set(m["animal_id"] for m in maps))

    print("  Preprocessing all pixels for CNN (2-channel)...")
    maps_proc: Dict[str, List[Dict]] = {}
    for rec in tqdm(maps, desc="  CNN preprocessing", ncols=80):
        pix_raw      = np.array([np.interp(grid, rec["grid"], px)
                                  for px in rec["pixels"]])
        pix_proc, pix_d2 = preprocess_map_pixels(
            pix_raw, grid, use_als=args.use_als,
            norm=args.norm, n_jobs=args.n_jobs,
        )
        pix_2ch = np.stack([pix_proc, pix_d2], axis=1).astype(np.float32)
        aid = rec["animal_id"]
        if aid not in maps_proc:
            maps_proc[aid] = []
        maps_proc[aid].append({"pixels": pix_2ch, "label": rec["label"]})

    fold_scores: List[float] = []
    fold_true:   List[int]   = []
    fold_pred:   List[int]   = []
    n_grid = len(grid)

    for animal in tqdm(animals, desc="  CNN LOGO folds", ncols=80):
        tr_pix:    List[np.ndarray] = []
        tr_labels: List[int]        = []
        for aid, recs in maps_proc.items():
            if aid == animal:
                continue
            for rec in recs:
                tr_pix.append(rec["pixels"])
                tr_labels.extend(
                    [le.transform([rec["label"]])[0]] * len(rec["pixels"])
                )
        X_tr = np.vstack(tr_pix)
        y_tr = np.array(tr_labels)

        counts = np.bincount(y_tr, minlength=n_classes)
        cw     = (len(y_tr) / (n_classes * counts + 1e-10)).tolist()

        run_optuna_this_fold = (
            fold_scores == [] and args.optuna_trials_cnn > 0
        )
        trainer = CNNTrainer(
            n_grid=n_grid, n_classes=n_classes, device=device,
            epochs=args.cnn_epochs, batch_size=args.cnn_batch,
            lr=args.cnn_lr, weight_decay=args.cnn_weight_decay,
            dropout=args.cnn_dropout,
            patience=args.cnn_patience,
            optuna_n_trials=(args.optuna_trials_cnn
                             if run_optuna_this_fold else 0),
            optuna_epochs=args.optuna_cnn_epochs,
        )
        trainer.fit(X_tr, y_tr, class_weights=cw)

        if run_optuna_this_fold and trainer.best_params:
            args.cnn_lr           = trainer.best_params.get("lr",           args.cnn_lr)
            args.cnn_weight_decay = trainer.best_params.get("weight_decay", args.cnn_weight_decay)
            args.cnn_dropout      = trainer.best_params.get("dropout",      args.cnn_dropout)
            print(f"  📌 Using Optuna params for remaining folds: "
                  f"lr={args.cnn_lr:.2e}  "
                  f"wd={args.cnn_weight_decay:.2e}  "
                  f"dropout={args.cnn_dropout:.3f}")

        te_pix:    List[np.ndarray] = []
        te_labels: List[int]        = []
        for rec in maps_proc[animal]:
            te_pix.append(rec["pixels"])
            te_labels.extend(
                [le.transform([rec["label"]])[0]] * len(rec["pixels"])
            )
        if not te_pix:
            continue

        X_te = np.vstack(te_pix)
        y_te = np.array(te_labels)

        trainer.model.eval()
        preds: List[int] = []
        with torch.no_grad():
            dataset_te = TensorDataset(torch.FloatTensor(X_te))
            loader_te  = DataLoader(dataset_te, batch_size=args.cnn_batch,
                                    shuffle=False)
            for (xb,) in loader_te:
                xb = xb.to(device)
                preds.extend(trainer.model(xb).argmax(dim=1).cpu().tolist())

        fold_scores.append(accuracy_score(y_te, preds))
        fold_true.extend(y_te.tolist())
        fold_pred.extend(preds)

    scores = np.array(fold_scores)
    acc    = scores.mean()
    print(f"\n  [CNN] LOGO: {acc:.3f} ± {scores.std():.3f}")
    print(classification_report(fold_true, fold_pred,
                                 target_names=le.classes_, digits=3))

    return pd.DataFrame([{
        "Model":     "CNN-1D-Res",
        "type":      "CNN",
        "LOGO_mean": acc,
        "LOGO_std":  scores.std(),
        "LOGO_min":  scores.min(),
        "LOGO_max":  scores.max(),
        "y_true":    fold_true,
        "y_pred":    fold_pred,
    }])


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    model,
    X: np.ndarray,
    y_enc: np.ndarray,
    groups: np.ndarray,
    observed_acc: float,
    n_perm: int = 200,
) -> float:
    """
    Non-parametric permutation significance test (LOGO).

    Returns p-value (fraction of permuted scores ≥ observed_acc).
    """
    logo = LeaveOneGroupOut()
    rng  = np.random.RandomState(42)
    perm_scores: List[float] = []

    for _ in tqdm(range(n_perm), desc="  Permuting", ncols=72):
        y_p = rng.permutation(y_enc)
        fs  = []
        for tr, te in logo.split(X, y_p, groups):
            model.fit(X[tr], y_p[tr])
            fs.append(accuracy_score(y_p[te], model.predict(X[te])))
        perm_scores.append(np.mean(fs))

    perm_arr = np.array(perm_scores)
    p_val    = (perm_arr >= observed_acc).mean()
    print(
        f"  Observed={observed_acc:.3f}, "
        f"perm={perm_arr.mean():.3f}±{perm_arr.std():.3f}, "
        f"p={p_val:.3f} "
        f"({'SIGNIFICANT ✅' if p_val < 0.05 else 'NOT significant ❌'})"
    )
    return p_val

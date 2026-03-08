"""
visualisation.py — Plots for training analysis (CV comparison, confusion
matrices, PCA, feature importance, SHAP, CNN saliency).

NOTE: inference-side plots (spectrum prediction, peak analysis, spatial map,
      comparison spectra) live in inference_utils.py to avoid circular imports.

Public API
----------
plot_cv_all(logo_df, gss_df, gkf_df, ...)
plot_confusion_matrix(y_true, y_pred, classes, ...)
plot_feature_importance(model, feat_names, ...)
plot_pca(X, y, aids, ...)
plot_cnn_saliency(trainer, X_sample, y_sample, ...)
plot_shap_ml(model, X, feat_names, ...)
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from Roman_spectre.constants import COLORS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, path: Path, save_plots: bool) -> None:
    if save_plots:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  📊 → {path.name}")
    plt.close("all")


# ---------------------------------------------------------------------------
# CV comparison
# ---------------------------------------------------------------------------

def plot_cv_all(
    logo_df: pd.DataFrame,
    gss_df: Optional[pd.DataFrame],
    gkf_df: Optional[pd.DataFrame],
    out_dir: Path,
    save_plots: bool,
    suffix: str,
) -> None:
    df = logo_df[["Model", "LOGO_mean", "LOGO_std"]].copy()
    if gss_df is not None:
        df = df.merge(gss_df[["Model", "GSS_mean", "GSS_std"]],
                      on="Model", how="left")
    if gkf_df is not None:
        df = df.merge(gkf_df[["Model", "GKF_mean", "GKF_std"]],
                      on="Model", how="left")
    df = df.sort_values("LOGO_mean")

    cols = [
        (c, s, t) for c, s, t in [
            ("LOGO_mean", "LOGO_std", "LOGO (main)"),
            ("GSS_mean",  "GSS_std",  "GroupShuffleSplit"),
            ("GKF_mean",  "GKF_std",  "StratifiedGroupKFold"),
        ] if c in df.columns
    ]

    fig, axes = plt.subplots(
        1, len(cols),
        figsize=(6 * len(cols), max(4, len(df) * 0.65 + 1)),
        sharey=True,
    )
    if len(cols) == 1:
        axes = [axes]
    for ax, (mc, sc, ttl) in zip(axes, cols):
        bars = ax.barh(df["Model"], df[mc], xerr=df.get(sc, 0),
                       color="steelblue", alpha=0.8, capsize=4)
        ax.axvline(1 / 3, ls="--", color="red", lw=1.5, label="random")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy")
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8)
        for bar, m in zip(bars, df[mc]):
            ax.text(m + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{m:.3f}", va="center", fontsize=9)
    plt.suptitle(f"CV Comparison {suffix}", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, out_dir / f"cv_all{suffix}.png", save_plots)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    classes: np.ndarray,
    title: str,
    out_dir: Path,
    save_plots: bool,
) -> None:
    from sklearn.metrics import confusion_matrix
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm,      annot=True, fmt="d",   cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f"{title} (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title(f"{title} (recall per class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.tight_layout()
    _save_fig(fig, out_dir / f"cm_{title.replace(' ', '_')}.png", save_plots)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model,
    feat_names: List[str],
    top_n: int = 30,
    title: str = "",
    out_dir: Optional[Path] = None,
    save_plots: bool = False,
) -> None:
    try:
        clf = (model.named_steps[list(model.named_steps)[-1]]
               if hasattr(model, "named_steps") else model)
        try:
            imp = clf.feature_importances_
        except AttributeError:
            imp = np.abs(clf.coef_).mean(axis=0)
        top_n = min(top_n, len(imp))
        idx   = np.argsort(imp)[-top_n:]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(top_n), imp[idx], color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([str(feat_names[i])[:38] for i in idx], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(title)
        plt.tight_layout()
        _save_fig(fig, out_dir / f"fi_{title.replace(' ', '_')}.png",
                  save_plots)
    except Exception as e:
        print(f"  [WARN] feature importance: {e}")


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def plot_pca(
    X: np.ndarray,
    y: np.ndarray,
    aids: np.ndarray,
    title: str,
    out_dir: Path,
    save_plots: bool,
    fname: str,
) -> None:
    pca = PCA(n_components=min(2, X.shape[1]))
    Xp  = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    for cls in np.unique(y):
        m = y == cls
        axes[0].scatter(
            Xp[m, 0],
            Xp[m, 1] if Xp.shape[1] > 1 else np.zeros(m.sum()),
            c=COLORS.get(cls, "gray"), label=cls, s=60, alpha=0.8,
        )
    axes[0].set_title(
        f"By class (PC1={var[0]:.1%}"
        + (f", PC2={var[1]:.1%}" if len(var) > 1 else "") + ")"
    )
    axes[0].legend()

    cmap = plt.cm.get_cmap("tab20", len(np.unique(aids)))
    for i, a in enumerate(np.unique(aids)):
        m = aids == a
        axes[1].scatter(
            Xp[m, 0],
            Xp[m, 1] if Xp.shape[1] > 1 else np.zeros(m.sum()),
            c=[cmap(i)], label=a, s=60, alpha=0.8,
        )
    axes[1].set_title("By animal")
    axes[1].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    _save_fig(fig, out_dir / fname, save_plots)


# ---------------------------------------------------------------------------
# CNN Saliency
# ---------------------------------------------------------------------------

def plot_cnn_saliency(
    trainer,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    grid: np.ndarray,
    classes: np.ndarray,
    title: str,
    out_dir: Path,
    save_plots: bool,
) -> None:
    from Roman_spectre.cnn_model import try_import_torch
    torch_mods = try_import_torch()
    if torch_mods[0] is None or trainer.model is None:
        return
    torch  = torch_mods[0]
    device = trainer.device
    model  = trainer.model.eval()

    saliency_per_class: Dict[int, np.ndarray] = {}
    for cls_idx in range(len(classes)):
        mask = y_sample == cls_idx
        if mask.sum() == 0:
            continue
        X_cls = torch.FloatTensor(X_sample[mask]).to(device)
        X_cls.requires_grad_(True)
        logits = model(X_cls)
        score  = logits[:, cls_idx].sum()
        score.backward()
        saliency = X_cls.grad.abs().mean(dim=(0, 1)).cpu().detach().numpy()
        saliency_per_class[cls_idx] = saliency

    if not saliency_per_class:
        return

    fig, axes = plt.subplots(
        len(saliency_per_class), 1,
        figsize=(12, 3 * len(saliency_per_class)),
        sharex=True,
    )
    if len(saliency_per_class) == 1:
        axes = [axes]

    for ax, (cls_idx, sal) in zip(axes, saliency_per_class.items()):
        mask    = y_sample == cls_idx
        mean_sp = X_sample[mask, 0, :].mean(axis=0)
        sal_n   = (sal - sal.min()) / (sal.max() - sal.min() + 1e-10)
        ax2 = ax.twinx()
        ax.plot(grid, mean_sp, color="steelblue", lw=1.5,
                label="Mean spectrum", alpha=0.8)
        ax2.fill_between(grid, sal_n, alpha=0.4, color="tomato",
                         label="Saliency")
        ax.set_ylabel("Intensity (norm.)", color="steelblue")
        ax2.set_ylabel("Saliency", color="tomato")
        ax.set_title(f"Saliency — {classes[cls_idx]}", fontsize=10)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    axes[-1].set_xlabel("Raman shift (cm⁻¹)")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_plots:
        fpath = out_dir / f"saliency_{title.replace(' ', '_')}.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"  📊 Saliency → {fpath.name}")
    plt.close("all")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def plot_shap_ml(
    model,
    X: np.ndarray,
    feat_names: List[str],
    classes: np.ndarray,
    title: str,
    out_dir: Path,
    save_plots: bool,
) -> None:
    try:
        import shap
    except ImportError:
        print("  [WARN] shap not installed. pip install shap")
        return

    try:
        clf = (model.named_steps[list(model.named_steps)[-1]]
               if hasattr(model, "named_steps") else model)
        explainer  = shap.TreeExplainer(clf)
        shap_vals  = explainer.shap_values(X)

        if isinstance(shap_vals, list):
            n_cls = len(shap_vals)
        else:
            shap_vals = [shap_vals[:, :, i] for i in range(shap_vals.shape[2])]
            n_cls = len(shap_vals)

        fig, axes = plt.subplots(1, n_cls, figsize=(7 * n_cls, 6))
        if n_cls == 1:
            axes = [axes]

        top_n = min(20, len(feat_names))
        for i, (ax, cls_name) in enumerate(zip(axes, classes)):
            vals = np.abs(shap_vals[i]).mean(axis=0)
            idx  = np.argsort(vals)[-top_n:]
            ax.barh(range(top_n), vals[idx], color="steelblue", alpha=0.8)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(
                [str(feat_names[j])[:35] for j in idx], fontsize=7
            )
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"SHAP — {cls_name}", fontsize=10)

        plt.suptitle(title, fontsize=12)
        plt.tight_layout()
        if save_plots:
            fpath = out_dir / f"shap_{title.replace(' ', '_')}.png"
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            print(f"  📊 SHAP → {fpath.name}")
        plt.close("all")

    except Exception as e:
        print(f"  [WARN] SHAP failed: {e}")

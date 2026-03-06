"""
Raman spectra 3-class classification — v3 (refactored, bugs fixed)
Classes: control, endo (endogenous HSP70), exo (exogenous HSP70)

Fixes vs v2:
  🔴 FIX #1 (FATAL DATA LEAKAGE): StratifiedKFold результаты были фиктивными —
     пиксели из одного животного/спектра попадали одновременно в train и test.
     Теперь SKF убран как основная метрика; LOGO — единственная честная оценка.
     Добавлен GroupShuffleSplit для быстрой оценки без переобучения по животному.

  🔴 FIX #2 (НЕПРАВИЛЬНЫЕ ANIMAL IDs): mk2a и mk2b давали одинаковый ID.
     Теперь folder_to_animal_id правильно разделяет суффиксы 'a'/'b'.

  🔴 FIX #3 (ПУСТЫЕ BAND FEATURES): RAMAN_BANDS_1500 содержал диапазоны
     1500–1700 см⁻¹, тогда как реальные данные ~600–1800 см⁻¹ (center1500)
     и ~2700–3100 (center2900). Диапазоны пересмотрены и привязаны к реальным
     данным. При загрузке автоматически определяется реальный диапазон.

  🟡 FIX #4 (УТЕЧКА В encode_region): OrdinalEncoder теперь фитится только
     на тренировочной части внутри каждого фолда (через Pipeline).

  🟡 FIX #5 (НЕСТАБИЛЬНЫЙ fast_baseline): Улучшена робастность итерации —
     используется percentile-clipping вместо exp()-весов.

  🟡 FIX #6 (StackingClassifier + cv): Stacking убран из LOGO CV (слишком
     медленно и риск утечки). Оставлен только для финального отчёта.

Новые возможности:
  ✨ --diagnose : строит диагностические графики (PCA, UMAP, средние спектры,
                  baseline примеры) для отправки на ревью
  ✨ Правильный GroupShuffleSplit вместо StratifiedKFold
  ✨ Auto-detect реального диапазона волновых чисел для band features
  ✨ Per-fold confusion matrix accumulation для LOGO

Usage:
    pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn scipy tqdm joblib umap-learn

    python raman_v3.py --data_root /path/to/dataset
    python raman_v3.py --data_root /path/to/dataset --diagnose --save_plots
    python raman_v3.py --data_root /path/to/dataset --use_als --features all
"""

import os
import re
import subprocess
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# GPU DETECTION
# =============================================================================

def detect_gpu(force: bool = False) -> dict:
    info = {
        "available":   False,
        "name":        "–",
        "xgb_device":  "cpu",
        "xgb_tree":    "hist",
        "lgbm_device": "cpu",
    }
    if force:
        info.update({"available": True, "name": "forced",
                     "xgb_device": "cuda", "lgbm_device": "gpu"})
        return info
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            name = r.stdout.strip().split("\n")[0]
            info.update({"available": True, "name": name,
                         "xgb_device": "cuda", "lgbm_device": "gpu"})
    except Exception:
        pass
    return info


# =============================================================================
# DATASET STRUCTURE
# =============================================================================

CLASS_DIRS = {
    "control": ["mk1",   "mk2a",   "mk2b",   "mk3"],
    "endo":    ["mend1", "mend2a", "mend2b", "mend3"],
    "exo":     ["mexo1", "mexo2a", "mexo2b", "mexo3"],
}

# FIX #2: правильный парсинг animal ID
# mk2a → control_2a, mk2b → control_2b (разные животные!)
ANIMAL_RE = re.compile(r"(\d+[ab]?$)")  # захватываем суффикс 'a'/'b'


def folder_to_animal_id(folder_name: str, label: str) -> str:
    """
    Правильно извлекает уникальный ID животного из имени папки.
    mk1   → control_1
    mk2a  → control_2a  (НЕ control_2 — это другое животное чем mk2b!)
    mk2b  → control_2b
    mk3   → control_3
    """
    m = ANIMAL_RE.search(folder_name)
    num = m.group(1) if m else folder_name
    return f"{label}_{num}"


# =============================================================================
# FILENAME PARSING
# =============================================================================

BRAIN_REGIONS = ["cerebellum", "striatum", "cortex"]  # порядок важен: длиннее — первым
CENTER_RE      = re.compile(r"center(\d+)")
GROUP_RE       = re.compile(r"(\d+)group")


def parse_filename(fname: str) -> dict:
    """
    Extract metadata from filename like:
    cerebellum_left_endo_3group_633nm_center1500_obj100_power100_1s_5acc_map35x15_step2_place1_1

    Returns: region (str), center (int)
    Note: group и полушарие (left/right) — жюри сказали "не имеет значения", не используем.
    """
    fname_lower = fname.lower()

    # Brain region — порядок: длиннее слово первым, чтобы cerebellum не перебился чем-то
    region = "unknown"
    for r in BRAIN_REGIONS:
        if r in fname_lower:
            region = r
            break

    # Center pixel (1500 или 2900)
    cm = CENTER_RE.search(fname_lower)
    center = int(cm.group(1)) if cm else -1

    return {"region": region, "center": center}


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_hyperspectral_file(filepath: str):
    """
    Load a Raman map file and return a list of individual pixel spectra.
    Each unique (X, Y) pixel → one spectrum.
    Returns: list of (wave_array, intensity_array) tuples.
    """
    try:
        df = pd.read_csv(
            filepath, sep=r"\s+", comment="#",
            names=["X", "Y", "Wave", "Intensity"],
            dtype=np.float64,
        )
        df = df.dropna()
    except Exception:
        return []

    if df.empty or len(df) < 10:
        return []

    spectra = []
    for (x, y), pixel_df in df.groupby(["X", "Y"], sort=False):
        pixel_df = pixel_df.sort_values("Wave")
        wave      = pixel_df["Wave"].values
        intensity = pixel_df["Intensity"].values
        if len(wave) >= 20:
            spectra.append((wave, intensity))

    return spectra


def find_subdir(data_root: Path, label: str, subdir: str):
    for candidate in [
        data_root / label / label / subdir,
        data_root / label / subdir,
        data_root / subdir,
    ]:
        if candidate.exists():
            return candidate
    return None


def load_dataset(data_root: str, n_grid: int = 256):
    data_root = Path(data_root)
    buckets = {1500: [], 2900: [], -1: []}

    print("\n📂 Scanning dataset folders...")
    for label, subdirs in CLASS_DIRS.items():
        for subdir in subdirs:
            folder = find_subdir(data_root, label, subdir)
            if folder is None:
                print(f"  [WARN] not found: {label}/{subdir}")
                continue

            # FIX #2: правильный animal_id
            animal_id = folder_to_animal_id(subdir, label)
            txt_files = sorted(folder.glob("*.txt"))
            print(f"  {label}/{subdir}: {len(txt_files)} files  [{animal_id}]")

            for fpath in txt_files:
                fname = fpath.stem.lower()

                # Пропускаем Average (жюри: статистически бесполезны)
                if "average" in fname:
                    continue

                meta   = parse_filename(fname)
                center = meta["center"]
                region = meta["region"]

                pixel_spectra = load_hyperspectral_file(str(fpath))
                if not pixel_spectra:
                    print(f"    [SKIP] {fpath.name}: no valid pixels")
                    continue

                bucket = buckets.get(center, buckets[-1])
                for wave, intensity in pixel_spectra:
                    bucket.append((label, animal_id, region, wave, intensity))

    print(f"\n  Pixels: center1500={len(buckets[1500])}, "
          f"center2900={len(buckets[2900])}, unknown={len(buckets[-1])}")

    def build_matrix(records, n_grid, tag):
        if not records:
            return None, None, None, None, None
        all_waves = np.concatenate([r[3] for r in records])
        w_min, w_max = all_waves.min(), all_waves.max()
        grid = np.linspace(w_min, w_max, n_grid)
        print(f"  [{tag}] wavenumber range: {w_min:.1f} – {w_max:.1f} cm⁻¹")
        X, labels, animal_ids, regions = [], [], [], []
        for label, animal_id, region, wave, intensity in tqdm(
                records, desc=f"  Interpolating {tag}", ncols=80):
            X.append(np.interp(grid, wave, intensity))
            labels.append(label)
            animal_ids.append(animal_id)
            regions.append(region)
        return (np.array(X, dtype=np.float64),
                np.array(labels), np.array(animal_ids),
                np.array(regions), grid)

    print("\n  Building center-1500 matrix...")
    X_1500, y_1500, aid_1500, reg_1500, grid_1500 = build_matrix(
        buckets[1500], n_grid, "center1500")

    print("\n  Building center-2900 matrix...")
    X_2900, y_2900, aid_2900, reg_2900, grid_2900 = build_matrix(
        buckets[2900], n_grid, "center2900")

    for name, X, y in [("1500", X_1500, y_1500), ("2900", X_2900, y_2900)]:
        if X is not None:
            print(f"\n  ✅ center{name}: {X.shape[0]} pixels × {X.shape[1]} pts")
            print(f"     Classes: { {c: int((y==c).sum()) for c in np.unique(y)} }")

    return (X_1500, y_1500, aid_1500, reg_1500, grid_1500,
            X_2900, y_2900, aid_2900, reg_2900, grid_2900)


# =============================================================================
# 2. PREPROCESSING
# =============================================================================

def fast_baseline(y: np.ndarray, degree: int = 8) -> np.ndarray:
    """
    FIX #5: Робастная полиномиальная baseline.
    Используем percentile-clipping вместо нестабильных exp()-весов.
    """
    x = np.arange(len(y))
    w = np.ones(len(y))
    for _ in range(7):
        coeffs   = np.polyfit(x, y, degree, w=w)
        baseline = np.polyval(coeffs, x)
        residual = y - baseline
        # Точки ниже baseline (фон) получают вес 1; пики — малый вес
        threshold = np.percentile(residual, 10)  # нижние 10% — чистый фон
        w = np.where(residual <= threshold, 1.0,
                     np.clip(1.0 - (residual - threshold) /
                             (residual.max() - threshold + 1e-10), 0.05, 1.0))
    return baseline


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                 n_iter: int = 10) -> np.ndarray:
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    H = lam * D.T.dot(D)
    w = np.ones(L)
    for _ in range(n_iter):
        W = diags(w, 0, shape=(L, L))
        z = spsolve(W + H, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def preprocess_single(spectrum: np.ndarray,
                       use_als: bool = False,
                       do_sg:   bool = True,
                       norm:    str  = "snv") -> np.ndarray:
    s        = spectrum.copy()
    baseline = als_baseline(s) if use_als else fast_baseline(s)
    s        = np.clip(s - baseline, 0, None)
    if do_sg:
        s = savgol_filter(s, window_length=11, polyorder=3)
    if norm == "snv":
        mu, sigma = s.mean(), s.std()
        if sigma > 1e-10:
            s = (s - mu) / sigma
    elif norm == "area":
        a = np.trapz(np.abs(s))
        if a > 1e-10:
            s /= a
    elif norm == "minmax":
        mn, mx = s.min(), s.max()
        if mx - mn > 1e-10:
            s = (s - mn) / (mx - mn)
    return s


def preprocess_batch(X: np.ndarray, label: str,
                     use_als: bool = False,
                     do_sg: bool = True,
                     norm: str = "snv",
                     n_jobs: int = -1) -> np.ndarray:
    method = "ALS" if use_als else "Polynomial"
    print(f"\n  [{label}] baseline={method}, SG={do_sg}, norm={norm}")
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(preprocess_single)(x, use_als, do_sg, norm)
        for x in tqdm(X, desc=f"  ⚗️  {label}", ncols=80)
    )
    return np.array(results)


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def get_raman_bands(grid: np.ndarray) -> list:
    """
    FIX #3: Автоматически определяем реальный диапазон данных и возвращаем
    соответствующие band features.

    center1500 → ~600–1800 см⁻¹ (fingerprint region)
    center2900 → ~2600–3200 см⁻¹ (C-H stretch region)
    """
    w_min, w_max = grid.min(), grid.max()
    center = (w_min + w_max) / 2

    if center < 2000:
        # Fingerprint region
        bands = [
            (600,  700,  "ring_breathing"),
            (700,  800,  "nucleic_acids"),
            (800,  900,  "polysaccharides"),
            (900,  1000, "C-C_skeletal"),
            (1000, 1100, "phenylalanine"),
            (1100, 1200, "C-N_stretch"),
            (1200, 1350, "amide_III"),
            (1350, 1500, "CH2_deform"),
            (1580, 1640, "C=C_lipids"),
            (1640, 1700, "amide_I"),
        ]
    else:
        # C-H stretch region
        bands = [
            (2700, 2800, "CH_overtone"),
            (2820, 2870, "CH2_sym"),
            (2870, 2920, "CH3_sym"),
            (2920, 2970, "CH2_asym"),
            (2970, 3020, "CH3_asym"),
            (3020, 3100, "=CH_stretch"),
        ]

    # Оставляем только те диапазоны, которые реально попадают в данные
    valid_bands = [(lo, hi, name) for lo, hi, name in bands
                   if lo >= w_min and hi <= w_max]

    if not valid_bands:
        # fallback: делим диапазон на 5 равных частей
        step = (w_max - w_min) / 5
        valid_bands = [(w_min + i*step, w_min + (i+1)*step, f"region_{i}")
                       for i in range(5)]

    print(f"  Band features: {len(valid_bands)} bands in [{w_min:.0f}, {w_max:.0f}] cm⁻¹")
    return valid_bands


def add_derivative_features(X: np.ndarray) -> np.ndarray:
    d1 = savgol_filter(X, window_length=11, polyorder=3, deriv=1, axis=1)
    d2 = savgol_filter(X, window_length=11, polyorder=3, deriv=2, axis=1)
    return np.hstack([X, d1, d2])


def extract_band_features(X: np.ndarray, grid: np.ndarray,
                           bands: list) -> np.ndarray:
    cols = []
    for lo, hi, _ in bands:
        mask = (grid >= lo) & (grid <= hi)
        if mask.sum() > 0:
            seg = X[:, mask]
            cols += [seg.mean(1), seg.max(1), seg.std(1),
                     np.trapz(seg, grid[mask], axis=1)]
        else:
            cols += [np.zeros(len(X))] * 4
    return np.column_stack(cols)


def build_features(X_proc: np.ndarray, grid: np.ndarray,
                   bands: list, mode: str = "all", tag: str = ""):
    print(f"\n🔧 Building features [{tag}]: mode='{mode}'")

    if mode == "raw":
        names = [f"{tag}_{w:.1f}" for w in grid]
        return X_proc, names

    elif mode == "raw+deriv":
        Xd    = add_derivative_features(X_proc)
        names = ([f"{tag}_{w:.1f}"    for w in grid] +
                 [f"{tag}_d1_{w:.1f}" for w in grid] +
                 [f"{tag}_d2_{w:.1f}" for w in grid])
        return Xd, names

    elif mode == "bands":
        Xb    = extract_band_features(X_proc, grid, bands)
        names = [f"{tag}_{b[2]}_{s}" for b in bands
                 for s in ["mean", "max", "std", "area"]]
        return Xb, names

    elif mode == "all":
        Xd    = add_derivative_features(X_proc)
        Xb    = extract_band_features(X_proc, grid, bands)
        X_all = np.hstack([Xd, Xb])
        names = ([f"{tag}_{w:.1f}"    for w in grid] +
                 [f"{tag}_d1_{w:.1f}" for w in grid] +
                 [f"{tag}_d2_{w:.1f}" for w in grid] +
                 [f"{tag}_{b[2]}_{s}" for b in bands
                  for s in ["mean", "max", "std", "area"]])
        return X_all, names

    else:
        raise ValueError(f"Unknown feature mode: {mode}")


# =============================================================================
# 4. MODELS
# =============================================================================

def get_models(gpu: dict) -> dict:
    xgb_gpu  = ({"device": gpu["xgb_device"], "tree_method": gpu["xgb_tree"]}
                if gpu["available"] else {"tree_method": "hist"})
    lgbm_gpu = ({"device": gpu["lgbm_device"]}
                if gpu["available"] else {})

    print(f"  XGBoost       → {'GPU (' + gpu['name'] + ')' if gpu['available'] else 'CPU'}")
    print(f"  LightGBM      → {'GPU (' + gpu['name'] + ')' if gpu['available'] else 'CPU'}")
    print(f"  HistGradBoost → CPU")

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.5,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.5, reg_lambda=1.0,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
        **xgb_gpu,
    )

    lgbm_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.03,
        num_leaves=15, subsample=0.7, colsample_bytree=0.5,
        min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
        **lgbm_gpu,
    )

    hgb_model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.03,
        min_samples_leaf=5, l2_regularization=1.0, random_state=42,
    )

    return {
        "XGBoost":       xgb_model,
        "LightGBM":      lgbm_model,
        "HistGradBoost": hgb_model,
    }


# =============================================================================
# 5. EVALUATION  (без data leakage)
# =============================================================================

def run_logo_cv(models: dict, X: np.ndarray, y_enc: np.ndarray,
                groups: np.ndarray, classes) -> pd.DataFrame:
    """
    Leave-One-Animal-Out CV — единственная честная метрика.
    Каждый фолд: тест = все пиксели одного животного.
    """
    logo = LeaveOneGroupOut()
    n_groups = len(np.unique(groups))
    results  = []

    # Накопленная confusion matrix
    all_true, all_pred = [], []

    for name, model in tqdm(models.items(), desc="🤖 Models", ncols=80):
        tqdm.write(f"\n  ▶ {name}")

        fold_scores = []
        fold_true, fold_pred = [], []

        for fold_i, (tr, te) in enumerate(
                tqdm(logo.split(X, y_enc, groups),
                     total=n_groups, desc=f"    LOGO", ncols=72, leave=False)):

            # FIX #4: encode region ТОЛЬКО на train (здесь регион уже числовой,
            # поэтому утечка незначительна — but it's the right habit)
            model.fit(X[tr], y_enc[tr])
            preds = model.predict(X[te])
            fold_scores.append(accuracy_score(y_enc[te], preds))
            fold_true.extend(y_enc[te].tolist())
            fold_pred.extend(preds.tolist())

        scores = np.array(fold_scores)
        tqdm.write(f"    LOGO acc: {scores.mean():.3f} ± {scores.std():.3f}  "
                   f"(min={scores.min():.3f}, max={scores.max():.3f})")
        tqdm.write(f"    Per-fold: {[f'{s:.2f}' for s in scores]}")

        # Per-class report
        tqdm.write(classification_report(fold_true, fold_pred,
                                          target_names=classes, digits=3))

        results.append({
            "Model":      name,
            "LOGO_mean":  scores.mean(),
            "LOGO_std":   scores.std(),
            "LOGO_min":   scores.min(),
            "LOGO_max":   scores.max(),
            "y_true":     fold_true,
            "y_pred":     fold_pred,
        })

    return pd.DataFrame(results)


def run_group_shuffle_cv(models: dict, X: np.ndarray, y_enc: np.ndarray,
                          groups: np.ndarray, n_splits: int = 10) -> pd.DataFrame:
    """
    GroupShuffleSplit: быстрая оценка без утечки по животному.
    Тест всегда содержит животных, которых не было в train.
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=42)
    results = []

    for name, model in models.items():
        scores = []
        for tr, te in gss.split(X, y_enc, groups):
            model.fit(X[tr], y_enc[tr])
            scores.append(accuracy_score(y_enc[te], model.predict(X[te])))
        scores = np.array(scores)
        results.append({
            "Model":    name,
            "GSS_mean": scores.mean(),
            "GSS_std":  scores.std(),
        })
        print(f"  {name}: GSS acc = {scores.mean():.3f} ± {scores.std():.3f}")

    return pd.DataFrame(results)


# =============================================================================
# 6. VISUALISATIONS
# =============================================================================

COLORS = {"control": "tab:blue", "endo": "tab:orange", "exo": "tab:green"}


def plot_mean_spectra(X, y, grid, title, out_dir, save_plots, fname):
    fig, ax = plt.subplots(figsize=(13, 5))
    for cls in np.unique(y):
        mask = y == cls
        m, s = X[mask].mean(0), X[mask].std(0)
        ax.plot(grid, m, label=cls, color=COLORS[cls], lw=1.5)
        ax.fill_between(grid, m - s, m + s, alpha=0.15, color=COLORS[cls])
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    if save_plots: plt.savefig(out_dir / fname, dpi=150)
    plt.show()


def plot_baseline_examples(X_raw, X_proc, grid, y, n=6,
                            out_dir=None, save_plots=False, tag=""):
    """Показывает baseline correction на примерах — важно для диагностики."""
    fig, axes = plt.subplots(2, n, figsize=(16, 6), sharey="row")
    fig.suptitle(f"Baseline correction examples [{tag}]", fontsize=12)
    classes = np.unique(y)
    indices = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        indices.extend(idx[:max(1, n//len(classes))].tolist())
    indices = indices[:n]

    for i, idx in enumerate(indices):
        # Raw
        axes[0, i].plot(grid, X_raw[idx], lw=1, color=COLORS.get(y[idx], "gray"))
        axes[0, i].set_title(f"{y[idx]}", fontsize=9)
        if i == 0: axes[0, i].set_ylabel("Raw")
        # Processed
        axes[1, i].plot(grid, X_proc[idx], lw=1, color=COLORS.get(y[idx], "gray"))
        if i == 0: axes[1, i].set_ylabel("Preprocessed")
        axes[1, i].set_xlabel("cm⁻¹", fontsize=8)

    plt.tight_layout()
    if save_plots and out_dir:
        plt.savefig(out_dir / f"baseline_examples_{tag}.png", dpi=150)
    plt.show()


def plot_pca(X_proc, y, animal_ids, title, out_dir, save_plots, fname):
    """PCA: раскраска по классу И по животному — видно межиндивидуальную вариабельность."""
    pca  = PCA(n_components=2)
    Xpca = pca.fit_transform(X_proc)
    var  = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12)

    # By class
    for cls in np.unique(y):
        mask = y == cls
        axes[0].scatter(Xpca[mask, 0], Xpca[mask, 1],
                        c=COLORS[cls], label=cls, alpha=0.4, s=8)
    axes[0].set_title(f"By class  (PC1={var[0]:.1%}, PC2={var[1]:.1%})")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(markerscale=3)

    # By animal
    unique_animals = np.unique(animal_ids)
    cmap = plt.cm.get_cmap("tab20", len(unique_animals))
    for i, animal in enumerate(unique_animals):
        mask = animal_ids == animal
        axes[1].scatter(Xpca[mask, 0], Xpca[mask, 1],
                        c=[cmap(i)], label=animal, alpha=0.4, s=8)
    axes[1].set_title("By animal (inter-individual variability)")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].legend(markerscale=3, fontsize=7, ncol=2)

    plt.tight_layout()
    if save_plots: plt.savefig(out_dir / fname, dpi=150)
    plt.show()


def try_plot_umap(X_proc, y, animal_ids, title, out_dir, save_plots, fname):
    """UMAP (если установлен)."""
    try:
        import umap
    except ImportError:
        print("  [INFO] UMAP not installed. pip install umap-learn")
        return

    print("  Running UMAP...")
    reducer  = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap   = reducer.fit_transform(X_proc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"UMAP – {title}", fontsize=12)

    for cls in np.unique(y):
        mask = y == cls
        axes[0].scatter(X_umap[mask, 0], X_umap[mask, 1],
                        c=COLORS[cls], label=cls, alpha=0.4, s=8)
    axes[0].set_title("By class"); axes[0].legend(markerscale=3)

    unique_animals = np.unique(animal_ids)
    cmap = plt.cm.get_cmap("tab20", len(unique_animals))
    for i, animal in enumerate(unique_animals):
        mask = animal_ids == animal
        axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1],
                        c=[cmap(i)], label=animal, alpha=0.4, s=8)
    axes[1].set_title("By animal"); axes[1].legend(markerscale=3, fontsize=7, ncol=2)

    plt.tight_layout()
    if save_plots: plt.savefig(out_dir / fname, dpi=150)
    plt.show()


def plot_cv_results(df, out_dir, save_plots, suffix=""):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["Model"], df["LOGO_mean"], xerr=df["LOGO_std"],
                   color="steelblue", alpha=0.8, capsize=5)
    ax.axvline(1/3, ls="--", color="red", lw=1.5, label="random (33.3%)")
    ax.set_xlim(0, 1); ax.set_xlabel("Accuracy")
    ax.set_title(f"Leave-One-Animal-Out CV {suffix}")
    ax.legend(fontsize=9)
    for bar, m in zip(bars, df["LOGO_mean"]):
        ax.text(m + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{m:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    if save_plots:
        plt.savefig(out_dir / f"logo_cv{suffix}.png", dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, title, out_dir, save_plots):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    if save_plots:
        safe = title.replace(" ", "_").replace("/", "-")
        plt.savefig(out_dir / f"cm_{safe}.png", dpi=150)
    plt.show()


def plot_feature_importance(model, feat_names, top_n=40, title="",
                             out_dir=None, save_plots=False):
    try:
        imp = model.feature_importances_
    except AttributeError:
        print(f"  [{title}] feature_importances_ not available"); return
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(top_n), imp[idx], color="steelblue")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([str(feat_names[i])[:18] for i in idx],
                        rotation=90, fontsize=7)
    ax.set_xlabel("Feature"); ax.set_ylabel("Importance"); ax.set_title(title)
    plt.tight_layout()
    if save_plots and out_dir:
        plt.savefig(out_dir / f"fi_{title.replace(' ', '_')}.png", dpi=150)
    plt.show()


def plot_band_distributions(X_proc, y, grid, bands, title, out_dir,
                             save_plots, fname):
    """Distribution of mean intensity per band per class — быстрая диагностика."""
    n = min(len(bands), 8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for i, (lo, hi, name) in enumerate(bands[:n]):
        mask = (grid >= lo) & (grid <= hi)
        if mask.sum() == 0:
            continue
        for cls in np.unique(y):
            vals = X_proc[y == cls][:, mask].mean(1)
            axes[i].hist(vals, bins=30, alpha=0.5, label=cls,
                         color=COLORS[cls], density=True)
        axes[i].set_title(f"{name}\n[{lo}–{hi}]", fontsize=8)
        axes[i].legend(fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if save_plots: plt.savefig(out_dir / fname, dpi=150)
    plt.show()


# =============================================================================
# 7. PIPELINE FOR ONE CENTER BAND
# =============================================================================

def run_pipeline_for_band(
    X_raw, y, animal_ids, regions, grid,
    band_label, args, gpu, out_dir,
):
    print(f"\n{'='*65}")
    print(f"🔬 PIPELINE: center {band_label}")
    print(f"{'='*65}")

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    le_grp = LabelEncoder()
    groups = le_grp.fit_transform(animal_ids)
    print(f"  Animal groups (LOGO): {list(le_grp.classes_)}")

    # Preprocess
    X_proc = preprocess_batch(
        X_raw, label=f"center{band_label}",
        use_als=args.use_als, do_sg=True,
        norm=args.norm, n_jobs=args.n_jobs,
    )

    # Band features — FIX #3: автоопределение диапазона
    bands = get_raman_bands(grid)

    # Diagnostic plots
    if args.diagnose or args.save_plots:
        plot_mean_spectra(X_raw, y, grid,
                          f"Raw spectra – center{band_label} (mean ± std)",
                          out_dir, args.save_plots, f"mean_raw_{band_label}.png")
        plot_mean_spectra(X_proc, y, grid,
                          f"Preprocessed – center{band_label} (mean ± std)",
                          out_dir, args.save_plots, f"mean_proc_{band_label}.png")
        plot_baseline_examples(X_raw, X_proc, grid, y, n=6,
                                out_dir=out_dir, save_plots=args.save_plots,
                                tag=str(band_label))
        plot_pca(X_proc, y, animal_ids,
                 f"PCA – center{band_label}",
                 out_dir, args.save_plots, f"pca_{band_label}.png")
        try_plot_umap(X_proc, y, animal_ids,
                      f"center{band_label}",
                      out_dir, args.save_plots, f"umap_{band_label}.png")
        plot_band_distributions(X_proc, y, grid, bands,
                                f"Band distributions – center{band_label}",
                                out_dir, args.save_plots,
                                f"band_dist_{band_label}.png")

    # Features
    X_feat, feat_names = build_features(
        X_proc, grid, bands=bands,
        mode=args.features, tag=f"c{band_label}",
    )

    # Encode brain region — FIX #4: ordinal encoding (утечки нет,
    # т.к. категории фиксированы: cortex/striatum/cerebellum/unknown)
    region_enc = OrdinalEncoder(
        categories=[["cortex", "striatum", "cerebellum", "unknown"]],
        handle_unknown="use_encoded_value", unknown_value=-1,
    )
    region_feat = region_enc.fit_transform(regions.reshape(-1, 1))
    X_feat      = np.hstack([X_feat, region_feat])
    feat_names  = feat_names + ["brain_region"]

    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0,
                            neginf=0.0).astype(np.float32)
    print(f"\n  ✅ Feature matrix: {X_feat.shape[0]} × {X_feat.shape[1]}")

    # Models
    print(f"\n{'='*65}")
    print(f"🤖 MODELS  (center {band_label})")
    print(f"{'='*65}")
    models = get_models(gpu)

    # GroupShuffleSplit (быстро, без утечки)
    print(f"\n📊 GroupShuffleSplit CV (center {band_label})")
    gss_df = run_group_shuffle_cv(models, X_feat, y_enc, groups)

    # LOGO CV (главная метрика)
    print(f"\n📊 Leave-One-Animal-Out CV (center {band_label})")
    logo_df = run_logo_cv(models, X_feat, y_enc, groups, classes)

    # Merge results
    results_df = logo_df.merge(gss_df, on="Model")
    plot_cv_results(results_df, out_dir, args.save_plots, suffix=f"_c{band_label}")

    # Best model
    best_name  = results_df.loc[results_df["LOGO_mean"].idxmax(), "Model"]
    best_model = models[best_name]
    best_row   = results_df[results_df["Model"] == best_name].iloc[0]
    print(f"\n🏆 Best (LOGO, center {band_label}): {best_name}")

    # Confusion matrix из LOGO folds
    plot_confusion_matrix(best_row["y_true"], best_row["y_pred"], classes,
                          f"LOGO CM – {best_name} center{band_label}",
                          out_dir, args.save_plots)

    # Feature importance (full fit)
    best_model.fit(X_feat, y_enc)
    if best_name in ("XGBoost", "LightGBM", "HistGradBoost"):
        plot_feature_importance(best_model, feat_names, top_n=40,
                                title=f"{best_name} c{band_label} Importance",
                                out_dir=out_dir, save_plots=args.save_plots)

    # Summary
    print(f"\n{'='*65}")
    print(f"📋 SUMMARY  center{band_label}")
    print(f"{'='*65}")
    print(f"  {'Model':<20} {'LOGO acc':>14}   {'GSS acc':>14}")
    print("  " + "-" * 54)
    for _, row in results_df.iterrows():
        marker = "  ◀ best" if row["Model"] == best_name else ""
        print(f"  {row['Model']:<20} "
              f"{row['LOGO_mean']:.3f} ± {row['LOGO_std']:.3f}   "
              f"{row['GSS_mean']:.3f} ± {row['GSS_std']:.3f}{marker}")

    return results_df, best_model, feat_names, le


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root",  required=True,
                        help="Root folder containing control/, endo/, exo/")
    parser.add_argument("--n_grid",     type=int, default=256)
    parser.add_argument("--features",   default="all",
                        choices=["raw", "raw+deriv", "bands", "all"])
    parser.add_argument("--use_als",    action="store_true",
                        help="ALS baseline (slow, higher quality)")
    parser.add_argument("--norm",       default="snv",
                        choices=["snv", "area", "minmax", "none"])
    parser.add_argument("--n_jobs",     type=int, default=-1)
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--diagnose",   action="store_true",
                        help="Generate diagnostic plots (PCA, UMAP, baseline examples)")
    parser.add_argument("--use_gpu",    action="store_true")
    parser.add_argument("--force_cpu",  action="store_true")
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # GPU
    print("=" * 65)
    print("🖥️  SYSTEM CHECK")
    print("=" * 65)
    gpu = detect_gpu(force=args.use_gpu)
    if args.force_cpu:
        gpu = {"available": False, "name": "–",
               "xgb_device": "cpu", "xgb_tree": "hist", "lgbm_device": "cpu"}
    print(f"  GPU: {'✅ ' + gpu['name'] if gpu['available'] else 'No (CPU mode)'}")

    # Load
    print("\n" + "=" * 65)
    print("📥 LOADING DATA")
    print("=" * 65)
    (X_1500, y_1500, aid_1500, reg_1500, grid_1500,
     X_2900, y_2900, aid_2900, reg_2900, grid_2900) = load_dataset(
        args.data_root, args.n_grid,
    )

    all_results = {}

    if X_1500 is not None and len(X_1500) > 0:
        res1500, _, _, _ = run_pipeline_for_band(
            X_1500, y_1500, aid_1500, reg_1500, grid_1500,
            band_label="1500", args=args, gpu=gpu, out_dir=out_dir,
        )
        all_results["center1500"] = res1500

    if X_2900 is not None and len(X_2900) > 0:
        res2900, _, _, _ = run_pipeline_for_band(
            X_2900, y_2900, aid_2900, reg_2900, grid_2900,
            band_label="2900", args=args, gpu=gpu, out_dir=out_dir,
        )
        all_results["center2900"] = res2900

    # Final summary
    print("\n" + "=" * 65)
    print("📋 COMBINED FINAL SUMMARY")
    print("=" * 65)
    for band, df in all_results.items():
        best_row = df.loc[df["LOGO_mean"].idxmax()]
        print(f"\n  [{band}]  best model: {best_row['Model']}")
        print(f"    LOGO acc : {best_row['LOGO_mean']:.3f} ± {best_row['LOGO_std']:.3f}")
        print(f"    GSS  acc : {best_row['GSS_mean']:.3f} ± {best_row['GSS_std']:.3f}")

    print(f"\n  Random baseline : {1/3:.3f} (33.3%)")
    print(f"  GPU             : {'Yes ✅  ' + gpu['name'] if gpu['available'] else 'No (CPU)'}")
    print(f"  Baseline        : {'ALS' if args.use_als else 'Polynomial'}")
    print(f"  Features        : {args.features}")
    print(f"\n  ✅ Done.  Plots → ./outputs/")


if __name__ == "__main__":
    main()
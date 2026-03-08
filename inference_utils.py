"""
inference_utils.py  —  Backend utilities for Raman v10
=======================================================
Совместим с моделями, обученными raman_v10.py:
  • ML-модели  → best_model_center*.pkl  /  best_model_fused.pkl
  • CNN-модели → cnn_weights_center*.pt  +  cnn_meta_center*.pkl

Предоставляет:
  1. RamanMLPredictor    — инференс ML-модели для одиночного спектра
  2. RamanCNNPredictor   — инференс CNN-модели для одиночного спектра
  3. RamanEnsemble       — ансамбль ML + CNN (или нескольких центров)
  4. plot_spectrum_prediction()  — график спектра с аннотацией пиков и
                                   баром вероятностей классов
  5. plot_peak_analysis()        — детальный разбор полос (обратная задача):
                                   важность каждой полосы по классам
  6. plot_spatial_map()          — карта предсказаний по пикселям (x, y)
  7. plot_comparison_spectra()   — средние спектры по классам + маркеры пиков

Зависимости:
  pip install numpy scipy scikit-learn joblib matplotlib torch
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from ml_models import OptunaRidgeClf
import numpy as np
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Цвета классов (единый стиль для всех графиков)
# ---------------------------------------------------------------------------
CLASS_COLORS: Dict[str, str] = {
    "control": "#2196F3",   # синий
    "endo":    "#FF9800",   # оранжевый
    "exo":     "#4CAF50",   # зелёный
}
CLASS_COLORS_RGBA: Dict[str, Tuple] = {
    "control": (0.13, 0.59, 0.95, 1.0),
    "endo":    (1.00, 0.60, 0.00, 1.0),
    "exo":     (0.30, 0.69, 0.31, 1.0),
}

# ---------------------------------------------------------------------------
# Биохимические аннотации полос (отображаются на графиках)
# ---------------------------------------------------------------------------
BAND_ANNOTATIONS_1500: Dict[str, Dict] = {
    "C-C_skel":  {"center": 950,  "label": "C–C\nskel",   "bio": "Phospholipids"},
    "phe":       {"center": 1062, "label": "Phe\n1062",   "bio": "Phospholipids"},
    "CN_str":    {"center": 1128, "label": "C–N\n1128",   "bio": "Valence C–C"},
    "amide_III": {"center": 1294, "label": "Amide III\n1294", "bio": "Major protein groups"},
    "CH2_def":   {"center": 1440, "label": "CH₂\n1440",   "bio": "Lipids / Nucleic acids"},
    "CC_lip":    {"center": 1610, "label": "C=C\nlip",    "bio": "Lipids"},
    "amide_I":   {"center": 1670, "label": "Amide I\n1670", "bio": "Proteins"},
}
BAND_ANNOTATIONS_2900: Dict[str, Dict] = {
    "CH_ov":   {"center": 2750, "label": "CH\nov",     "bio": "CH overtone"},
    "CH2_sym": {"center": 2850, "label": "CH₂\n2850", "bio": "Symmetric CH₂"},
    "CH3_sym": {"center": 2895, "label": "CH₃\n2895", "bio": "Symmetric CH₃"},
    "CH2_asy": {"center": 2940, "label": "CH₂\n2940", "bio": "Asymmetric CH₂"},
    "CH3_asy": {"center": 2990, "label": "CH₃\n2990", "bio": "Asymmetric CH₃"},
    "CH_str":  {"center": 3060, "label": "CH\nstr",   "bio": "CH stretch"},
}


def _band_annotations_for_grid(grid: np.ndarray) -> Dict[str, Dict]:
    center = (grid.min() + grid.max()) / 2
    return BAND_ANNOTATIONS_1500 if center < 2000 else BAND_ANNOTATIONS_2900


# =============================================================================
# PREPROCESSING  (копия из raman_v10, чтобы utils был автономным)
# =============================================================================

def fast_baseline(y: np.ndarray, degree: int = 6) -> np.ndarray:
    x = np.arange(len(y))
    w = np.ones(len(y))
    for _ in range(7):
        c   = np.polyfit(x, y, degree, w=w)
        bl  = np.polyval(c, x)
        res = y - bl
        thr = np.percentile(res, 15)
        rng = max(res.max() - thr, 1e-10)
        w   = np.where(res <= thr, 1.0,
                       np.clip(1.0 - (res - thr) / rng, 0.05, 1.0))
    return bl


def als_baseline(
    y: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10
) -> np.ndarray:
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    H = lam * D.T.dot(D)
    w = np.ones(L)
    for _ in range(n_iter):
        W = diags(w, 0, shape=(L, L))
        z = spsolve(W + H, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def preprocess_spectrum(
    s: np.ndarray,
    grid: np.ndarray,
    use_als: bool = False,
    norm: str = "snv",
) -> Tuple[np.ndarray, np.ndarray]:
    """Возвращает (preprocessed_spectrum, second_derivative)."""
    s  = s.copy()
    bl = als_baseline(s) if use_als else fast_baseline(s)
    s  = np.clip(s - bl, 0, None)
    s  = savgol_filter(s, window_length=11, polyorder=3)
    d2 = savgol_filter(s, window_length=11, polyorder=3, deriv=2)
    if norm == "snv":
        mu, sigma = s.mean(), s.std()
        if sigma > 1e-10:
            s = (s - mu) / sigma
    elif norm == "peak_phe":
        mask = (grid >= 988) & (grid <= 1018)
        ref  = s[mask].max() if mask.sum() > 0 else 0
        if ref > 1e-3:
            s = s / ref
        else:
            mu, sigma = s.mean(), s.std()
            if sigma > 1e-10:
                s = (s - mu) / sigma
    elif norm == "area":
        a = np.trapz(np.abs(s))
        if a > 1e-10:
            s /= a
    mu2, sigma2 = d2.mean(), d2.std()
    if sigma2 > 1e-10:
        d2 = (d2 - mu2) / sigma2
    return s, d2


def get_raman_bands(
    grid: np.ndarray,
) -> List[Tuple[float, float, str]]:
    w_min, w_max = grid.min(), grid.max()
    center = (w_min + w_max) / 2
    if center < 2000:
        bands = [
            (900,  1000, "C-C_skel"),
            (1000, 1100, "phe"),
            (1100, 1200, "CN_str"),
            (1200, 1350, "amide_III"),
            (1350, 1500, "CH2_def"),
            (1580, 1640, "CC_lip"),
            (1640, 1700, "amide_I"),
        ]
    else:
        bands = [
            (2700, 2800, "CH_ov"),
            (2820, 2870, "CH2_sym"),
            (2870, 2920, "CH3_sym"),
            (2920, 2970, "CH2_asy"),
            (2970, 3020, "CH3_asy"),
            (3020, 3100, "CH_str"),
        ]
    return [(lo, hi, nm) for lo, hi, nm in bands
            if lo >= w_min - 5 and hi <= w_max + 5]


def extract_spectrum_features(
    spec: np.ndarray,
    d2: np.ndarray,
    grid: np.ndarray,
    bands: List[Tuple],
    tag: str = "",
) -> Tuple[np.ndarray, List[str]]:
    """Точная копия функции из raman_v10 — попиксельное извлечение признаков."""
    feats:      List[float] = []
    names:      List[str]   = []
    band_areas: List[float] = []

    for lo, hi, band_name in bands:
        mask  = (grid >= lo) & (grid <= hi)
        g_seg = grid[mask]
        pfx   = f"{tag}_{band_name}"

        if mask.sum() == 0:
            zero_names = [
                f"{pfx}_area", f"{pfx}_peak_int", f"{pfx}_peak_pos",
                f"{pfx}_mean", f"{pfx}_std",      f"{pfx}_skew",
                f"{pfx}_kurt",
                f"{pfx}_d2_area", f"{pfx}_d2_min", f"{pfx}_d2_min_pos",
            ]
            feats.extend([0.0] * len(zero_names))
            names.extend(zero_names)
            band_areas.append(0.0)
            continue

        seg    = spec[mask]
        seg_d2 = d2[mask]

        area     = float(np.trapz(seg, g_seg))
        band_areas.append(area)
        pk_idx   = int(np.argmax(seg))
        peak_int = float(seg[pk_idx])
        peak_pos = float(g_seg[pk_idx])
        mean     = float(seg.mean())
        std      = float(seg.std()) + 1e-10
        centered = seg - mean
        skew     = float((centered ** 3).mean() / (std ** 3))
        kurt     = float((centered ** 4).mean() / (std ** 4)) - 3.0
        d2_area    = float(np.trapz(np.abs(seg_d2), g_seg))
        d2_min_idx = int(np.argmin(seg_d2))
        d2_min     = float(seg_d2[d2_min_idx])
        d2_min_pos = float(g_seg[d2_min_idx])

        feats.extend([area, peak_int, peak_pos, mean, std, skew, kurt,
                      d2_area, d2_min, d2_min_pos])
        names.extend([
            f"{pfx}_area",    f"{pfx}_peak_int", f"{pfx}_peak_pos",
            f"{pfx}_mean",    f"{pfx}_std",      f"{pfx}_skew",
            f"{pfx}_kurt",
            f"{pfx}_d2_area", f"{pfx}_d2_min",   f"{pfx}_d2_min_pos",
        ])

    n = len(bands)
    for i in range(n):
        for j in range(i + 1, n):
            ratio = band_areas[i] / (band_areas[j] + 1e-10)
            feats.append(ratio)
            names.append(f"{tag}_{bands[i][2]}_over_{bands[j][2]}")

    return np.array(feats, dtype=np.float32), names


def featurize_single_raw_spectrum(
    wave: np.ndarray,
    intensity: np.ndarray,
    grid: np.ndarray,
    bands: List[Tuple],
    center_tag: str,
    use_als: bool = False,
    norm: str = "snv",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Полный пайплайн: raw (wave, intensity) → вектор признаков.

    Returns
    -------
    feats      : (n_features,)
    spec_proc  : (n_grid,)  — предобработанный спектр (для отрисовки)
    d2         : (n_grid,)  — вторая производная (для отрисовки)
    """
    spec_interp      = np.interp(grid, wave, intensity)
    spec_proc, d2    = preprocess_spectrum(spec_interp, grid,
                                           use_als=use_als, norm=norm)
    feats, _         = extract_spectrum_features(spec_proc, d2, grid,
                                                  bands, tag=center_tag)
    return feats, spec_proc, d2


# =============================================================================
# ЗАГРУЗКА ФАЙЛА
# =============================================================================

def load_spectrum_file(
    filepath: Union[str, Path],
    wave_min: Optional[float] = None,
    wave_max: Optional[float] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Читает .txt файл со спектром.

    Поддерживает форматы:
      • «Wave Intensity»           — 2 столбца
      • «X Y Wave Intensity»       — 4 столбца (гиперспектральная карта)

    Returns
    -------
    list of (wave, intensity) — один элемент на пиксель.
    """
    fpath = Path(filepath)
    try:
        df = pd.read_csv(
            fpath, sep=r"\s+", comment="#", header=None, dtype=np.float64
        ).dropna()
    except Exception as e:
        raise IOError(f"Cannot read {fpath.name}: {e}") from e

    if df.shape[1] == 2:
        w, i = df.iloc[:, 0].values, df.iloc[:, 1].values
        if wave_min is not None: mask = w >= wave_min; w, i = w[mask], i[mask]
        if wave_max is not None: mask = w <= wave_max; w, i = w[mask], i[mask]
        order = np.argsort(w)
        return [(w[order], i[order])]

    elif df.shape[1] >= 4:
        df.columns = (["X", "Y", "Wave", "Intensity"]
                      + list(range(df.shape[1] - 4)))
        if wave_min is not None: df = df[df["Wave"] >= wave_min]
        if wave_max is not None: df = df[df["Wave"] <= wave_max]
        spectra = []
        for _, pix in df.groupby(["X", "Y"], sort=False):
            pix = pix.sort_values("Wave")
            if len(pix) >= 20:
                spectra.append((pix["Wave"].values, pix["Intensity"].values))
        return spectra
    else:
        raise ValueError(f"Unexpected number of columns: {df.shape[1]}")


# =============================================================================
# 1. ML PREDICTOR
# =============================================================================

class RamanMLPredictor:
    """
    Инференс ML-модели (best_model_center*.pkl / best_model_fused.pkl).

    Пайплайн соответствует raman_v10:
      raw file → interpolate → preprocess → extract_spectrum_features → predict

    Parameters
    ----------
    model_pkl_path : str | Path
        Путь к .pkl файлу, сохранённому raman_v10.py.
    """

    def __init__(self, model_pkl_path: Union[str, Path]) -> None:
        payload          = joblib.load(model_pkl_path)
        self.model       = payload["model"]
        self.le          = payload["label_encoder"]
        self.grid        = payload["grid"]
        self.bands       = payload["bands"]
        self.center_tag  = payload["center_tag"]
        self.norm        = payload.get("norm", "snv")
        self.use_als     = payload.get("use_als", False)
        wave_range       = payload.get("wave_range", (None, None))
        self.wave_min, self.wave_max = wave_range
        self.feat_names  = payload.get("feat_names", [])
        self.logo_acc    = payload.get("logo_acc", None)
        self.classes_    = list(self.le.classes_)

    # ------------------------------------------------------------------

    def predict_from_file(
        self,
        filepath: Union[str, Path],
        return_spectrum: bool = True,
    ) -> Dict[str, Any]:
        """
        Предсказание из .txt файла.

        Parameters
        ----------
        filepath       : путь к файлу спектра
        return_spectrum: если True, добавляет в результат предобработанный
                         спектр и d2 (нужны для графиков)

        Returns
        -------
        dict с ключами:
          prediction    : str  — predicted class label
          probabilities : dict — {class: float}
          confidence    : float
          spectrum_proc : np.ndarray  (если return_spectrum=True)
          d2            : np.ndarray  (если return_spectrum=True)
          grid          : np.ndarray
          band_stats    : dict  — статистики по полосам (для plot_peak_analysis)
          feat_names    : list[str]
          n_pixels      : int
        """
        raw_spectra = load_spectrum_file(
            filepath, wave_min=self.wave_min, wave_max=self.wave_max
        )
        if not raw_spectra:
            raise ValueError(f"No valid spectra in {filepath}")

        # Используем первый пиксель (или среднее если их несколько)
        if len(raw_spectra) == 1:
            wave, intensity = raw_spectra[0]
        else:
            # Интерполируем все пиксели и усредняем для итогового спектра
            all_interp = np.array([
                np.interp(self.grid, w, i) for w, i in raw_spectra
            ])
            wave      = self.grid
            intensity = all_interp.mean(axis=0)

        return self.predict_from_array(wave, intensity,
                                       return_spectrum=return_spectrum,
                                       n_pixels=len(raw_spectra))

    def predict_from_array(
        self,
        wave: np.ndarray,
        intensity: np.ndarray,
        return_spectrum: bool = True,
        n_pixels: int = 1,
    ) -> Dict[str, Any]:
        """Предсказание из массивов wave / intensity."""
        feats, spec_proc, d2 = featurize_single_raw_spectrum(
            wave, intensity,
            self.grid, self.bands, self.center_tag,
            use_als=self.use_als, norm=self.norm,
        )

        X = np.nan_to_num(feats.reshape(1, -1).astype(np.float32))
        pred_idx  = self.model.predict(X)[0]
        pred_label = self.le.inverse_transform([pred_idx])[0]

        try:
            proba_arr = self.model.predict_proba(X)[0]
        except AttributeError:
            # RidgeClassifier: softmax через decision_function
            scores = self.model.decision_function(X)[0]
            e = np.exp(scores - scores.max())
            proba_arr = e / e.sum()

        proba_dict = {
            self.le.inverse_transform([i])[0]: float(proba_arr[i])
            for i in range(len(proba_arr))
        }
        confidence = float(proba_arr.max())

        result: Dict[str, Any] = {
            "prediction":    pred_label,
            "probabilities": proba_dict,
            "confidence":    confidence,
            "grid":          self.grid,
            "band_stats":    self._compute_band_stats(spec_proc, d2),
            "feat_names":    self.feat_names,
            "n_pixels":      n_pixels,
            "center_tag":    self.center_tag,
        }
        if return_spectrum:
            result["spectrum_proc"] = spec_proc
            result["d2"]            = d2
        return result

    def _compute_band_stats(
        self, spec: np.ndarray, d2: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Возвращает статистики по каждой полосе для отрисовки peak_analysis.
        """
        stats: Dict[str, Dict] = {}
        for lo, hi, band_name in self.bands:
            mask  = (self.grid >= lo) & (self.grid <= hi)
            g_seg = self.grid[mask]
            if mask.sum() == 0:
                stats[band_name] = {}
                continue
            seg    = spec[mask]
            seg_d2 = d2[mask]
            pk_idx = int(np.argmax(seg))
            stats[band_name] = {
                "lo":        lo,
                "hi":        hi,
                "area":      float(np.trapz(seg, g_seg)),
                "peak_int":  float(seg[pk_idx]),
                "peak_pos":  float(g_seg[pk_idx]),
                "mean":      float(seg.mean()),
                "std":       float(seg.std()),
                "d2_min_pos": float(g_seg[int(np.argmin(seg_d2))]),
                "seg":        seg,
                "grid_seg":   g_seg,
            }
        return stats


# =============================================================================
# 2. CNN PREDICTOR
# =============================================================================

class RamanCNNPredictor:
    """
    Инференс CNN-модели (cnn_weights_center*.pt + cnn_meta_center*.pkl).

    Parameters
    ----------
    weights_path : str | Path   — путь к .pt файлу весов
    meta_path    : str | Path   — путь к .pkl метаданным
    device       : str          — 'cpu' | 'cuda' | 'auto'
    """

    def __init__(
        self,
        weights_path: Union[str, Path],
        meta_path: Union[str, Path],
        device: str = "auto",
    ) -> None:
        meta             = joblib.load(meta_path)
        self.le          = meta["label_encoder"]
        self.grid        = meta["grid"]
        self.bands       = meta["bands"]
        self.center_tag  = meta["center_tag"]
        self.norm        = meta.get("norm", "snv")
        self.use_als     = meta.get("use_als", False)
        self.n_grid      = meta["n_grid"]
        self.n_classes   = meta["n_classes"]
        self.dropout     = meta.get("dropout", 0.4)
        wave_range       = meta.get("wave_range", (None, None))
        self.wave_min, self.wave_max = wave_range
        self.classes_    = list(self.le.classes_)
        self.best_optuna = meta.get("best_optuna_params", {})

        self._torch, self._nn = self._load_torch_model(
            weights_path, device
        )

    def _load_torch_model(self, weights_path, device_str):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required for CNN predictor. pip install torch")

        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        # Строим архитектуру (копия из raman_v10.build_cnn_model)
        model = _build_cnn_architecture(
            self.n_grid, self.n_classes, nn, self.dropout
        ).to(device)
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        model.eval()
        return torch, model

    def predict_from_file(
        self,
        filepath: Union[str, Path],
        return_spectrum: bool = True,
    ) -> Dict[str, Any]:
        raw_spectra = load_spectrum_file(
            filepath, wave_min=self.wave_min, wave_max=self.wave_max
        )
        if not raw_spectra:
            raise ValueError(f"No valid spectra in {filepath}")

        if len(raw_spectra) == 1:
            wave, intensity = raw_spectra[0]
        else:
            all_interp = np.array([
                np.interp(self.grid, w, i) for w, i in raw_spectra
            ])
            wave      = self.grid
            intensity = all_interp.mean(axis=0)

        return self.predict_from_array(wave, intensity,
                                       return_spectrum=return_spectrum,
                                       n_pixels=len(raw_spectra))

    def predict_from_array(
        self,
        wave: np.ndarray,
        intensity: np.ndarray,
        return_spectrum: bool = True,
        n_pixels: int = 1,
    ) -> Dict[str, Any]:
        spec_interp      = np.interp(self.grid, wave, intensity)
        spec_proc, d2    = preprocess_spectrum(spec_interp, self.grid,
                                               use_als=self.use_als,
                                               norm=self.norm)
        spec_2ch = np.stack([spec_proc, d2], axis=0)[np.newaxis, ...]  # (1,2,L)

        with self._torch.no_grad():
            # 1. Получаем устройство, на котором сейчас находится модель (cpu или cuda)
            device = next(self._nn.parameters()).device 
            
            # 2. Создаем тензор и сразу переносим его на нужное устройство
            X_t    = self._torch.FloatTensor(spec_2ch.astype(np.float32)).to(device)
            
            # 3. Делаем предсказание
            logits = self._nn(X_t)
            
            # 4. ОБЯЗАТЕЛЬНО возвращаем результат на cpu перед вызовом .numpy()
            proba_arr = self._torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx   = int(np.argmax(proba_arr))
        pred_label = self.le.inverse_transform([pred_idx])[0]
        proba_dict = {
            self.le.inverse_transform([i])[0]: float(proba_arr[i])
            for i in range(len(proba_arr))
        }

        # Band stats (аналогично ML)
        ml_helper = RamanMLPredictor.__new__(RamanMLPredictor)
        ml_helper.grid  = self.grid
        ml_helper.bands = self.bands
        band_stats = ml_helper._compute_band_stats(spec_proc, d2)

        result: Dict[str, Any] = {
            "prediction":    pred_label,
            "probabilities": proba_dict,
            "confidence":    float(proba_arr.max()),
            "grid":          self.grid,
            "band_stats":    band_stats,
            "feat_names":    [],
            "n_pixels":      n_pixels,
            "center_tag":    self.center_tag,
        }
        if return_spectrum:
            result["spectrum_proc"] = spec_proc
            result["d2"]            = d2
        return result


def _build_cnn_architecture(n_grid, n_classes, nn_module, dropout=0.4):
    """Архитектура CNN идентична raman_v10.build_cnn_model."""
    nn = nn_module

    class SEBlock1d(nn.Module):
        def __init__(self, channels, reduction=8):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc   = nn.Sequential(
                nn.Linear(channels, max(channels // reduction, 1)),
                nn.ReLU(),
                nn.Linear(max(channels // reduction, 1), channels),
                nn.Sigmoid(),
            )
        def forward(self, x):
            s = self.pool(x).squeeze(-1)
            return x * self.fc(s).unsqueeze(-1)

    class ResBlock1d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel=7, stride=1):
            super().__init__()
            pad = kernel // 2
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                          padding=pad, bias=False),
                nn.BatchNorm1d(out_ch), nn.ReLU(),
                nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            self.se      = SEBlock1d(out_ch)
            self.relu    = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.shortcut = (
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_ch),
                )
                if (in_ch != out_ch or stride != 1) else nn.Identity()
            )
        def forward(self, x):
            out = self.se(self.conv(x))
            return self.dropout(self.relu(out + self.shortcut(x)))

    class RamanResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(2, 32, 15, padding=7, bias=False),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            )
            self.layer1 = ResBlock1d(32,  64,  7, stride=2)
            self.layer2 = ResBlock1d(64,  128, 5, stride=2)
            self.layer3 = ResBlock1d(128, 128, 3, stride=1)
            self.gap    = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )
        def forward(self, x):
            return self.classifier(self.gap(
                self.layer3(self.layer2(self.layer1(self.stem(x))))
            ))

    return RamanResNet()


# =============================================================================
# 3. ENSEMBLE
# =============================================================================

class RamanEnsemble:
    """
    Ансамбль нескольких предикторов (ML + CNN, разные центры).

    Итоговые вероятности — взвешенное среднее по всем предикторам.

    Parameters
    ----------
    predictors : list of (predictor, weight)
        predictor — RamanMLPredictor или RamanCNNPredictor
        weight    — float (например, LOGO accuracy модели)
    """

    def __init__(
        self,
        predictors: List[Tuple[Any, float]],
    ) -> None:
        self.predictors = predictors
        total = sum(w for _, w in predictors)
        self.weights = [w / total for _, w in predictors]

    def predict_from_file(
        self,
        filepath: Union[str, Path],
        return_spectrum: bool = True,
    ) -> Dict[str, Any]:
        results = []
        for predictor, _ in self.predictors:
            try:
                r = predictor.predict_from_file(
                    filepath, return_spectrum=return_spectrum
                )
                results.append(r)
            except Exception as e:
                print(f"  [WARN] Predictor {type(predictor).__name__} failed: {e}")

        if not results:
            raise RuntimeError("All predictors failed")

        # Усредняем вероятности
        classes = list(results[0]["probabilities"].keys())
        avg_proba = np.zeros(len(classes))
        for res, w in zip(results, self.weights):
            p = np.array([res["probabilities"][c] for c in classes])
            avg_proba += w * p

        pred_label = classes[int(np.argmax(avg_proba))]
        proba_dict = dict(zip(classes, avg_proba.tolist()))

        ensemble_result = results[0].copy()
        ensemble_result.update({
            "prediction":    pred_label,
            "probabilities": proba_dict,
            "confidence":    float(avg_proba.max()),
            "individual":    results,
        })
        return ensemble_result


# =============================================================================
# 4. PLOT: SPECTRUM + PREDICTION
# =============================================================================

def plot_spectrum_prediction(
    result: Dict[str, Any],
    title: str = "",
    figsize: Tuple[int, int] = (14, 7),
    show_d2: bool = True,
    show_bands: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Основной график для бэкенда.

    Компоновка:
      ┌─────────────────────────────────────┬──────────┐
      │  Спектр + подсветка полос + пики    │  Bar     │
      │                                     │  вероят- │
      ├─────────────────────────────────────┤  ностей  │
      │  Вторая производная (опц.)          │          │
      └─────────────────────────────────────┴──────────┘

    Parameters
    ----------
    result      : dict, возвращённый RamanMLPredictor.predict_from_*
    title       : заголовок графика
    show_d2     : показывать вторую производную
    show_bands  : подсвечивать рамановские полосы
    save_path   : если указан — сохранить файл
    """
    grid       = result["grid"]
    spec       = result.get("spectrum_proc")
    d2         = result.get("d2")
    proba      = result["probabilities"]
    pred_label = result["prediction"]
    band_stats = result.get("band_stats", {})
    annotations = _band_annotations_for_grid(grid)

    n_rows = 2 if (show_d2 and d2 is not None) else 1
    fig = plt.figure(figsize=figsize, facecolor="#FAFAFA")
    gs  = gridspec.GridSpec(
        n_rows, 2,
        width_ratios=[3, 1],
        hspace=0.35, wspace=0.25,
        left=0.07, right=0.97, top=0.92, bottom=0.10,
    )
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_bar  = fig.add_subplot(gs[:, 1])
    if show_d2 and d2 is not None:
        ax_d2 = fig.add_subplot(gs[1, 0], sharex=ax_spec)

    pred_color = CLASS_COLORS.get(pred_label, "#607D8B")

    # ── Spectrum ─────────────────────────────────────────────────────────────
    if spec is not None:
        ax_spec.plot(grid, spec, color="#212121", lw=1.5, zorder=5,
                     label="Preprocessed spectrum")

        # Подсветка полос
        if show_bands:
            band_colors = [
                "#E3F2FD", "#E8F5E9", "#FFF3E0", "#F3E5F5",
                "#E0F7FA", "#FCE4EC", "#F9FBE7",
            ]
            for idx, (band_name, bst) in enumerate(band_stats.items()):
                if not bst:
                    continue
                lo, hi = bst["lo"], bst["hi"]
                color  = band_colors[idx % len(band_colors)]
                ax_spec.axvspan(lo, hi, alpha=0.35, color=color, zorder=1)

                # Подпись пика
                ann = annotations.get(band_name, {})
                lbl = ann.get("label", band_name)
                pk  = bst.get("peak_pos", (lo + hi) / 2)
                pi  = bst.get("peak_int", 0)
                ax_spec.annotate(
                    lbl,
                    xy=(pk, pi),
                    xytext=(pk, pi + 0.12 * (spec.max() - spec.min())),
                    fontsize=7, ha="center", color="#424242",
                    arrowprops=dict(arrowstyle="-", color="#9E9E9E",
                                    lw=0.8, alpha=0.7),
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="#BDBDBD", alpha=0.85),
                )

        ax_spec.set_ylabel("Intensity (norm.)", fontsize=9)
        ax_spec.set_title(
            f"Raman spectrum  ·  {result.get('center_tag', '')}",
            fontsize=10, pad=6,
        )
        ax_spec.tick_params(labelsize=8)
        ax_spec.spines[["top", "right"]].set_visible(False)

        # Тонкая цветная рамка по цвету предсказанного класса
        for spine in ["left", "bottom"]:
            ax_spec.spines[spine].set_edgecolor(pred_color)
            ax_spec.spines[spine].set_linewidth(2)

    # ── Second derivative ────────────────────────────────────────────────────
    if show_d2 and d2 is not None:
        ax_d2.plot(grid, d2, color="#78909C", lw=1.0, alpha=0.8)
        ax_d2.axhline(0, color="#BDBDBD", lw=0.7, ls="--")
        ax_d2.set_xlabel("Raman shift (cm⁻¹)", fontsize=9)
        ax_d2.set_ylabel("2nd deriv.", fontsize=9)
        ax_d2.tick_params(labelsize=8)
        ax_d2.spines[["top", "right"]].set_visible(False)
        if show_bands:
            for band_name, bst in band_stats.items():
                if not bst:
                    continue
                dp = bst.get("d2_min_pos")
                if dp is not None:
                    ax_d2.axvline(dp, color="#EF5350", lw=0.8,
                                  ls=":", alpha=0.7)
    else:
        if spec is not None:
            ax_spec.set_xlabel("Raman shift (cm⁻¹)", fontsize=9)

    # ── Probability bar ──────────────────────────────────────────────────────
    classes = list(proba.keys())
    values  = [proba[c] for c in classes]
    colors  = [CLASS_COLORS.get(c, "#90A4AE") for c in classes]
    bars    = ax_bar.barh(classes, values, color=colors,
                          alpha=0.85, height=0.55, edgecolor="white", lw=1.5)

    for bar, val, cls in zip(bars, values, classes):
        ax_bar.text(
            min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left",
            fontsize=10, fontweight="bold",
            color=CLASS_COLORS.get(cls, "#424242"),
        )
        if cls == pred_label:
            bar.set_edgecolor(pred_color)
            bar.set_linewidth(2.5)

    ax_bar.set_xlim(0, 1.15)
    ax_bar.axvline(1/3, color="#BDBDBD", lw=1, ls="--", alpha=0.7)
    ax_bar.set_xlabel("Probability", fontsize=9)
    ax_bar.set_title("Prediction", fontsize=10)
    ax_bar.tick_params(labelsize=9)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # Вердикт
    conf = result.get("confidence", max(values))
    conf_str = f"Confidence: {conf:.1%}"
    ax_bar.text(
        0.5, -0.18, conf_str,
        transform=ax_bar.transAxes,
        ha="center", fontsize=9, color="#616161",
    )

    # Большой заголовок
    suptitle = title or f"Prediction: {pred_label.upper()}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=pred_color, y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# =============================================================================
# 5. PLOT: PEAK ANALYSIS  (обратная задача)
# =============================================================================

def plot_peak_analysis(
    result: Dict[str, Any],
    title: str = "",
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Детальный разбор полос — «обратная задача»:
    показывает, какие спектральные полосы наиболее информативны.

    Компоновка:
      ┌──────────────────────────┬──────────────────────────┐
      │  Спектр + выделенные     │  Bar: площади полос      │
      │  сегменты по полосам     │  (area под кривой)       │
      ├──────────────────────────┴──────────────────────────┤
      │  Детали каждой полосы: мини-спектр + d2             │
      └─────────────────────────────────────────────────────┘

    Parameters
    ----------
    result : dict from RamanMLPredictor.predict_from_*
    """
    grid       = result["grid"]
    spec       = result.get("spectrum_proc")
    d2         = result.get("d2")
    band_stats = result.get("band_stats", {})
    pred_label = result["prediction"]
    annotations = _band_annotations_for_grid(grid)

    active_bands = {
        k: v for k, v in band_stats.items() if v and v.get("area", 0) != 0
    }
    n_bands = len(active_bands)
    if n_bands == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No band data", ha="center", va="center")
        return fig

    pred_color = CLASS_COLORS.get(pred_label, "#607D8B")
    band_palette = plt.cm.get_cmap("tab10", n_bands)

    # Сетка: 2 строки верху + n_bands мини-графиков внизу
    n_cols_mini = min(n_bands, 4)
    n_rows_mini = (n_bands + n_cols_mini - 1) // n_cols_mini

    fig = plt.figure(
        figsize=(figsize[0], figsize[1] + 2.5 * n_rows_mini),
        facecolor="#FAFAFA",
    )
    outer_gs = gridspec.GridSpec(
        2, 1, hspace=0.4,
        height_ratios=[3, 1.8 * n_rows_mini],
        left=0.06, right=0.97, top=0.93, bottom=0.04,
    )
    top_gs   = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[0], wspace=0.3
    )
    mini_gs  = gridspec.GridSpecFromSubplotSpec(
        n_rows_mini, n_cols_mini,
        subplot_spec=outer_gs[1],
        hspace=0.55, wspace=0.35,
    )

    ax_full = fig.add_subplot(top_gs[0])
    ax_area = fig.add_subplot(top_gs[1])

    # ── Полный спектр с подсветкой полос ─────────────────────────────────────
    if spec is not None:
        ax_full.plot(grid, spec, color="#212121", lw=1.4, zorder=5)
        for idx, (band_name, bst) in enumerate(active_bands.items()):
            color = band_palette(idx)
            lo, hi = bst["lo"], bst["hi"]
            mask   = (grid >= lo) & (grid <= hi)
            ax_full.fill_between(
                grid[mask], spec[mask], alpha=0.55,
                color=color, zorder=3,
            )
            ax_full.axvline(bst["peak_pos"], color=color, lw=0.8,
                            ls="--", alpha=0.7, zorder=4)
            ann = annotations.get(band_name, {})
            ax_full.text(
                bst["peak_pos"], spec[mask].max() * 1.02,
                ann.get("label", band_name).split("\n")[0],
                fontsize=7, ha="center", color=color, fontweight="bold",
            )
        ax_full.set_xlabel("Raman shift (cm⁻¹)", fontsize=9)
        ax_full.set_ylabel("Intensity (norm.)", fontsize=9)
        ax_full.set_title("Spectrum with band segmentation", fontsize=10)
        ax_full.tick_params(labelsize=8)
        ax_full.spines[["top", "right"]].set_visible(False)

    # ── Горизонтальный бар площадей ──────────────────────────────────────────
    band_names = list(active_bands.keys())
    areas      = [active_bands[b]["area"] for b in band_names]
    bio_labels = [
        annotations.get(b, {}).get("bio", b)[:18] for b in band_names
    ]
    colors_bar = [band_palette(i) for i in range(n_bands)]

    bars = ax_area.barh(
        bio_labels, areas, color=colors_bar,
        alpha=0.8, height=0.6, edgecolor="white", lw=1
    )
    ax_area.set_xlabel("Band area (a.u.)", fontsize=9)
    ax_area.set_title("Band areas (informativeness)", fontsize=10)
    ax_area.tick_params(labelsize=8)
    ax_area.spines[["top", "right"]].set_visible(False)

    # Нормированная важность поверх бара
    max_area = max(areas) if areas else 1
    for bar, area in zip(bars, areas):
        pct = area / (max_area + 1e-10)
        ax_area.text(
            area + 0.01 * max_area,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.0%}", va="center", fontsize=8, color="#424242",
        )

    # ── Мини-графики по каждой полосе ────────────────────────────────────────
    for idx, (band_name, bst) in enumerate(active_bands.items()):
        row = idx // n_cols_mini
        col = idx % n_cols_mini
        ax_m = fig.add_subplot(mini_gs[row, col])
        color = band_palette(idx)
        ann   = annotations.get(band_name, {})

        g_seg, seg = bst["grid_seg"], bst["seg"]
        ax_m.plot(g_seg, seg, color=color, lw=1.5)
        ax_m.fill_between(g_seg, seg, alpha=0.25, color=color)

        # Маркер пика
        pk_pos, pk_int = bst["peak_pos"], bst["peak_int"]
        ax_m.axvline(pk_pos, color=color, lw=1, ls="--", alpha=0.8)
        ax_m.scatter([pk_pos], [pk_int], color=color, s=30, zorder=5)
        ax_m.text(pk_pos, pk_int * 1.05, f"{pk_pos:.0f}",
                  fontsize=7, ha="center", color=color)

        # d2 на вторичной оси
        if d2 is not None:
            mask   = (grid >= bst["lo"]) & (grid <= bst["hi"])
            d2_seg = d2[mask]
            ax_d2m = ax_m.twinx()
            ax_d2m.plot(g_seg, d2_seg, color="#90A4AE", lw=0.8,
                        alpha=0.6, ls=":")
            ax_d2m.axhline(0, color="#CFD8DC", lw=0.5)
            ax_d2m.set_yticks([])

        ax_m.set_title(
            f"{ann.get('label', band_name).replace(chr(10), ' ')}\n"
            f"{ann.get('bio', '')}",
            fontsize=7, pad=3,
        )
        ax_m.tick_params(labelsize=7)
        ax_m.spines[["top", "right"]].set_visible(False)
        for spine in ["left", "bottom"]:
            ax_m.spines[spine].set_edgecolor(color)

    suptitle = title or f"Peak Analysis · {pred_label.upper()} · {result.get('center_tag', '')}"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold",
                 color=pred_color, y=0.97)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# =============================================================================
# 6. PLOT: SPATIAL MAP
# =============================================================================

def plot_spatial_map(
    predictions: List[Dict[str, Any]],
    xy_coords: List[Tuple[float, float]],
    title: str = "",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Карта предсказаний по пространственным координатам пикселей.

    Parameters
    ----------
    predictions : list of result dicts (по одному на пиксель)
    xy_coords   : list of (x, y) координат пикселей
    """
    if not predictions:
        raise ValueError("Empty predictions list")

    classes   = sorted(set(r["prediction"] for r in predictions))
    class_idx = {c: i for i, c in enumerate(classes)}
    colors    = [CLASS_COLORS.get(c, "#90A4AE") for c in classes]
    cmap      = LinearSegmentedColormap.from_list("raman", colors, N=len(classes))

    xs = np.array([p[0] for p in xy_coords])
    ys = np.array([p[1] for p in xy_coords])
    zs = np.array([class_idx[r["prediction"]] for r in predictions], dtype=float)
    confs = np.array([r.get("confidence", 1.0) for r in predictions])

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="#FAFAFA")
    fig.suptitle(title or "Spatial Prediction Map", fontsize=12,
                 fontweight="bold", y=0.99)

    # Class map
    sc1 = axes[0].scatter(xs, ys, c=zs, cmap=cmap,
                          vmin=-0.5, vmax=len(classes) - 0.5,
                          s=60, alpha=0.9, edgecolors="none")
    axes[0].set_title("Class map", fontsize=10)
    axes[0].set_xlabel("X (px)"); axes[0].set_ylabel("Y (px)")
    axes[0].set_aspect("equal")
    patches = [
        mpatches.Patch(color=CLASS_COLORS.get(c, "#90A4AE"), label=c)
        for c in classes
    ]
    axes[0].legend(handles=patches, fontsize=8, loc="best")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Confidence map
    sc2 = axes[1].scatter(xs, ys, c=confs, cmap="RdYlGn",
                          vmin=0, vmax=1, s=60, alpha=0.9,
                          edgecolors="none")
    axes[1].set_title("Confidence map", fontsize=10)
    axes[1].set_xlabel("X (px)")
    axes[1].set_aspect("equal")
    plt.colorbar(sc2, ax=axes[1], label="Confidence", shrink=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# =============================================================================
# 7. PLOT: COMPARISON SPECTRA (средние спектры по классам + маркеры пиков)
# =============================================================================

def plot_comparison_spectra(
    spectra_by_class: Dict[str, np.ndarray],
    grid: np.ndarray,
    title: str = "",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    highlight_peaks: bool = True,
) -> plt.Figure:
    """
    Средние спектры по классам в стиле Рис.2 из задания.

    Parameters
    ----------
    spectra_by_class : dict {class_name: np.ndarray (n_pixels, n_grid)}
    grid             : общая ось волновых чисел
    highlight_peaks  : отмечать известные рамановские пики вертикальными линиями

    Пики аннотируются так же как на Рис.2: 1062, 1128, 1294, 1420, 1440
    и биохимические метки.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#FAFAFA")
    bands   = get_raman_bands(grid)
    annotations = _band_annotations_for_grid(grid)

    # Offset для разделения спектров (как на Рис.2)
    offsets = {cls: i * 0.8 for i, cls in enumerate(sorted(spectra_by_class))}

    for cls, spectra in sorted(spectra_by_class.items()):
        arr    = np.asarray(spectra)
        mean_s = arr.mean(axis=0) if arr.ndim == 2 else arr
        std_s  = arr.std(axis=0) if arr.ndim == 2 else np.zeros_like(arr)
        offset = offsets[cls]
        color  = CLASS_COLORS.get(cls, "#607D8B")

        ax.plot(grid, mean_s + offset, color=color, lw=1.8,
                label=f"{cls} (n={len(arr) if arr.ndim == 2 else 1})",
                zorder=5)
        if arr.ndim == 2 and len(arr) > 1:
            ax.fill_between(
                grid,
                mean_s + offset - std_s,
                mean_s + offset + std_s,
                alpha=0.18, color=color, zorder=3,
            )

    # Вертикальные маркеры известных пиков
    if highlight_peaks:
        center_grid = (grid.min() + grid.max()) / 2
        if center_grid < 2000:
            key_peaks = [
                (1062, "1062\nphospholipids"),
                (1128, "1128\nvalence C-C"),
                (1294, "1294\nprotein groups"),
                (1420, "1420\nnucleic acids"),
                (1440, "1440\nlipids"),
            ]
        else:
            key_peaks = [
                (2850, "2850\nCH₂ sym"),
                (2895, "2895\nCH₃ sym"),
                (2940, "2940\nCH₂ asym"),
                (2990, "2990\nCH₃ asym"),
            ]

        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1
        for wave_pos, peak_label in key_peaks:
            if grid.min() <= wave_pos <= grid.max():
                ax.axvline(wave_pos, color="#424242", lw=0.9,
                           ls="-", alpha=0.5, zorder=2)
                ax.text(
                    wave_pos, ax.get_ylim()[1] * 0.97,
                    peak_label,
                    fontsize=7, ha="center", va="top",
                    color="#424242", rotation=90,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="none", alpha=0.7),
                )

    ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=11)
    ax.set_ylabel("Intensity (offset, norm.)", fontsize=11)
    ax.set_title(title or "Mean Raman spectra by class", fontsize=12,
                 pad=8)
    ax.legend(fontsize=10, loc="upper right",
              framealpha=0.9, edgecolor="#BDBDBD")
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# =============================================================================
# CONVENIENCE: загрузка предикторов по папке outputs/
# =============================================================================

def load_predictors_from_dir(
    outputs_dir: Union[str, Path] = "outputs",
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Автоматически загружает все модели из папки outputs/.

    Returns
    -------
    dict с ключами:
      'ml_center1500', 'ml_center2900', 'ml_fused',
      'cnn_center1500', 'cnn_center2900',
      'ensemble'  (если есть хоть одна модель)
    """
    d    = Path(outputs_dir)
    pred: Dict[str, Any] = {}
    ens_list: List[Tuple] = []

    for center in ["center1500", "center2900"]:
        ml_path = d / f"best_model_{center}.pkl"
        if ml_path.exists():
            try:
                p = RamanMLPredictor(ml_path)
                pred[f"ml_{center}"] = p
                w = p.logo_acc if p.logo_acc else 0.5
                ens_list.append((p, w))
                print(f"  ✅ ML {center} loaded  (LOGO={w:.3f})")
            except Exception as e:
                print(f"  [WARN] ML {center}: {e}")

        wt_path   = d / f"cnn_weights_{center}.pt"
        meta_path = d / f"cnn_meta_{center}.pkl"
        if wt_path.exists() and meta_path.exists():
            try:
                p = RamanCNNPredictor(wt_path, meta_path, device=device)
                pred[f"cnn_{center}"] = p
                ens_list.append((p, 0.5))
                print(f"  ✅ CNN {center} loaded")
            except Exception as e:
                print(f"  [WARN] CNN {center}: {e}")

    fused_path = d / "best_model_fused.pkl"
    if fused_path.exists():
        try:
            p = RamanMLPredictor(fused_path)
            pred["ml_fused"] = p
            print(f"  ✅ ML fused loaded")
        except Exception as e:
            print(f"  [WARN] ML fused: {e}")

    if ens_list:
        pred["ensemble"] = RamanEnsemble(ens_list)
        print(f"  ✅ Ensemble: {len(ens_list)} predictors")

    return pred
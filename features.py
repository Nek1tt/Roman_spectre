"""
features.py — Raman band definitions and per-spectrum feature extraction.

Public API
----------
get_raman_bands(grid)                          → List[(lo, hi, name)]
extract_spectrum_features(spec, d2, grid, ...) → (features, names)
build_pixel_feature_matrix(maps, ...)          → dict
featurize_single_raw_spectrum(wave, ...)       → np.ndarray
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from preprocessing import preprocess_spectrum


# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------

def get_raman_bands(grid: np.ndarray) -> List[Tuple[float, float, str]]:
    """Return the Raman band list appropriate for the given wavenumber grid."""
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


# ---------------------------------------------------------------------------
# Per-spectrum feature vector
# ---------------------------------------------------------------------------

def extract_spectrum_features(
    spec: np.ndarray,
    d2: np.ndarray,
    grid: np.ndarray,
    bands: List[Tuple],
    tag: str = "",
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract a fixed-length feature vector from one preprocessed spectrum.

    Features per band (×n_bands):
      area, peak_int, peak_pos, mean, std, skew, kurt,
      d2_area, d2_min, d2_min_pos
    Plus all pairwise band-area ratios.

    Returns
    -------
    feats : (n_features,)
    names : list of feature name strings
    """
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
                f"{pfx}_mean", f"{pfx}_std",       f"{pfx}_skew",
                f"{pfx}_kurt",
                f"{pfx}_d2_area", f"{pfx}_d2_min", f"{pfx}_d2_min_pos",
            ]
            feats.extend([0.0] * len(zero_names))
            names.extend(zero_names)
            band_areas.append(0.0)
            continue

        seg    = spec[mask]
        seg_d2 = d2[mask]

        area = float(np.trapz(seg, g_seg))
        band_areas.append(area)

        pk_idx   = int(np.argmax(seg))
        peak_int = float(seg[pk_idx])
        peak_pos = float(g_seg[pk_idx])

        mean = float(seg.mean())
        std  = float(seg.std()) + 1e-10
        centered = seg - mean
        skew = float((centered ** 3).mean() / (std ** 3))
        kurt = float((centered ** 4).mean() / (std ** 4)) - 3.0

        d2_area    = float(np.trapz(np.abs(seg_d2), g_seg))
        d2_min_idx = int(np.argmin(seg_d2))
        d2_min     = float(seg_d2[d2_min_idx])
        d2_min_pos = float(g_seg[d2_min_idx])

        feats.extend([area, peak_int, peak_pos, mean, std, skew, kurt,
                      d2_area, d2_min, d2_min_pos])
        names.extend([
            f"{pfx}_area",    f"{pfx}_peak_int", f"{pfx}_peak_pos",
            f"{pfx}_mean",    f"{pfx}_std",       f"{pfx}_skew",
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


# ---------------------------------------------------------------------------
# Batch feature matrix from map records
# ---------------------------------------------------------------------------

def build_pixel_feature_matrix(
    maps: List[Dict],
    center_tag: str,
    use_als: bool = False,
    norm: str = "snv",
    n_jobs: int = -1,
) -> Optional[Dict]:
    """
    Build a pixel-level feature matrix from a list of map records.

    Each record is expected to have keys: label, animal_id, pixels, grid.

    Returns
    -------
    dict with keys: X, y, aids, feat_names, grid, bands
    """
    if not maps:
        return None

    w_min_c = max(m["grid"].min() for m in maps)
    w_max_c = min(m["grid"].max() for m in maps)
    grid    = np.linspace(w_min_c, w_max_c, len(maps[0]["grid"]))
    bands   = get_raman_bands(grid)
    print(f"  [{center_tag}] {len(maps)} maps, {len(bands)} bands, "
          f"grid=[{grid.min():.0f}, {grid.max():.0f}]")

    X:          List[np.ndarray]    = []
    y:          List[str]           = []
    aids:       List[str]           = []
    feat_names: Optional[List[str]] = None

    def _process_pixel(px_raw: np.ndarray):
        return preprocess_spectrum(px_raw, grid, use_als=use_als, norm=norm)

    for rec in tqdm(maps, desc=f"  Pixel features {center_tag}", ncols=80):
        pix_raw  = np.array([np.interp(grid, rec["grid"], px)
                             for px in rec["pixels"]])
        processed = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_process_pixel)(px) for px in pix_raw
        )
        for spec, d2 in processed:
            feats, fnames = extract_spectrum_features(
                spec, d2, grid, bands, tag=center_tag
            )
            X.append(feats)
            if feat_names is None:
                feat_names = fnames
            y.append(rec["label"])
            aids.append(rec["animal_id"])

    X_arr    = np.nan_to_num(np.array(X, dtype=np.float32))
    y_arr    = np.array(y)
    aids_arr = np.array(aids)

    n_px_total = X_arr.shape[0]
    n_maps     = len(maps)
    print(f"  Pixel-level matrix: {n_px_total} px × {X_arr.shape[1]} feat "
          f"({n_px_total // max(n_maps, 1)} px/map avg)")

    return {
        "X": X_arr, "y": y_arr, "aids": aids_arr,
        "feat_names": feat_names or [], "grid": grid, "bands": bands,
    }


# ---------------------------------------------------------------------------
# Single-spectrum featurisation (used during inference)
# ---------------------------------------------------------------------------

def featurize_single_raw_spectrum(
    wave: np.ndarray,
    intensity: np.ndarray,
    grid: np.ndarray,
    bands: List[Tuple],
    center_tag: str,
    use_als: bool = False,
    norm: str = "snv",
) -> np.ndarray:
    """Raw (wave, intensity) arrays → feature vector."""
    spec_interp   = np.interp(grid, wave, intensity)
    spec_proc, d2 = preprocess_spectrum(spec_interp, grid,
                                        use_als=use_als, norm=norm)
    feats, _      = extract_spectrum_features(spec_proc, d2, grid, bands,
                                              tag=center_tag)
    return feats

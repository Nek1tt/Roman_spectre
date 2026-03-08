"""
data_loading.py — Dataset scanning and raw spectrum file loading.

Public API
----------
load_hyperspectral_file(filepath, wave_min, wave_max) → List[(wave, intensity)]
load_dataset_maps(data_root, n_grid)                  → Dict[int, List[Dict]]
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Roman_spectre.constants import (
    CLASS_DIRS, ANIMAL_RE, BRAIN_REGIONS,
    CENTER_RE, PLACE_RE, BAND_RANGES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def folder_to_animal_id(folder_name: str, label: str) -> str:
    m = ANIMAL_RE.search(folder_name)
    return f"{label}_{m.group(1) if m else folder_name}"


def parse_filename(fname: str) -> Dict:
    fname  = fname.lower()
    region = next((r for r in BRAIN_REGIONS if r in fname), "unknown")
    cm     = CENTER_RE.search(fname)
    pm     = PLACE_RE.search(fname)
    return {
        "region": region,
        "center": int(cm.group(1)) if cm else -1,
        "place":  pm.group(1) if pm else "0",
    }


def find_subdir(data_root: Path, label: str, subdir: str) -> Optional[Path]:
    for candidate in [
        data_root / label / label / subdir,
        data_root / label / subdir,
        data_root / subdir,
    ]:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Single-file loaders
# ---------------------------------------------------------------------------

def load_hyperspectral_file(
    filepath: str,
    wave_min: Optional[float] = None,
    wave_max: Optional[float] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Read a 4-column (X, Y, Wave, Intensity) hyperspectral map file.

    Returns a list of (wave, intensity) arrays, one per pixel.
    """
    try:
        df = pd.read_csv(
            filepath, sep=r"\s+", comment="#",
            names=["X", "Y", "Wave", "Intensity"],
            dtype=np.float64,
        ).dropna()
    except Exception:
        return []

    if df.empty or len(df) < 10:
        return []

    if wave_min is not None:
        df = df[df["Wave"] >= wave_min]
    if wave_max is not None:
        df = df[df["Wave"] <= wave_max]
    if df.empty:
        return []

    spectra = []
    for (_, _), pix in df.groupby(["X", "Y"], sort=False):
        pix  = pix.sort_values("Wave")
        wave = pix["Wave"].values
        intn = pix["Intensity"].values
        if len(wave) >= 20:
            spectra.append((wave, intn))
    return spectra


# ---------------------------------------------------------------------------
# Full dataset scan
# ---------------------------------------------------------------------------

def load_dataset_maps(
    data_root: str,
    n_grid: int = 256,
) -> Dict[int, List[Dict]]:
    """
    Walk the dataset folder tree and load all hyperspectral maps.

    Returns
    -------
    dict mapping spectral centre (1500 / 2900) to list of map records.
    Each record has: label, animal_id, region, place_id, pixels, grid.
    """
    data_root = Path(data_root)
    maps: Dict[int, List[Dict]] = {1500: [], 2900: []}

    print("\n📂 Scanning dataset folders...")
    for label, subdirs in CLASS_DIRS.items():
        for subdir in subdirs:
            folder = find_subdir(data_root, label, subdir)
            if folder is None:
                print(f"  [WARN] not found: {label}/{subdir}")
                continue

            animal_id = folder_to_animal_id(subdir, label)
            txt_files = sorted(folder.glob("*.txt"))
            print(f"  {label}/{subdir}: {len(txt_files)} files  [{animal_id}]")

            for fpath in txt_files:
                fname = fpath.stem.lower()
                if "average" in fname:
                    continue

                meta   = parse_filename(fname)
                center = meta["center"]
                if center not in maps:
                    continue

                w_min, w_max  = BAND_RANGES[center]
                pixel_spectra = load_hyperspectral_file(
                    str(fpath), wave_min=w_min, wave_max=w_max
                )
                if not pixel_spectra:
                    continue

                all_w  = np.concatenate([s[0] for s in pixel_spectra])
                grid   = np.linspace(all_w.min(), all_w.max(), n_grid)
                pixels = np.array([np.interp(grid, s[0], s[1])
                                   for s in pixel_spectra])
                maps[center].append({
                    "label":     label,
                    "animal_id": animal_id,
                    "region":    meta["region"],
                    "place_id":  f"{animal_id}_p{meta['place']}",
                    "pixels":    pixels,
                    "grid":      grid,
                })

    for c, recs in maps.items():
        if recs:
            n_px = [len(r["pixels"]) for r in recs]
            print(f"\n  ✅ center{c}: {len(recs)} maps, "
                  f"~{np.mean(n_px):.0f} px/map (total {sum(n_px)} px)")
            labels = [r["label"] for r in recs]
            print(f"     Classes: { {cl: labels.count(cl) for cl in set(labels)} }")

    return maps

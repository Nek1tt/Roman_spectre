"""
app_gradio.py — Raman Spectrum Analysis Interface
Прямая задача: спектр → предсказание класса
Обратная задача: класс → характерные спектральные маркеры
"""

import os
import tempfile
import traceback
import warnings

warnings.filterwarnings("ignore")

from ml_models import OptunaRidgeClf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
import gradio as gr
import joblib

# Обновленные импорты из inference_utils.py
from inference_utils import (
    RamanMLPredictor,
    RamanCNNPredictor,
    preprocess_spectrum,
    get_raman_bands,
    fast_baseline,
    als_baseline,
)

# ─────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────
PRELOADED_MODELS = {
    "Центр 1500 см⁻¹ (амиды, белки)": "outputs/best_model_center1500.pkl",
    "Центр 2900 см⁻¹ (липиды, CH)": "outputs/best_model_center2900.pkl",
}

BRAIN_REGIONS = ["unknown", "cortex", "striatum", "cerebellum"]

CLASS_COLORS = {
    "control": "#4A9EFF",
    "endo":    "#FF7043",
    "exo":     "#66BB6A",
}
CLASS_LABELS = {
    "control": "Контроль",
    "endo":    "Эндогенная HSP70",
    "exo":     "Экзогенная HSP70",
}

# ─────────────────────────────────────────────────────────
# Тема matplotlib
# ─────────────────────────────────────────────────────────
def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  "#0F1117",
        "axes.facecolor":    "#161B22",
        "axes.edgecolor":    "#30363D",
        "axes.labelcolor":   "#C9D1D9",
        "axes.titlecolor":   "#E6EDF3",
        "xtick.color":       "#8B949E",
        "ytick.color":       "#8B949E",
        "grid.color":        "#21262D",
        "grid.linewidth":    0.8,
        "text.color":        "#C9D1D9",
        "legend.facecolor":  "#161B22",
        "legend.edgecolor":  "#30363D",
        "legend.labelcolor": "#C9D1D9",
        "font.family":       "monospace",
        "font.size":         10,
    })

# ─────────────────────────────────────────────────────────
# Загрузка файла
# ─────────────────────────────────────────────────────────
def _save_uploaded_file(uploaded) -> str:
    if uploaded is None:
        raise ValueError("Файл не загружен.")
    if isinstance(uploaded, str) and os.path.exists(uploaded):
        return uploaded
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    data = uploaded.read() if hasattr(uploaded, "read") else open(uploaded, "rb").read()
    if isinstance(data, str):
        data = data.encode()
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def _load_dataframe(filepath):
    """
    Автоматически определяет формат файла:
      • 2 колонки → одиночный спектр (Wave, Intensity) (без X, Y)
      • 4+ колонки → гиперспектральная карта (X, Y, Wave, Intensity)
    Всегда возвращает DataFrame с колонками X, Y, Wave, Intensity.
    """
    raw = pd.read_csv(
        filepath, sep=r"\s+", comment="#", header=None, dtype=np.float64,
    ).dropna()

    if raw.shape[1] == 2:
        # Одиночный спектр — добавляем фиктивные X=0, Y=0 для совместимости
        raw.columns =["Wave", "Intensity"]
        raw.insert(0, "X", 0.0)
        raw.insert(1, "Y", 0.0)
    elif raw.shape[1] >= 4:
        raw = raw.iloc[:, :4]
        raw.columns =["X", "Y", "Wave", "Intensity"]
    else:
        raise ValueError(f"Неожиданное число колонок: {raw.shape[1]}. "
                         "Ожидается 2 (Wave Intensity) или 4 (X Y Wave Intensity).")
    return raw


def _load_all_spectra(df, predictor):
    """Загрузить все пиксельные спектры из карты (или один для 2-колоночного)."""
    spectra_raw, spectra_proc, coords = [], [],[]
    for (x, y), pix in df.groupby(["X", "Y"]):
        pix = pix.sort_values("Wave")
        if len(pix) < 20:
            continue
        wave = pix["Wave"].values
        intn = pix["Intensity"].values
        interp = np.interp(predictor.grid, wave, intn)
        
        # preprocess_spectrum возвращает (spec, d2) - берем только спектр для отрисовок
        proc, _ = preprocess_spectrum(interp, predictor.grid, predictor.use_als, predictor.norm)
        
        spectra_raw.append(interp)
        spectra_proc.append(proc)
        coords.append((x, y))
    return (np.array(spectra_raw), np.array(spectra_proc), np.array(coords))

# ─────────────────────────────────────────────────────────
# ПРЯМАЯ ЗАДАЧА: спектр → класс
# ─────────────────────────────────────────────────────────
def _fig_forward_prediction(result, region, predictor, spectra_proc, bands):
    """Визуализация результата прямой задачи."""
    _apply_dark_theme()
    probs = result["probabilities"]
    pred  = result["prediction"]

    fig = plt.figure(figsize=(14, 8), facecolor="#0F1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Средний спектр с аннотацией полос ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    mean_spec = spectra_proc.mean(axis=0)
    std_spec  = spectra_proc.std(axis=0) if len(spectra_proc) > 1 else np.zeros_like(mean_spec)
    ax1.fill_between(predictor.grid, mean_spec - std_spec, mean_spec + std_spec,
                     alpha=0.2, color="#4A9EFF", label="±1σ (если карта)")
    ax1.plot(predictor.grid, mean_spec, color="#4A9EFF", lw=1.8, label="Средний спектр")

    band_colors =["#FF7043", "#66BB6A", "#FFD54F", "#CE93D8",
                   "#4FC3F7", "#A5D6A7", "#FFAB91"]
    for i, (lo, hi, bname) in enumerate(bands):
        mask = (predictor.grid >= lo) & (predictor.grid <= hi)
        if mask.sum() > 0:
            ax1.axvspan(lo, hi, alpha=0.12, color=band_colors[i % len(band_colors)],
                        label=bname)

    pred_color = CLASS_COLORS.get(pred, "#FFFFFF")
    ax1.set_title(f"Анализ спектров · Предсказание: {CLASS_LABELS.get(pred, pred)}",
                  fontsize=11, color=pred_color, pad=10)
    ax1.set_xlabel("Рамановский сдвиг (см⁻¹)")
    ax1.set_ylabel("Интенсивность (норм.)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=4, loc="upper right")

    # ── 2. Вероятности классов ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    cls_names = [CLASS_LABELS.get(c, c) for c in probs.keys()]
    cls_vals  = list(probs.values())
    cls_colors =[CLASS_COLORS.get(c, "#888") for c in probs.keys()]
    bars = ax2.barh(cls_names, cls_vals, color=cls_colors, edgecolor="#30363D",
                    height=0.55)
    ax2.set_xlim(0, 1)
    ax2.axvline(1/3, ls="--", color="#8B949E", lw=1, alpha=0.6, label="случайный уровень")
    for bar, val in zip(bars, cls_vals):
        ax2.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=10, color="#E6EDF3")
    ax2.set_title("Вероятности классов", fontsize=10)
    ax2.set_xlabel("P(класс)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="x", alpha=0.3)

    pred_idx = list(probs.keys()).index(pred)
    bars[pred_idx].set_edgecolor("#FFD700")
    bars[pred_idx].set_linewidth(2)

    # ── 3. Интенсивность по полосам ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    band_areas = []
    band_names_short =[]
    for lo, hi, bname in bands:
        mask = (predictor.grid >= lo) & (predictor.grid <= hi)
        if mask.sum() > 0:
            areas = np.trapz(spectra_proc[:, mask], predictor.grid[mask], axis=1)
            band_areas.append(areas.mean())
            band_names_short.append(bname)
    if band_areas:
        colors_b = [band_colors[i % len(band_colors)] for i in range(len(band_areas))]
        ax3.bar(band_names_short, band_areas, color=colors_b, edgecolor="#30363D")
        ax3.set_title("Средняя площадь полос", fontsize=10)
        ax3.set_xlabel("Полоса")
        ax3.set_ylabel("∫ I dν")
        ax3.tick_params(axis="x", rotation=45, labelsize=7)
        ax3.grid(axis="y", alpha=0.3)

    plt.suptitle("ПРЯМАЯ ЗАДАЧА: Определение экспериментальной группы",
                 fontsize=12, color="#E6EDF3", y=1.01)
    return fig


# ─────────────────────────────────────────────────────────
# ОБРАТНАЯ ЗАДАЧА: класс → спектральные маркеры
# ─────────────────────────────────────────────────────────
def _fig_inverse_task(predictor, spectra_proc, bands, result):
    """
    Обратная задача: показать, какие спектральные области
    наиболее характерны для предсказанного класса.
    """
    _apply_dark_theme()
    pred = result["prediction"]
    probs = result["probabilities"]

    band_stats = {}
    for lo, hi, bname in bands:
        mask = (predictor.grid >= lo) & (predictor.grid <= hi)
        if mask.sum() == 0:
            continue
        seg = spectra_proc[:, mask]
        band_stats[bname] = {
            "mean": seg.mean(),
            "std":  seg.std(),
            "area_mean": np.trapz(seg.mean(axis=0), predictor.grid[mask]),
            "cv":   seg.std() / (np.abs(seg.mean()) + 1e-10),
        }

    fig = plt.figure(figsize=(14, 10), facecolor="#0F1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    band_colors =["#FF7043", "#66BB6A", "#FFD54F", "#CE93D8",
                   "#4FC3F7", "#A5D6A7", "#FFAB91"]

    # ── 1. Спектр с выделением значимых полос ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    mean_spec = spectra_proc.mean(axis=0)
    std_spec  = spectra_proc.std(axis=0) if len(spectra_proc) > 1 else np.zeros_like(mean_spec)

    cv_values = {b: s["cv"] for b, s in band_stats.items()}
    max_cv = max(cv_values.values()) if cv_values else 1
    if max_cv == 0:
        max_cv = 1

    for i, (lo, hi, bname) in enumerate(bands):
        if bname not in cv_values:
            continue
        importance = cv_values[bname] / (max_cv + 1e-10)
        ax1.axvspan(lo, hi, alpha=0.08 + 0.35 * importance,
                    color=band_colors[i % len(band_colors)])
        if importance > 0.5:
            mid = (lo + hi) / 2
            ax1.axvline(mid, color=band_colors[i % len(band_colors)],
                        lw=1, ls=":", alpha=0.7)
            ax1.text(mid, mean_spec.max() * 0.92, bname,
                     ha="center", fontsize=7, rotation=90,
                     color=band_colors[i % len(band_colors)], alpha=0.85)

    ax1.plot(predictor.grid, mean_spec, color="#4A9EFF", lw=2)
    ax1.fill_between(predictor.grid, mean_spec - std_spec, mean_spec + std_spec,
                     alpha=0.15, color="#4A9EFF")
    
    pred_color = CLASS_COLORS.get(pred, "#FFFFFF")
    ax1.set_title(
        f"Спектральные маркеры для класса: {CLASS_LABELS.get(pred, pred)} "
        f"(прозрачность = вариативность полосы)",
        color=pred_color, fontsize=10, pad=8,
    )
    ax1.set_xlabel("Рамановский сдвиг (см⁻¹)")
    ax1.set_ylabel("Интенсивность (норм.)")
    ax1.grid(alpha=0.2)

    # ── 2. Карта вариативности полос ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if band_stats and max(cv_values.values()) > 0:
        bnames = list(band_stats.keys())
        cvs    = [band_stats[b]["cv"] for b in bnames]
        colors_b = [band_colors[i % len(band_colors)] for i in range(len(bnames))]
        bars = ax2.barh(bnames, cvs, color=colors_b, edgecolor="#30363D", height=0.6)
        ax2.set_title("Коэффициент вариации по полосам\n(высокий = более информативная)", fontsize=9)
        ax2.set_xlabel("CV = σ/|μ|")
        ax2.grid(axis="x", alpha=0.3)
        top3 = sorted(range(len(cvs)), key=lambda i: cvs[i], reverse=True)[:3]
        for idx in top3:
            bars[idx].set_edgecolor("#FFD700")
            bars[idx].set_linewidth(2)
    else:
        ax2.text(0.5, 0.5, "Один спектр - нет вариации", 
                 ha="center", va="center", color="#8B949E")
        ax2.axis("off")

    # ── 3. Диагностические соотношения полос ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if len(bands) >= 2:
        ratio_names, ratio_vals = [],[]
        for i in range(len(bands)):
            for j in range(i+1, len(bands)):
                lo_i, hi_i, nm_i = bands[i]
                lo_j, hi_j, nm_j = bands[j]
                m_i = (predictor.grid >= lo_i) & (predictor.grid <= hi_i)
                m_j = (predictor.grid >= lo_j) & (predictor.grid <= hi_j)
                if m_i.sum() > 0 and m_j.sum() > 0:
                    a_i = np.trapz(spectra_proc[:, m_i].mean(axis=0), predictor.grid[m_i])
                    a_j = np.trapz(spectra_proc[:, m_j].mean(axis=0), predictor.grid[m_j])
                    ratio_names.append(f"{nm_i}/{nm_j}")
                    ratio_vals.append(a_i / (a_j + 1e-10))

        if ratio_names:
            deviations =[abs(v - 1) for v in ratio_vals]
            top_idx = sorted(range(len(deviations)), key=lambda i: deviations[i], reverse=True)[:8]
            top_names =[ratio_names[i] for i in top_idx]
            top_vals  = [ratio_vals[i] for i in top_idx]
            colors_r  =["#FF7043" if v > 1 else "#66BB6A" for v in top_vals]
            ax3.barh(top_names, top_vals, color=colors_r, edgecolor="#30363D", height=0.6)
            ax3.axvline(1.0, ls="--", color="#FFD700", lw=1.2, alpha=0.8, label="=1 (равные)")
            ax3.set_title("Топ диагностических соотношений полос", fontsize=9)
            ax3.set_xlabel("I_A / I_B")
            ax3.legend(fontsize=8)
            ax3.grid(axis="x", alpha=0.3)

    # ── 4. Пространственная карта (если >1 пикселя) ───────────────────
    ax4 = fig.add_subplot(gs[2, :])

    if len(band_stats) > 0 and spectra_proc.shape[0] > 1:
        top_bands_idx = sorted(
            range(len(bands)),
            key=lambda i: cv_values.get(bands[i][2], 0),
            reverse=True
        )[:3]
        for rank, bidx in enumerate(top_bands_idx):
            lo, hi, bname = bands[bidx]
            mask = (predictor.grid >= lo) & (predictor.grid <= hi)
            if mask.sum() == 0:
                continue
            areas = np.trapz(spectra_proc[:, mask], predictor.grid[mask], axis=1)
            color = band_colors[bidx % len(band_colors)]
            ax4.hist(areas, bins=min(30, len(areas)),
                     alpha=0.55, color=color, edgecolor="#30363D",
                     label=f"{bname} ({lo:.0f}–{hi:.0f} см⁻¹)")
        ax4.set_title("Распределение площадей топ-3 полос по пикселям карты", fontsize=10)
        ax4.set_xlabel("Площадь полосы ∫ I dν")
        ax4.set_ylabel("Число пикселей")
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Недостаточно пикселей (карта не загружена)",
                 transform=ax4.transAxes, ha="center", va="center",
                 color="#8B949E", fontsize=11)
        ax4.axis("off")

    plt.suptitle("ОБРАТНАЯ ЗАДАЧА: Спектральные маркеры и диагностические признаки",
                 fontsize=12, color="#E6EDF3", y=1.01)
    return fig


# ─────────────────────────────────────────────────────────
# Пространственная тепловая карта
# ─────────────────────────────────────────────────────────
def _fig_spatial_heatmaps(spectra_proc, coords, predictor, bands):
    """Тепловые карты интенсивности полос по пространству."""
    _apply_dark_theme()
    # Если пикселей меньше 4 (например, загружен одиночный спектр из 2 колонок),
    # построение тепловой карты не имеет смысла
    if len(coords) < 4:
        return None

    xs = coords[:, 0]
    ys = coords[:, 1]
    uniq_x = np.unique(xs)
    uniq_y = np.unique(ys)

    n_bands = len(bands)
    cols = min(3, n_bands)
    rows = int(np.ceil(n_bands / cols))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4 * rows),
                             facecolor="#0F1117")
    axes = np.array(axes).flatten() if n_bands > 1 else [axes]

    for idx, (lo, hi, bname) in enumerate(bands):
        mask = (predictor.grid >= lo) & (predictor.grid <= hi)
        if mask.sum() == 0:
            axes[idx].axis("off")
            continue
        intensity = spectra_proc[:, mask].mean(axis=1)
        heat = np.full((len(uniq_y), len(uniq_x)), np.nan)
        for i in range(len(intensity)):
            xi = np.where(uniq_x == xs[i])[0][0]
            yi = np.where(uniq_y == ys[i])[0][0]
            heat[yi, xi] = intensity[i]
        ax = axes[idx]
        im = ax.imshow(heat, origin="lower", aspect="auto",
                       cmap="inferno", interpolation="nearest")
        ax.set_title(f"{bname}\n{lo:.0f}–{hi:.0f} см⁻¹", fontsize=8, pad=4)
        ax.set_xlabel("X (мкм)", fontsize=7)
        ax.set_ylabel("Y (мкм)", fontsize=7)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_bands, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Пространственные тепловые карты Рамановских полос",
                 fontsize=12, color="#E6EDF3")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
# Предобработка: raw vs processed
# ─────────────────────────────────────────────────────────
def _fig_preprocessing(df, predictor):
    """Показать эффект предобработки на примере одного спектра."""
    _apply_dark_theme()
    pix = list(df.groupby(["X", "Y"]))[0][1].sort_values("Wave")
    wave = pix["Wave"].values
    intn = pix["Intensity"].values
    interp = np.interp(predictor.grid, wave, intn)

    bl = (als_baseline(interp) if predictor.use_als else fast_baseline(interp))
    clipped = np.clip(interp - bl, 0, None)
    smoothed = savgol_filter(clipped, window_length=11, polyorder=3)
    
    # preprocess_spectrum возвращает (spec, d2)
    proc, _ = preprocess_spectrum(interp, predictor.grid, predictor.use_als, predictor.norm)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#0F1117")
    steps =[
        (wave, intn,     "#4A9EFF",  "1. Исходный спектр"),
        (predictor.grid, bl,         "#FF7043",  "2. Базовая линия"),
        (predictor.grid, smoothed,   "#66BB6A",  "3. После вычитания и сглаживания"),
        (predictor.grid, proc,       "#FFD54F",  "4. После нормировки (SNV)"),
    ]
    for ax, (x, y, c, title) in zip(axes.flatten(), steps):
        ax.plot(x, y, color=c, lw=1.5)
        if title == "2. Базовая линия":
            ax.plot(wave, intn, color="#4A9EFF", lw=1, alpha=0.5, label="Исходный")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10, color=c)
        ax.set_xlabel("Рамановский сдвиг (см⁻¹)", fontsize=8)
        ax.set_ylabel("Интенсивность", fontsize=8)
        ax.grid(alpha=0.2)
    plt.suptitle("Этапы предобработки спектра", fontsize=12, color="#E6EDF3")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
# Описание класса
# ─────────────────────────────────────────────────────────
def _class_description(pred, probs):
    descriptions = {
        "control": (
            "🔵 КОНТРОЛЬНАЯ ГРУППА\n\n"
            "Здоровая мозговая ткань без дополнительного стресса. "
            "Базовый уровень экспрессии белков теплового шока HSP70. "
            "Характеризуется стандартным профилем рамановских полос "
            "амидов и липидов."
        ),
        "endo": (
            "🟠 ЭНДОГЕННАЯ ЭКСПРЕССИЯ HSP70\n\n"
            "Повышенная экспрессия шаперона HSP70 в ответ на "
            "эндогенный стресс (гипоксия, термошок). "
            "Нейропротективная реакция. Возможно изменение соотношения "
            "амид I / амид III и липидного профиля."
        ),
        "exo": (
            "🟢 ЭКЗОГЕННАЯ ЭКСПРЕССИЯ HSP70\n\n"
            "Введение экзогенного белка HSP70 извне. "
            "Терапевтический подход с потенциально иным спектральным "
            "профилем. Ожидаются изменения в полосах CH-стретчинга "
            "и амидных областях по сравнению с контролем."
        ),
    }
    desc = descriptions.get(pred, f"Класс: {pred}")
    conf = probs.get(pred, 0)
    conf_bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
    return (
        f"{desc}\n\n"
        f"Уверенность: {conf:.1%}\n"
        f"[{conf_bar}]\n\n"
        f"Все вероятности:\n"
        + "\n".join(f"  {CLASS_LABELS.get(c,c)}: {p:.3f}" for c, p in probs.items())
    )


# ─────────────────────────────────────────────────────────
# Делегирование предсказания в Predictor
# ─────────────────────────────────────────────────────────
def _predict_from_df(df: pd.DataFrame, predictor, region: str) -> dict:
    """
    Усредняет интенсивности по всем загруженным спектрам 
    и передает в метод predict_from_array предиктора.
    """
    raw_spectra = []
    for _, pix in df.groupby(["X", "Y"], sort=False):
        pix = pix.sort_values("Wave")
        if len(pix) >= 20:
            raw_spectra.append((pix["Wave"].values, pix["Intensity"].values))

    if not raw_spectra:
        raise ValueError("Нет пригодных спектров (нужно минимум 20 точек на спектр).")

    # Интерполируем к сетке и усредняем
    pixels = np.array([np.interp(predictor.grid, s[0], s[1]) for s in raw_spectra])
    mean_intensity = pixels.mean(axis=0)

    # Модель внутри делает featurize_single_raw_spectrum (baseline -> preprocess -> predict)
    result = predictor.predict_from_array(
        wave=predictor.grid,
        intensity=mean_intensity,
        return_spectrum=True,
        n_pixels=len(raw_spectra)
    )
    result["region_used"] = region
    return result


# ─────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────
def run_analysis(
    model_type,        # "pkl" | "cnn"
    model_choice,      # Radio — preloaded PKL
    uploaded_pkl,      # File — пользовательский .pkl
    uploaded_pt,       # File — .pt веса CNN
    uploaded_meta,     # File — .pkl мета для CNN
    uploaded_txt,      # File — спектр
    region,            # str
):
    try:
        if uploaded_txt is None:
            return ("⚠ Загрузите .txt файл спектра.", None, None, None, None)

        txt_path = _save_uploaded_file(uploaded_txt)
        df       = _load_dataframe(txt_path)

        # ── Выбор и загрузка модели ──────────────────────────────────
        if model_type == "pkl":
            if uploaded_pkl is not None:
                pkl_path = _save_file_with_ext(uploaded_pkl, ".pkl")
            else:
                pkl_path = PRELOADED_MODELS.get(model_choice, "")
            if not pkl_path or not os.path.exists(pkl_path):
                return (
                    f"⚠ PKL модель не найдена: {pkl_path}\n"
                    "Выберите предустановленную или загрузите свой .pkl файл.",
                    None, None, None, None,
                )
            
            predictor = RamanMLPredictor(pkl_path)
            spectra_raw, spectra_proc, coords = _load_all_spectra(df, predictor)
            if len(spectra_proc) == 0:
                return ("⚠ Не удалось загрузить спектры из файла.", None, None, None, None)
            
            bands  = predictor.bands
            result = _predict_from_df(df, predictor, region)
            model_tag = f"ML/PKL · {os.path.basename(pkl_path)}"

        elif model_type == "cnn":
            if uploaded_pt is None or uploaded_meta is None:
                return (
                    "⚠ Для CNN модели нужно загрузить оба файла:\n"
                    "  • .pt  — веса модели\n"
                    "  • .pkl — мета-файл (cnn_meta_*.pkl)",
                    None, None, None, None,
                )
            pt_path   = _save_file_with_ext(uploaded_pt,   ".pt")
            meta_path = _save_file_with_ext(uploaded_meta, ".pkl")
            
            predictor = RamanCNNPredictor(pt_path, meta_path, device="auto")
            spectra_raw, spectra_proc, coords = _load_all_spectra(df, predictor)
            if len(spectra_proc) == 0:
                return ("⚠ Не удалось загрузить спектры из файла.", None, None, None, None)
            
            bands  = predictor.bands
            result = _predict_from_df(df, predictor, region)
            model_tag = f"CNN · {os.path.basename(pt_path)}"

        else:
            return ("⚠ Неизвестный тип модели.", None, None, None, None)

        pred  = result["prediction"]
        probs = result["probabilities"]

        # ── Текстовое описание ───────────────────────────────────────
        text_out = (
            f"[Модель: {model_tag}]\n\n"
            + _class_description(pred, probs)
        )

        # ── Графики ──────────────────────────────────────────────────
        fig_forward  = _fig_forward_prediction(result, region, predictor, spectra_proc, bands)
        fig_inverse  = _fig_inverse_task(predictor, spectra_proc, bands, result)
        fig_preproc  = _fig_preprocessing(df, predictor)
        fig_heatmaps = _fig_spatial_heatmaps(spectra_proc, coords, predictor, bands)

        return text_out, fig_forward, fig_inverse, fig_preproc, fig_heatmaps

    except Exception as e:
        tb = traceback.format_exc()
        return f"❌ Ошибка:\n{str(e)}\n\n{tb}", None, None, None, None


def _save_file_with_ext(uploaded, ext: str) -> str:
    """Сохраняет загруженный файл во временный файл с нужным расширением."""
    if isinstance(uploaded, str) and os.path.exists(uploaded):
        return uploaded
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    if hasattr(uploaded, "read"):
        data = uploaded.read()
    else:
        data = open(uploaded, "rb").read()
    if isinstance(data, str):
        data = data.encode()
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────
# CSS / стиль
# ─────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

:root {
    --bg-primary: #0D1117;
    --bg-secondary: #161B22;
    --bg-card: #21262D;
    --border: #30363D;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --accent-blue: #4A9EFF;
    --accent-orange: #FF7043;
    --accent-green: #66BB6A;
    --accent-gold: #FFD700;
}

body, .gradio-container {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #1A3A5C, #4A9EFF) !important;
    border: 1px solid var(--accent-blue) !important;
    color: #fff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-size: 0.85rem !important;
}

.gr-button-primary:hover {
    background: linear-gradient(135deg, #4A9EFF, #1A3A5C) !important;
    box-shadow: 0 0 18px rgba(74,158,255,0.4) !important;
}

label, .gr-form label {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

.gr-textbox textarea, .gr-textbox input {
    font-family: 'IBM Plex Mono', monospace !important;
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    font-size: 0.82rem !important;
}

.title-block {
    font-family: 'IBM Plex Mono', monospace;
    text-align: center;
    padding: 28px 0 12px;
    letter-spacing: 0.06em;
}

.title-block h1 {
    font-size: 1.6rem;
    color: var(--accent-blue);
    margin: 0;
    font-weight: 600;
}

.title-block p {
    color: var(--text-secondary);
    font-size: 0.78rem;
    margin: 6px 0 0;
    letter-spacing: 0.12em;
}

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--accent-gold);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
    margin: 8px 0;
}

.tab-nav button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
}
"""

# ─────────────────────────────────────────────────────────
# Интерфейс Gradio
# ─────────────────────────────────────────────────────────
with gr.Blocks(
    title="Raman Brain Tissue Classifier",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="orange",
        neutral_hue="slate",
    ),
) as demo:

    gr.HTML("""
    <div class="title-block">
        <h1>⬡ RAMAN SPECTRUM ANALYZER</h1>
        <p>классификация мозговой ткани · HSP70 экспрессия · нейропротекция</p>
    </div>
    """)

    with gr.Row():
        # ── Панель управления ─────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.HTML('<div class="section-header">// тип модели</div>')

            model_type = gr.Radio(
                choices=[("ML / PKL", "pkl"), ("CNN / PyTorch", "cnn")],
                value="pkl",
                label="Бэкенд предсказания",
            )

            # ── PKL блок ──────────────────────────────────────────────
            with gr.Group(visible=True) as pkl_group:
                gr.HTML('<div class="section-header">// ml модель (.pkl)</div>')
                model_choice = gr.Radio(
                    choices=list(PRELOADED_MODELS.keys()),
                    value=list(PRELOADED_MODELS.keys())[0],
                    label="Предустановленная модель",
                )
                uploaded_pkl = gr.File(
                    label="Или загрузите свой .pkl (необязательно)",
                    file_types=[".pkl"],
                )

            # ── CNN блок ──────────────────────────────────────────────
            with gr.Group(visible=False) as cnn_group:
                gr.HTML('<div class="section-header">// cnn модель (.pt + .pkl)</div>')
                uploaded_pt = gr.File(
                    label="Веса CNN (.pt)",
                    file_types=[".pt"],
                )
                uploaded_meta = gr.File(
                    label="Мета-файл CNN (cnn_meta_*.pkl)",
                    file_types=[".pkl"],
                )
                gr.HTML("""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                            color:#8B949E;line-height:1.6;padding:4px 0;">
                    Файлы генерируются при обучении:<br>
                    <b style="color:#4A9EFF">cnn_weights_center*.pt</b><br>
                    <b style="color:#4A9EFF">cnn_meta_center*.pkl</b>
                </div>
                """)

            gr.HTML('<div class="section-header">// данные</div>')

            uploaded_txt = gr.File(
                label="Файл спектра (.txt)",
                file_types=[".txt"],
            )

            region = gr.Dropdown(
                choices=BRAIN_REGIONS,
                value="unknown",
                label="Регион мозга",
            )

            gr.HTML('<div class="section-header">// описание регионов</div>')
            gr.HTML("""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                        color:#8B949E; line-height:1.7; padding: 6px 0;">
                <b style="color:#4A9EFF">cortex</b> — кора головного мозга<br>
                <b style="color:#FF7043">striatum</b> — полосатое тело<br>
                <b style="color:#66BB6A">cerebellum</b> — мозжечок<br>
                <b style="color:#FFD700">unknown</b> — неизвестный регион
            </div>
            """)

            run_btn = gr.Button(
                "▶  ЗАПУСТИТЬ АНАЛИЗ",
                variant="primary",
                size="lg",
            )

            gr.HTML('<div class="section-header">// результат</div>')
            out_text = gr.Textbox(
                label="Предсказание",
                lines=14,
                max_lines=20
            )

        # ── Вкладки с визуализациями ──────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs():

                with gr.Tab("📊 Прямая задача"):
                    gr.HTML("""
                    <p style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                              color:#8B949E; padding:4px 0 8px;">
                    Спектр → экспериментальная группа. Средний спектр карты,
                    вероятности классов, интенсивности полос.
                    </p>""")
                    out_forward = gr.Plot(label="")

                with gr.Tab("🔬 Обратная задача"):
                    gr.HTML("""
                    <p style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                              color:#8B949E; padding:4px 0 8px;">
                    Класс → спектральные маркеры. Какие полосы наиболее
                    диагностичны? Соотношения полос и вариативность.
                    </p>""")
                    out_inverse = gr.Plot(label="")

                with gr.Tab("🗺 Тепловые карты"):
                    gr.HTML("""
                    <p style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                              color:#8B949E; padding:4px 0 8px;">
                    Пространственное распределение интенсивности каждой
                    спектральной полосы по пикселям карты.
                    </p>""")
                    out_heatmaps = gr.Plot(label="")

                with gr.Tab("⚙ Предобработка"):
                    gr.HTML("""
                    <p style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                              color:#8B949E; padding:4px 0 8px;">
                    Пошаговая визуализация предобработки: исходный спектр,
                    базовая линия, вычитание, нормировка.
                    </p>""")
                    out_preproc = gr.Plot(label="")

    gr.HTML("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                color:#484F58; text-align:center; padding:16px 0 8px;
                border-top:1px solid #21262D; margin-top:20px;">
        Raman Brain Tissue Classifier · HSP70 Expression Analysis ·
        Confocal Raman Microscopy (Renishaw inVia Qontor)
    </div>
    """)

    # ── Переключение видимости блоков ─────────────────────────────────
    def _toggle_model_blocks(choice):
        return (
            gr.update(visible=(choice == "pkl")),
            gr.update(visible=(choice == "cnn")),
        )

    model_type.change(
        fn=_toggle_model_blocks,
        inputs=[model_type],
        outputs=[pkl_group, cnn_group],
    )

    # ── Запуск анализа ────────────────────────────────────────────────
    run_btn.click(
        fn=run_analysis,
        inputs=[
            model_type,
            model_choice,
            uploaded_pkl,
            uploaded_pt,
            uploaded_meta,
            uploaded_txt,
            region,
        ],
        outputs=[out_text, out_forward, out_inverse, out_preproc, out_heatmaps],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=8080,
        inbrowser=False,
        show_error=True,
        share=False,
    )

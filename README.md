# 🔬 Raman Spectrum Analyzer — HSP70 Expression Detection

![Analyzer](./Raman_analyzer.jpg)

Решение задачи классификации рамановских спектров мозговой ткани для определения экспериментальной группы: **контроль**, **эндогенная экспрессия HSP70**, **экзогенная экспрессия HSP70**. Разработано в рамках хакатона Nuclear IT Hack 2026

Получены на конфокальном рамановском микроскопе **Renishaw inVia Qontor** (два спектральных окна: ~1500 и ~2900 см⁻¹).

---

## 📄 Материалы

| Тип | Файл |
|-----|------|
| 📊 Презентация | [Bang Bang presentation](<./Bang Bang presentation.pdf>) |
| 📝 Статья | [Bang Bang article](./article_raman_hsp70.pdf) |

---

## 🚀 Быстрый старт

### Через Docker (рекомендуется)

```bash
git clone https://github.com/Nek1tt/Raman-Spectrum-Analyzer.git
cd Raman-Spectrum-Analyzer
docker build -t raman-classifier .
docker run -p 8080:8080 raman-classifier
```

Откройте в браузере: **http://localhost:8080**

---

## 📁 Структура репозитория

```
Raman-Spectrum-Analyzer/
│
├── app_gradio.py          # Gradio-интерфейс (прямая и обратная задача)
├── main.py                # CLI-точка входа для обучения / инференса
├── pipeline.py            # Главный пайплайн обучения (один центр + fusion)
│
├── data_loading.py        # Сканирование датасета и загрузка .txt файлов
├── preprocessing.py       # Коррекция базовой линии, сглаживание, нормировка
├── features.py            # Определение полос и извлечение признаков
├── ml_models.py           # ML-модели (Ridge, SVM, XGBoost, LightGBM …)
├── cnn_model.py           # CNN: 2-канальный 1D ResNet + SE-блок
├── evaluation.py          # Кросс-валидация: LOGO, SGKF, GSS
├── inference.py           # CLI-инференс на новых .txt файлах
├── inference_utils.py     # Предикторы (ML / CNN) и функции визуализации
├── visualisation.py       # Графики обучения (PCA, CM, saliency, SHAP)
├── constants.py           # Общие константы (цвета, диапазоны, пути)
│
├── requirements.txt
├── Dockerfile
├── .github/
│   └── workflows/
│       └── docker-check.yml   # CI: проверка сборки и запуска Docker
├── .gitignore
├── README.md
│
├── control/               # Контрольная группа
│   └── control/
│       ├── mk1/           # 12 карт → animal_id: control_1
│       ├── mk2a/          # 24 карты → animal_id: control_2a
│       ├── mk2b/          # 24 карты → animal_id: control_2b
│       └── mk3/           # 20 карт  → animal_id: control_3
│
├── endo/                  # Эндогенная экспрессия HSP70
│   └── endo/
│       ├── mend1/         # 13 карт  → animal_id: endo_1
│       ├── mend2a/        # 26 карт  → animal_id: endo_2a
│       ├── mend2b/        # 24 карты → animal_id: endo_2b
│       └── mend3/         # 12 карт  → animal_id: endo_3
│
├── exo/                   # Экзогенная экспрессия HSP70
│   └── exo/
│       ├── mexo1/         # 12 карт  → animal_id: exo_1
│       ├── mexo2a/        # 24 карты → animal_id: exo_2a
│       ├── mexo2b/        # 24 карты → animal_id: exo_2b
│       └── mexo3/         # 24 карты → animal_id: exo_3
│
└── outputs/               # Создаётся автоматически при обучении
    ├── best_model_center1500.pkl
    ├── best_model_center2900.pkl
    ├── cnn_weights_center1500.pt
    ├── cnn_weights_center2900.pt
    ├── cnn_meta_center1500.pkl
    ├── cnn_meta_center2900.pkl
    └── *.png               # Графики: PCA, матрицы ошибок, saliency
```

Каждый `.txt` файл — гиперспектральная карта в формате `X Y Wave Intensity` (4 столбца) или `Wave Intensity` (2 столбца).

---

## 🧬 Задача

| Параметр | Значение |
|----------|----------|
| Тип задачи | Многоклассовая классификация (3 класса) |
| Классы | `control`, `endo`, `exo` |
| Спектральные окна | ~927–2002 см⁻¹ (center1500), ~2650–3288 см⁻¹ (center2900) |
| Карт в датасете | 118 (40 control · 36 endo · 42 exo) |
| Пикселей | ~61 950 (~525 пикселей/карта) |

---

## ⚙️ Установка

```bash
git clone https://github.com/Nek1tt/Raman-Spectrum-Analyzer.git
cd Raman-Spectrum-Analyzer
pip install -r requirements.txt
```

> **GPU (NVIDIA CUDA):** установите нужную версию PyTorch с [pytorch.org](https://pytorch.org/get-started/locally/) перед запуском `pip install -r requirements.txt`.

---

## 🖥️ Gradio-интерфейс

```bash
python app_gradio.py
```

Открывает веб-приложение на **http://127.0.0.1:8080** с четырьмя вкладками:

| Вкладка | Описание |
|---------|----------|
| 📊 Прямая задача | Спектр → предсказание класса + вероятности |
| 🔬 Обратная задача | Класс → информативные спектральные маркеры |
| 🗺 Тепловые карты | Пространственное распределение интенсивностей полос |
| ⚙ Предобработка | Пошаговая визуализация пайплайна предобработки |

---

## 🏋️ Обучение моделей (CLI)

### Базовый запуск

```bash
python main.py --data_root ./
```

### Полный запуск (параметры из эксперимента)

```bash
python main.py \
    --data_root ./ \
    --n_grid 512 \
    --norm snv \
    --use_als \
    --fuse_bands \
    --save_plots \
    --optuna_trials_ridge 50 \
    --optuna_trials_cnn 25 \
    --optuna_cnn_epochs 15 \
    --cnn_epochs 120 \
    --cnn_batch 256 \
    --cnn_patience 30 \
    --n_jobs -1
```

### Windows (одна строка)

```cmd
python main.py --data_root ./ --n_grid 512 --norm snv --use_als --fuse_bands --save_plots --optuna_trials_ridge 50 --optuna_trials_cnn 25 --optuna_cnn_epochs 15 --cnn_epochs 120 --cnn_batch 256 --cnn_patience 30 --n_jobs -1
```

---

## 🔧 Параметры командной строки (`main.py`)

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--data_root` | `./` | Корневая папка с данными |
| `--n_grid` | `256` | Точек интерполяционной сетки |
| `--norm` | `snv` | Нормировка: `snv` / `peak_phe` / `area` / `minmax` |
| `--use_als` | `False` | Коррекция базовой линии ALS |
| `--fuse_bands` | `False` | Объединить center1500 + center2900 |
| `--save_plots` | `False` | Сохранять PNG-графики |
| `--n_jobs` | `-1` | Параллельные потоки (`-1` = все ядра) |
| `--skip_cnn` | `False` | Пропустить CNN |
| `--skip_ml` | `False` | Пропустить ML-модели |
| `--cnn_epochs` | `80` | Эпохи обучения CNN |
| `--cnn_batch` | `256` | Размер батча CNN |
| `--cnn_lr` | `1e-3` | Learning rate CNN |
| `--cnn_patience` | `15` | Early stopping patience |
| `--optuna_trials_ridge` | `20` | Испытаний Optuna для Ridge |
| `--optuna_trials_cnn` | `15` | Испытаний Optuna для CNN |
| `--optuna_cnn_epochs` | `10` | Эпох CNN в одном испытании Optuna |
| `--permutation_test` | `False` | Тест перестановок (p-value) |
| `--n_permutations` | `200` | Число перестановок |

---

## 🔍 Инференс на новых данных

```bash
# ML-модель
python main.py \
    --load_model outputs/best_model_center1500.pkl \
    --predict_dir ./new_spectra/

# CNN-модель
python main.py \
    --load_cnn outputs/cnn_weights_center1500.pt \
    --cnn_meta outputs/cnn_meta_center1500.pkl \
    --predict_dir ./new_spectra/
```

Результаты сохраняются в `outputs/predictions.csv`.

---

## 🏗️ Архитектура пайплайна

```
.txt файлы (X Y Wave Intensity)
          │
          ▼
  Интерполяция → единая сетка (n_grid точек)
          │
          ▼
  Коррекция базовой линии
  (ALS или итерационный полином)
          │
          ▼
  Сглаживание (Savitzky-Golay) + вторая производная
          │
          ▼
  Нормировка (SNV)
          │
     ┌────┴────┐
     ▼         ▼
  75 признаков    2-канальный вход (spec + d2)
  (6–7 полос ×    для CNN
  10 статистик
  + ratios)
     │         │
     ▼         ▼
  ML-модели  CNN 1D ResNet+SE
  (Ridge,    (stem → layer1→2→3
   SVM,       → GAP → FC)
   XGB,
   LGBM,
   HistGB)
     └────┬────┘
          ▼
   Кросс-валидация:
   LOGO / SGKF / GSS
          │
          ▼
   outputs/*.pkl  outputs/*.pt
```

---

## 📊 Стратегии кросс-валидации

| Стратегия | Код | Описание |
|-----------|-----|----------|
| Leave-One-Group-Out | **LOGO** | Одно животное в тесте — наиболее строгая оценка |
| StratifiedGroupKFold | **SGKF** | 4 фолда с балансом классов и группировкой по животным |
| GroupShuffleSplit | **GSS** | 10 случайных разбиений с разделением по животным |

---

## 📈 Результаты финального обучения

### center1500 (~1500 см⁻¹: белки, амиды, липиды)

| Модель | LOGO acc | LOGO std | SGKF acc | GSS acc |
|--------|----------|----------|----------|---------|
| **RidgeClf** | **0.377** | 0.258 | **0.441** | 0.362 |
| LogReg | 0.377 | 0.242 | 0.434 | 0.360 |
| CNN-1D-ResNet | 0.356 | 0.197 | — | — |
| LinearSVC | 0.341 | 0.261 | 0.433 | 0.327 |
| HistGB | 0.291 | 0.228 | 0.408 | 0.271 |
| LightGBM | 0.283 | 0.230 | 0.407 | 0.268 |
| XGBoost | 0.279 | 0.234 | 0.406 | 0.265 |

### center2900 (~2900 см⁻¹: CH-колебания, липиды)

| Модель | LOGO acc | LOGO std | SGKF acc | GSS acc |
|--------|----------|----------|----------|---------|
| **RidgeClf** | **0.426** | 0.218 | **0.436** | 0.376 |
| LogReg | 0.409 | 0.198 | 0.423 | 0.375 |
| CNN-1D-ResNet | 0.371 | 0.205 | — | — |
| LinearSVC | 0.365 | 0.209 | 0.421 | 0.310 |
| HistGB | 0.301 | 0.193 | 0.408 | 0.277 |
| LightGBM | 0.298 | 0.199 | 0.410 | 0.275 |
| XGBoost | 0.298 | 0.204 | 0.412 | 0.272 |

> Случайный уровень (3 класса): **0.333**

---

## 🖥️ Требования к оборудованию

| Компонент | Рекомендуется | Минимум |
|-----------|--------------|---------|
| CPU | 8+ ядер | 4 ядра |
| RAM | 32 GB | 16 GB |
| GPU | NVIDIA RTX 3060+ (CUDA) | CPU-режим |
| Диск | 10 GB | 5 GB |

> Полный прогон (center1500 + center2900, ML + CNN, Optuna) занял ~7 часов на **RTX 4060 Ti**.

---

## 👥 Команда разработки

- [Абрамов Никита](https://github.com/Nek1tt)
- [Абдылдаев Нуршат](https://github.com/stakanmoloka)
- [Ижденева Влада](https://github.com/izhdenevav)
- [Мартынов Богдан](https://github.com/Hom4ikTop4ik)

---

## 📧 Контакты

- https://t.me/Nek1tJO - Telegram (Абрамов Никита)
- n.abramov@g.nsu.ru (Абрамов Никита)


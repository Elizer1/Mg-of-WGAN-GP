# Deep Learning Prediction & WGAN-GP Data Augmentation (CMS Reproducibility Package)
> PyQt5 + LightGBM + CatBoost for modeling and visualization; includes three pipelines: baseline (no augmentation), WGAN-GP generator, and modeling with augmented data.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](#environment--dependencies)
[![OS](https://img.shields.io/badge/OS-Windows%2010%20Pro-informational)](#environment--dependencies)
[![Reproducible](https://img.shields.io/badge/Reproducible-Yes-brightgreen)](#minimal-reproducible-example-mre)

---

## Directory Layout (aligned with your current repo)
```
.
├── 01_origin_model_build/      # Baseline model: training & evaluation without augmentation
├── 02_wgan-gp_build/           # Train WGAN-GP (to generate synthetic samples)
├── 03_wgan-gp_model_build/     # Train/evaluate with augmented (synthetic) data
├── 04_results_sample/          # Final model outputs (for verification)
└── README.md
```

---

## Environment & Dependencies
- **OS**: Windows 10 Pro  
- **Python**: 3.11  
- **IDE (optional)**: PyCharm 2023.1  
- **Memory**: 16 GB  
- **GPU**: Ryzen 6750 GRE, 10 GB (Note: CatBoost/LightGBM use **CPU** by default; on AMD GPUs prefer CPU for stability/reproducibility.)

### Using pip (recommended)
```bash
python -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt
```
**requirements.txt (example; pin/trim as needed)**
```
pyqt5>=5.15
lightgbm>=4.3
catboost>=1.2
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.8
scipy>=1.11
pyyaml>=6.0
joblib>=1.3
tqdm>=4.66
```

> If you use `conda`, provide an `environment.yml`. For Qt issues, ensure the Microsoft VC++ runtime is installed.

---

## Data Layout & Notes
Create a `data/` folder at repo root:
```
data/
├── sample/   # Small sample data (public, included in repo)
└── raw/      # Full/raw data (omit if restricted)
```
Include a **Data Availability Statement** in the paper/repo with a DOI (Zenodo / Mendeley Data) or an access note for restricted data.

---

## Minimal Reproducible Example (MRE)

### 1) Baseline (no augmentation)
```bash
# go to baseline
cd 01_origin_model_build

# train (example command)
python train.py --config ./config.yaml --data ../data/sample/sample.csv --out ./outputs/
```

### 2) Train the WGAN-GP generator
```bash
cd ../02_wgan-gp_build

# train the generator (example)
python train_gan.py --config ./config.yaml --data ../data/sample/sample.csv --out ./outputs/

# generate synthetic samples
python generate.py --checkpoint ./outputs/best.ckpt --n_samples 2000 --out ../03_wgan-gp_model_build/synth.csv
```

### 3) Modeling with augmented data
```bash
cd ../03_wgan-gp_model_build

# (example) merge raw + synthetic data
python mix_data.py --raw ../data/sample/sample.csv --synth ./synth.csv --out ./train_aug.csv

# (example) train and evaluate with augmented data
python train.py --config ./config.yaml --data ./train_aug.csv --out ./outputs/
python evaluate.py --pred ./outputs/pred.csv --y ../data/sample/label.csv --out ./outputs/figs/
```

---

## PyQt5 GUI (optional)
To provide a single entry point for GUI execution and visualization:
```bash
python -m src.gui.main --pipeline baseline   # or: wgan, wgan_aug
```
- In the GUI, select `config.yaml` and the data path, then start training.
- Metrics and figures are saved under each pipeline’s `outputs/` directory by default.

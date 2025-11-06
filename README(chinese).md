# 深度学习预测与 WGAN-GP 数据增强（CMS 复现实验包）
> 基于 PyQt5 + LightGBM + CatBoost 的建模与可视化；包含原始基线、WGAN‑GP 生成器、增强后建模三条流水线。

[![Python](https://img.shields.io/badge/Python-3.11-blue)](#环境与依赖)
[![OS](https://img.shields.io/badge/OS-Windows%2010%20Pro-informational)](#环境与依赖)
[![Reproducible](https://img.shields.io/badge/Reproducible-Yes-brightgreen)](#复现步骤)

---

## 目录结构（已对齐你当前仓库）
```
.
├── 01_origin_model_build/      # 基线模型：不做增强的训练与评估
├── 02_wgan-gp_build/           # 训练 WGAN‑GP（生成合成样本）
├── 03_wgan-gp_model_build/     # 用合成样本增强后再训练与评估
├── 04_results_sample/          # 最终模型（用于验证）
└── README.md
```

---

## 环境与依赖
- **操作系统**：Windows 10 专业版  
- **Python**：3.11  
- **IDE（可选）**：PyCharm 2023.1  
- **内存**：16 GB  
- **显卡**：锐龙 6750 GRE 10 GB（注意：CatBoost/LightGBM 默认采用 **CPU**；如需 GPU 请另行配置相应后端，AMD 平台通常使用 CPU 更稳妥）

### 使用 pip（推荐）
```bash
python -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt
```
**requirements.txt（示例，按需精简/锁定版本）**
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

> 如使用 `conda`，可提供 `environment.yml`；若包含 Qt 相关问题，先确保系统安装了 VC++ 运行库。

---

## 数据放置与说明
- 建议在项目根目录新增 `data/`：
```
data/
├── sample/             # 小样例数据（可公开、仓库内提供）
└── raw/                # 原始/完整数据（如受限可不提交）
```
- 在论文/仓库中提供 **Data Availability Statement**：给出公开仓库与 DOI（如 Zenodo / Mendeley Data）或受限说明。

---

## 复现步骤（Minimal Reproducible Example, MRE）

### 1) 纯基线（不增强）
```bash
# 进入基线目录
cd 01_origin_model_build

# 训练（示例命令）
python train.py --config ./config.yaml --data ../data/sample/sample.csv --out ./outputs/

### 2) 训练 WGAN‑GP 生成器
```bash
cd ../02_wgan-gp_build

# 训练生成器（示例命令）
python train_gan.py --config ./config.yaml --data ../data/sample/sample.csv --out ./outputs/

# 生成合成样本
python generate.py --checkpoint ./outputs/best.ckpt --n_samples 2000 --out ../03_wgan-gp_model_build/synth.csv
```

### 3) 数据增强后建模
```bash
cd ../03_wgan-gp_model_build

---

## PyQt5 图形界面（可选）
如需 GUI 一键运行与结果展示，建议将入口统一为：
```bash
python -m src.gui.main --pipeline baseline   # 或 wgan, wgan_aug
```
- 在 GUI 中选择 `config.yaml` 与数据路径后启动训练；
- 输出的指标与图像默认保存到对应子目录下的 `outputs/`。





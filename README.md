# MindBalance v2 🧘
### A Data-Driven Model Integrating Indian Knowledge Systems and Psychological Variables for Student Stress Prediction
*BSc Data Science | Final Year | IKS Integration Project*

---

## Project Overview

MindBalance v2 is a full machine learning pipeline that predicts student stress tiers (Low / Moderate / High) using 24 features across 5 domains (Psychological, Physiological, Environmental, Academic, Social) augmented with 4 IKS variables (Yoga frequency, Yoga duration, Mindfulness, Pranayama). The best model achieves **76.8% accuracy**, **ROC-AUC 0.889**, and **Cohen's Kappa 0.653** (substantial agreement).

---

## Project Structure

```
mindbalance_v2/
├── data/
│   └── StressLevelDataset.csv     1,100 students, 24 features + target
├── src/
│   ├── config.py                  Central configuration, feature groups, palette
│   ├── pipeline.py                sklearn ColumnTransformer + SMOTE ImbPipeline
│   ├── eda.py                     Full EDA: distributions, heatmap, boxplots, IKS scatter
│   ├── train.py                   5-model comparison + GridSearchCV tuning
│   ├── evaluate.py                Confusion matrix, ROC-AUC, PR curves, Cohen's Kappa
│   ├── shap_explainer.py          SHAP global bar, per-class, waterfall, dependence plot
│   └── iks_engine.py              Scored IKS Wellness Engine + population analysis
├── outputs/                       All 16 charts auto-saved here
├── models/
│   ├── best_model.pkl             Saved best model pipeline
│   └── metrics.json               Final evaluation metrics
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

python src/eda.py            # EDA — 4 charts
python src/train.py          # Train 5 models, tune best — 1 comparison chart
python src/evaluate.py       # Full evaluation suite — 4 charts
python src/shap_explainer.py # SHAP explanations — 4 charts
python src/iks_engine.py --demo         # Population analysis + 2 demo profiles
python src/iks_engine.py --interactive  # Enter your own data
```

---

## Model Results

| Model | Accuracy | F1 (weighted) | ROC-AUC | Cohen's κ | CV Accuracy |
|-------|----------|---------------|---------|-----------|-------------|
| **Logistic Regression** ✓ | **76.8%** | **75.9%** | **0.889** | **0.653** | **70.1%** |
| XGBoost | 64.1% | 63.0% | 0.823 | 0.462 | 60.6% |
| SVM (RBF) | 67.3% | 66.7% | 0.853 | 0.508 | 64.8% |
| Random Forest | 56.8% | 55.5% | 0.771 | 0.350 | 58.4% |
| KNN | 46.4% | 45.5% | 0.648 | 0.197 | 46.9% |

---

## IKS Wellness Score

Each student receives a scored IKS Wellness metric (0–100) based on:

| Component | Weight | Max |
|-----------|--------|-----|
| Yoga frequency (days/week) | 35% | 7 |
| Mindfulness (mins/day) | 25% | 60 |
| Yoga duration (mins/session) | 20% | 60 |
| Pranayama (binary) | 20% | 1 |

Mean IKS score is **57.8 for Low-stress** vs **46.1 for High-stress** students — confirming the protective effect of IKS practice.

---

## Key Finding

Yoga Days/Week is the **5th most important predictor** of stress tier (SHAP analysis), ranking ahead of academic and social variables. This validates the IKS hypothesis: consistent Yoga and Pranayama practice is a measurable protective factor against student stress.

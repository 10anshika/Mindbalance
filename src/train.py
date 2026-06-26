"""
MindBalance v2 | train.py
Compares 5 classifiers, tunes the best with GridSearchCV, saves model.

Run: python src/train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import pickle, json, time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              cohen_kappa_score, classification_report)

from pipeline import build_pipeline, load_data
from config import DATA_PATH, MODEL_DIR, OUTPUT_DIR, LABEL_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NAVY  = "#1B3A6B"
GOLD  = "#C8973A"
SLATE = "#4A5568"

# ── Load ─────────────────────────────────────────────────────────────────────
X, y = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("=" * 60)
print("  MindBalance v2 — Model Training & Comparison")
print("=" * 60)
print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}  |  Features: {X.shape[1]}")
print(f"  Class distribution (train): {dict(y_train.value_counts().sort_index())}\n")

# ── Model zoo ────────────────────────────────────────────────────────────────
MODELS = {
    "Random Forest":      RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost":            XGBClassifier(n_estimators=200, random_state=42,
                                        eval_metric="mlogloss", verbosity=0),
    "Logistic Regression":LogisticRegression(max_iter=1000, random_state=42),
    "SVM (RBF)":          SVC(probability=True, random_state=42),
    "KNN":                KNeighborsClassifier(n_neighbors=7),
}

cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows  = []
best_name, best_score, best_pipe = None, -1, None

for name, clf in MODELS.items():
    t0   = time.time()
    pipe = build_pipeline(clf, use_smote=True)
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, average="weighted")
    kappa   = cohen_kappa_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    cv_mean = cross_val_score(pipe, X_train, y_train, cv=cv,
                               scoring="accuracy", n_jobs=-1).mean()
    elapsed = time.time() - t0

    print(f"  {name:<22}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  k={kappa:.3f}  CV={cv_mean:.3f}  [{elapsed:.1f}s]")
    rows.append({"Model": name, "Accuracy": acc, "F1 (weighted)": f1,
                 "ROC-AUC": auc, "Cohen Kappa": kappa, "CV Accuracy": cv_mean})

    if cv_mean > best_score:
        best_score, best_name, best_pipe = cv_mean, name, pipe

print(f"\n  Best model: {best_name}  (CV={best_score:.3f})\n")

# ── Save comparison table ─────────────────────────────────────────────────────
results_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
results_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

# ── Plot model comparison ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
metrics  = ["Accuracy", "F1 (weighted)", "ROC-AUC", "Cohen Kappa"]
x        = np.arange(len(results_df))
width    = 0.2
colors   = [NAVY, GOLD, "#2F855A", "#C53030"]

for i, (metric, color) in enumerate(zip(metrics, colors)):
    bars = ax.bar(x + i*width, results_df[metric], width,
                  label=metric, color=color, alpha=0.88, edgecolor="white")

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(results_df["Model"], rotation=15, ha="right", fontsize=10)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model Comparison — 5 Classifiers across 4 Metrics",
             fontsize=13, fontweight="bold", color=NAVY, pad=12)
ax.legend(fontsize=10, frameon=True, loc="lower right")
ax.spines[["top","right"]].set_visible(False)
ax.set_facecolor("#F7F8FA")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] Saved: 01_model_comparison.png")

# ── GridSearchCV on best model ────────────────────────────────────────────────
print(f"\n  Tuning {best_name} with GridSearchCV (5-fold)...")

if "Random Forest" in best_name:
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth":    [8, 12, None],
        "model__min_samples_leaf": [1, 3, 5],
    }
elif "XGBoost" in best_name:
    param_grid = {
        "model__n_estimators":  [100, 200],
        "model__max_depth":     [4, 6, 8],
        "model__learning_rate": [0.05, 0.1, 0.2],
    }
else:
    param_grid = {}

if param_grid:
    gs = GridSearchCV(best_pipe, param_grid, cv=cv, scoring="roc_auc_ovr",
                      n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best_pipe = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV AUC: {gs.best_score_:.4f}")

# ── Final evaluation on tuned model ──────────────────────────────────────────
y_pred  = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)
print("\n  Final Classification Report:")
print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

# ── Save model + metrics ──────────────────────────────────────────────────────
with open(f"{MODEL_DIR}/best_model.pkl", "wb") as f:
    pickle.dump(best_pipe, f)

final_metrics = {
    "best_model":    best_name,
    "accuracy":      round(accuracy_score(y_test, y_pred), 4),
    "f1_weighted":   round(f1_score(y_test, y_pred, average="weighted"), 4),
    "roc_auc":       round(roc_auc_score(y_test, y_proba, multi_class="ovr"), 4),
    "cohen_kappa":   round(cohen_kappa_score(y_test, y_pred), 4),
}
with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

print(f"\n  [OK] Model saved: models/best_model.pkl")
print(f"  Accuracy={final_metrics['accuracy']:.3f}  AUC={final_metrics['roc_auc']:.3f}  k={final_metrics['cohen_kappa']:.3f}")
print("\n  Next: python src/evaluate.py")

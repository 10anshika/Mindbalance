"""
MindBalance v2 | evaluate.py
Full evaluation: Confusion matrix, ROC-AUC curves, PR curves, Cohen's Kappa.

Run: python src/evaluate.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize

from pipeline import load_data
from config import DATA_PATH, MODEL_DIR, OUTPUT_DIR, LABEL_NAMES, PALETTE

NAVY = "#1B3A6B"; GOLD = "#C8973A"; SLATE = "#4A5568"

# ── Load ──────────────────────────────────────────────────────────────────────
X, y = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

with open(f"{MODEL_DIR}/best_model.pkl", "rb") as f:
    pipe = pickle.load(f)

y_pred  = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)
y_bin   = label_binarize(y_test, classes=[0,1,2])

print("=" * 55)
print("  MindBalance v2 — Evaluation Suite")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
print(f"  Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}\n")

# ── Plot 1: Confusion Matrix ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix",
             fontsize=13, fontweight="bold", color=NAVY, pad=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_confusion_matrix.png", dpi=150)
plt.close()
print("  ✓ 02_confusion_matrix.png")

# ── Plot 2: ROC-AUC curves (One-vs-Rest per class) ───────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
colors = PALETTE

for i, (label, color) in enumerate(zip(LABEL_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random baseline")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — One-vs-Rest (per Stress Tier)",
             fontsize=13, fontweight="bold", color=NAVY, pad=10)
ax.legend(fontsize=10, frameon=True)
ax.spines[["top","right"]].set_visible(False)
ax.set_facecolor("#F7F8FA")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_roc_curves.png", dpi=150)
plt.close()
print("  ✓ 03_roc_curves.png")

# ── Plot 3: Precision-Recall curves ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for i, (label, color) in enumerate(zip(LABEL_NAMES, colors)):
    prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
    ap = average_precision_score(y_bin[:, i], y_proba[:, i])
    ax.plot(rec, prec, color=color, lw=2, label=f"{label} (AP = {ap:.3f})")

ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curves — Per Stress Tier",
             fontsize=13, fontweight="bold", color=NAVY, pad=10)
ax.legend(fontsize=10, frameon=True)
ax.spines[["top","right"]].set_visible(False)
ax.set_facecolor("#F7F8FA")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_pr_curves.png", dpi=150)
plt.close()
print("  ✓ 04_pr_curves.png")

# ── Plot 4: Per-class metrics bar chart ──────────────────────────────────────
from sklearn.metrics import precision_score, recall_score, f1_score as f1s
metrics_data = {
    "Precision": precision_score(y_test, y_pred, average=None),
    "Recall":    recall_score(y_test, y_pred, average=None),
    "F1-Score":  f1s(y_test, y_pred, average=None),
}
x     = np.arange(3)
width = 0.25
m_colors = [NAVY, GOLD, "#2F855A"]

fig, ax = plt.subplots(figsize=(8, 5))
for i, (metric, vals) in enumerate(metrics_data.items()):
    bars = ax.bar(x + i*width, vals, width, label=metric,
                  color=m_colors[i], alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.2f}", ha="center", fontsize=9, color=NAVY)

ax.set_xticks(x + width)
ax.set_xticklabels(LABEL_NAMES, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Precision · Recall · F1 per Stress Tier",
             fontsize=13, fontweight="bold", color=NAVY, pad=10)
ax.legend(fontsize=10, frameon=True)
ax.spines[["top","right"]].set_visible(False)
ax.set_facecolor("#F7F8FA")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_per_class_metrics.png", dpi=150)
plt.close()
print("  ✓ 05_per_class_metrics.png")

print("\n  ✅ Evaluation complete → Next: python src/shap_explainer.py")

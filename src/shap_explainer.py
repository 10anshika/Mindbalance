"""
MindBalance v2 | shap_explainer.py
SHAP global (beeswarm + bar) + individual (waterfall) explanations.

Run: python src/shap_explainer.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from pipeline import load_data
from config import DATA_PATH, MODEL_DIR, OUTPUT_DIR, ALL_FEATURES, LABEL_NAMES
from config import PSYCHOLOGICAL, PHYSIOLOGICAL, ENVIRONMENTAL, ACADEMIC, SOCIAL, IKS

NAVY = "#1B3A6B"; GOLD = "#C8973A"

FEATURE_LABELS = {
    "anxiety_level":          "Anxiety Level",
    "self_esteem":            "Self-Esteem",
    "mental_health_history":  "Mental Health History",
    "depression":             "Depression Score",
    "headache":               "Headache Frequency",
    "blood_pressure":         "Blood Pressure",
    "sleep_quality":          "Sleep Quality",
    "breathing_problem":      "Breathing Problems",
    "noise_level":            "Noise Level",
    "living_conditions":      "Living Conditions",
    "safety":                 "Safety",
    "basic_needs":            "Basic Needs Met",
    "academic_performance":   "Academic Performance",
    "study_load":             "Study Load",
    "teacher_student_rel":    "Teacher-Student Relation",
    "future_career_concerns": "Future Career Concerns",
    "social_support":         "Social Support",
    "peer_pressure":          "Peer Pressure",
    "extracurricular_act":    "Extracurricular Activity",
    "bullying":               "Bullying",
    "yoga_days_per_week":     "Yoga Days/Week",
    "yoga_duration_mins":     "Yoga Duration (mins)",
    "mindfulness_mins_day":   "Mindfulness (mins/day)",
    "pranayama_practice":     "Pranayama Practice",
}

GROUP_COLOR = {}
for f in PSYCHOLOGICAL: GROUP_COLOR[f] = "#C53030"
for f in PHYSIOLOGICAL: GROUP_COLOR[f] = "#B7791F"
for f in ENVIRONMENTAL: GROUP_COLOR[f] = "#2B6CB0"
for f in ACADEMIC:      GROUP_COLOR[f] = "#553C9A"
for f in SOCIAL:        GROUP_COLOR[f] = "#2F855A"
for f in IKS:           GROUP_COLOR[f] = GOLD

# ── Load model & data ─────────────────────────────────────────────────────────
X, y = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

with open(f"{MODEL_DIR}/best_model.pkl", "rb") as f:
    pipe = pickle.load(f)

# Get the preprocessed test data for SHAP
preprocessor = pipe.named_steps["preprocessor"]
X_test_proc  = preprocessor.transform(X_test)
model        = pipe.named_steps["model"]

# Reconstruct feature names after ColumnTransformer
from config import WIDE_SCALE, NARROW_SCALE
feat_names_proc = WIDE_SCALE + NARROW_SCALE

# Also keep raw data for display
X_test_raw = X_test.copy()

print("=" * 55)
print("  MindBalance v2 — SHAP Explainability")
print("=" * 55)

# ── SHAP Explainer ────────────────────────────────────────────────────────────
print("  Computing SHAP values...")
try:
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_proc)
    # Tree: list of arrays OR 3D array
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    is_tree = True
except Exception:
    explainer   = shap.KernelExplainer(model.predict_proba, X_test_proc[:30])
    shap_raw    = explainer.shap_values(X_test_proc[:100])
    # KernelExplainer returns (n, features, classes) for multi-class
    if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        shap_values = [shap_raw[:, :, i] for i in range(shap_raw.shape[2])]
    elif isinstance(shap_raw, list):
        shap_values = shap_raw
    else:
        shap_values = [shap_raw]
    is_tree = False
    X_test_proc = X_test_proc[:100]  # match subset

print("  ✓ SHAP values computed")

# ── Plot 1: Global bar — mean |SHAP| across all classes ──────────────────────
# shap_values is now always a list of 3 arrays, each (n, n_features)
mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)

feat_names_proc = WIDE_SCALE + NARROW_SCALE

importance_df = pd.DataFrame({
    "feature":    feat_names_proc,
    "importance": mean_abs,
    "label":      [FEATURE_LABELS.get(f, f) for f in feat_names_proc],
    "color":      [GROUP_COLOR.get(f, "#718096") for f in feat_names_proc],
}).sort_values("importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(importance_df["label"], importance_df["importance"],
               color=importance_df["color"], edgecolor="white", height=0.65)
for bar, val in zip(bars, importance_df["importance"]):
    ax.text(val + 0.001, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9, color=NAVY)

# Legend
from matplotlib.patches import Patch
legend_items = [
    Patch(color="#C53030", label="Psychological"),
    Patch(color="#B7791F", label="Physiological"),
    Patch(color="#2B6CB0", label="Environmental"),
    Patch(color="#553C9A", label="Academic"),
    Patch(color="#2F855A", label="Social"),
    Patch(color=GOLD,      label="IKS / Yoga"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=9, frameon=True)
ax.set_title("SHAP Feature Importance — Mean |SHAP| (Top 15 Features)",
             fontsize=13, fontweight="bold", color=NAVY, pad=12)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.spines[["top","right"]].set_visible(False)
ax.set_facecolor("#F7F8FA")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_shap_global_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 06_shap_global_bar.png")

# ── Plot 2: Per-class SHAP bar (top 8 per class side-by-side) ────────────────
class_colors = ["#2B6CB0", "#B7791F", "#C53030"]
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)

for cls in range(3):
    sv = shap_values[cls]   # always list now
    mean_abs_cls = np.abs(sv).mean(axis=0)
    idx   = np.argsort(mean_abs_cls)[-8:]
    labels = [FEATURE_LABELS.get(feat_names_proc[i], feat_names_proc[i]) for i in idx]
    vals   = mean_abs_cls[idx]
    colors = [GROUP_COLOR.get(feat_names_proc[i], "#718096") for i in idx]

    axes[cls].barh(labels, vals, color=colors, edgecolor="white", height=0.6)
    axes[cls].set_title(f"{LABEL_NAMES[cls]} Stress\nTop SHAP Drivers",
                        fontsize=12, fontweight="bold", color=class_colors[cls])
    axes[cls].set_xlabel("Mean |SHAP|", fontsize=10)
    axes[cls].spines[["top","right"]].set_visible(False)
    axes[cls].set_facecolor("#F7F8FA")

fig.suptitle("Per-Class SHAP Feature Importance", fontsize=14,
             fontweight="bold", color=NAVY, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_shap_per_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 07_shap_per_class.png")

# ── Plot 3: Individual waterfall — one High-stress student ────────────────────
# Find a high-stress student in test set
y_pred_test = pipe.predict(X_test)
# Match subset size if kernel explainer was used
n_shap = shap_values[0].shape[0]
y_test_sub  = y_test.values[:n_shap]
y_pred_sub  = y_pred_test[:n_shap]
high_idx    = np.where((y_test_sub == 2) & (y_pred_sub == 2))[0]
if len(high_idx) > 0:
    idx = high_idx[0]
    sv_high = shap_values[2][idx]
    base    = (explainer.expected_value[2]
               if isinstance(explainer.expected_value, (list, np.ndarray))
               else explainer.expected_value)

    # Manual waterfall (top 10 contributors)
    contrib = pd.DataFrame({"feature": [FEATURE_LABELS.get(f,f) for f in feat_names_proc],
                             "shap":    sv_high})
    contrib = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index).head(10)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = ["#C53030" if v > 0 else "#2B6CB0" for v in contrib["shap"]]
    bars    = ax.barh(contrib["feature"], contrib["shap"], color=colors,
                      edgecolor="white", height=0.6)
    ax.axvline(0, color=NAVY, linewidth=1)
    for bar, val in zip(bars, contrib["shap"]):
        ax.text(val + (0.001 if val >= 0 else -0.001),
                bar.get_y()+bar.get_height()/2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=9, color=NAVY)

    ax.set_title("SHAP Waterfall — Individual High-Stress Student Explanation",
                 fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax.set_xlabel("SHAP Value (impact on High-Stress prediction)", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#C53030", label="Increases stress →"),
                        Patch(color="#2B6CB0", label="Decreases stress ←")],
              fontsize=9, frameon=True)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#F7F8FA")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_shap_waterfall_individual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 08_shap_waterfall_individual.png")

# ── Plot 4: IKS SHAP dependence — Yoga Days vs SHAP for High-Stress ──────────
yoga_idx = feat_names_proc.index("yoga_days_per_week") if "yoga_days_per_week" in feat_names_proc else None
if yoga_idx is not None:
    sv_high_all = shap_values[2]   # always list now
    yoga_raw    = X_test_proc[:, yoga_idx]
    yoga_shap   = sv_high_all[:, yoga_idx]

    mindful_idx = feat_names_proc.index("mindfulness_mins_day") if "mindfulness_mins_day" in feat_names_proc else None
    mindful_raw = X_test_proc[:, mindful_idx] if mindful_idx else np.zeros(len(yoga_raw))

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(yoga_raw, yoga_shap, c=mindful_raw,
                    cmap="RdYlGn", alpha=0.6, s=30, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Mindfulness (scaled)", fontsize=10)
    ax.axhline(0, color=NAVY, linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Yoga Days/Week (scaled)", fontsize=11)
    ax.set_ylabel("SHAP value (impact on High-Stress class)", fontsize=11)
    ax.set_title("SHAP Dependence: Yoga Practice → High-Stress Prediction\n(coloured by Mindfulness intensity)",
                 fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#F7F8FA")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_shap_yoga_dependence.png", dpi=150)
    plt.close()
    print("  ✓ 09_shap_yoga_dependence.png")

print("\n  ✅ SHAP complete → Next: python src/iks_engine.py")

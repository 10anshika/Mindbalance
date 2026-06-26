"""
MindBalance v2 | eda.py
Full EDA: distributions, correlations, IKS analysis, class balance.

Run: python src/eda.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import (DATA_PATH, OUTPUT_DIR, LABEL_NAMES, PALETTE,
                    PSYCHOLOGICAL, PHYSIOLOGICAL, ENVIRONMENTAL, ACADEMIC, SOCIAL, IKS,
                    ALL_FEATURES, NAVY, GOLD, SLATE)

df = pd.read_csv(DATA_PATH)
df["stress_label"] = df["stress_level"].map({0:"Low",1:"Moderate",2:"High"})

print("=" * 55)
print("  MindBalance v2 — Exploratory Data Analysis")
print("=" * 55)
print(f"  Shape: {df.shape}")
print(f"  Null values: {df.isnull().sum().sum()}")
print(f"\n  Class Distribution:")
print(df["stress_level"].value_counts().sort_index().to_string())

# ── Plot 1: Class distribution ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
counts = df["stress_level"].value_counts().sort_index()
colors = PALETTE
bars   = ax.bar([LABEL_NAMES[i] for i in counts.index], counts.values,
                color=colors, edgecolor="white", width=0.55)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            str(val), ha="center", fontsize=12, fontweight="bold", color=NAVY)
ax.set_title("Stress Level Distribution (n=1,100)", fontsize=13, fontweight="bold", color=NAVY, pad=10)
ax.set_ylabel("Number of Students"); ax.set_ylim(0, counts.max()+60)
ax.spines[["top","right"]].set_visible(False); ax.set_facecolor("#F7F8FA")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_01_class_distribution.png", dpi=150)
plt.close(); print("  ✓ eda_01_class_distribution.png")

# ── Plot 2: Full correlation heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))
corr    = df[ALL_FEATURES + ["stress_level"]].corr()
mask    = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
            center=0, linewidths=0.4, linecolor="white",
            annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Correlation Heatmap — All 24 Features + Stress Level",
             fontsize=13, fontweight="bold", color=NAVY, pad=12)
plt.xticks(rotation=40, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_02_correlation_heatmap.png", dpi=150)
plt.close(); print("  ✓ eda_02_correlation_heatmap.png")

# ── Plot 3: Feature category group boxplots ───────────────────────────────────
groups = [("Psychological", PSYCHOLOGICAL), ("IKS / Yoga", IKS),
          ("Physiological", PHYSIOLOGICAL), ("Academic", ACADEMIC)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
tier_colors = PALETTE

for ax, (group_name, features) in zip(axes.flatten(), groups):
    # Normalise to 0-1 for comparability
    sub = df[features + ["stress_level"]].copy()
    for f in features:
        sub[f] = (sub[f] - sub[f].min()) / (sub[f].max() - sub[f].min() + 1e-9)
    melted = sub.melt(id_vars="stress_level", value_vars=features,
                      var_name="Feature", value_name="Normalised Value")

    for tier, color in zip([0,1,2], tier_colors):
        data_tier = [melted[melted["stress_level"]==tier][melted["Feature"]==f]["Normalised Value"].values
                     for f in features]
        positions = np.arange(len(features)) + tier*0.28 - 0.28
        bp = ax.boxplot(data_tier, positions=positions, widths=0.22, patch_artist=True,
                        medianprops={"color":"white","linewidth":1.5})
        for patch in bp["boxes"]: patch.set_facecolor(color); patch.set_alpha(0.75)

    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels([f.replace("_"," ").title() for f in features],
                       rotation=30, ha="right", fontsize=9)
    ax.set_title(f"{group_name} Features by Stress Tier",
                 fontsize=12, fontweight="bold", color=NAVY)
    ax.set_ylabel("Normalised Value (0–1)"); ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#F7F8FA")

legend_patches = [mpatches.Patch(color=PALETTE[i], label=LABEL_NAMES[i]) for i in range(3)]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=11,
           title="Stress Tier", frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Feature Distributions by Stress Tier — Normalised Comparison",
             fontsize=13, fontweight="bold", color=NAVY, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_03_feature_groups_boxplot.png", dpi=150, bbox_inches="tight")
plt.close(); print("  ✓ eda_03_feature_groups_boxplot.png")

# ── Plot 4: IKS variables scatter matrix ─────────────────────────────────────
iks_df = df[IKS + ["stress_level"]].copy()
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
pairs = [("yoga_days_per_week","mindfulness_mins_day"),
         ("yoga_duration_mins","mindfulness_mins_day"),
         ("yoga_days_per_week","yoga_duration_mins"),
         ("mindfulness_mins_day","pranayama_practice")]

for ax, (x_feat, y_feat) in zip(axes.flatten(), pairs):
    for tier, color, name in zip([0,1,2], PALETTE, LABEL_NAMES):
        sub = df[df["stress_level"]==tier]
        jx  = sub[x_feat] + np.random.uniform(-0.1, 0.1, len(sub))
        jy  = sub[y_feat] + np.random.uniform(-0.1, 0.1, len(sub))
        ax.scatter(jx, jy, c=color, alpha=0.4, s=18, label=name, edgecolors="none")
    ax.set_xlabel(x_feat.replace("_"," ").title(), fontsize=10)
    ax.set_ylabel(y_feat.replace("_"," ").title(), fontsize=10)
    ax.set_title(f"{x_feat.replace('_',' ').title()} vs\n{y_feat.replace('_',' ').title()}",
                 fontsize=11, fontweight="bold", color=NAVY)
    ax.spines[["top","right"]].set_visible(False); ax.set_facecolor("#F7F8FA")

handles = [mpatches.Patch(color=PALETTE[i], label=LABEL_NAMES[i]) for i in range(3)]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False,
           bbox_to_anchor=(0.5,-0.02))
plt.suptitle("IKS Variable Relationships by Stress Tier",
             fontsize=13, fontweight="bold", color=NAVY, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_04_iks_scatter.png", dpi=150, bbox_inches="tight")
plt.close(); print("  ✓ eda_04_iks_scatter.png")

print("\n  ✅ EDA complete → Next: python src/train.py")

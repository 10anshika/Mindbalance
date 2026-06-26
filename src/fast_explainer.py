"""
MindBalance v2 | fast_explainer.py
Fast feature contribution estimator — no SHAP dependency.

Method
------
For each feature we compute:
  deviation_i  = preprocessed(user)[i] - preprocessed(baseline)[i]
  contribution_i = deviation_i × net_coef_i

where net_coef = coef_[HIGH_class] - coef_[LOW_class] from the fitted
Logistic Regression, and baseline = mean feature values of low-stress
students in the training data.

A positive contribution means that feature is pushing your risk toward
High Stress relative to the healthy-student average.
A negative contribution means that feature is protective.

Falls back to raw normalised deviation if the model is not LogisticRegression.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from config import (ALL_FEATURES, WIDE_SCALE, NARROW_SCALE,
                    PSYCHOLOGICAL, PHYSIOLOGICAL, ENVIRONMENTAL,
                    ACADEMIC, SOCIAL, IKS, DATA_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# Feature display names (user-facing, no jargon)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DISPLAY = {
    "anxiety_level":          "Anxiety level",
    "self_esteem":            "Self-esteem",
    "mental_health_history":  "Mental health history",
    "depression":             "Depression score",
    "headache":               "Headache frequency",
    "blood_pressure":         "Blood pressure",
    "sleep_quality":          "Sleep quality",
    "breathing_problem":      "Breathing difficulties",
    "noise_level":            "Noise level",
    "living_conditions":      "Living conditions",
    "safety":                 "Sense of safety",
    "basic_needs":            "Basic needs met",
    "academic_performance":   "Academic performance",
    "study_load":             "Study workload",
    "teacher_student_rel":    "Teacher relationship",
    "future_career_concerns": "Career concerns",
    "social_support":         "Social support",
    "peer_pressure":          "Peer pressure",
    "extracurricular_act":    "Extracurricular activity",
    "bullying":               "Bullying",
    "yoga_days_per_week":     "Yoga frequency",
    "yoga_duration_mins":     "Yoga session length",
    "mindfulness_mins_day":   "Daily mindfulness",
    "pranayama_practice":     "Pranayama practice",
}

# Domain → (label, color)
DOMAIN_INFO = {}
for f in PSYCHOLOGICAL:  DOMAIN_INFO[f] = ("Psychological",  "#C53030")
for f in PHYSIOLOGICAL:  DOMAIN_INFO[f] = ("Physiological",  "#B7791F")
for f in ENVIRONMENTAL:  DOMAIN_INFO[f] = ("Environment",    "#2B6CB0")
for f in ACADEMIC:       DOMAIN_INFO[f] = ("Academic",       "#2B6CB0")
for f in SOCIAL:         DOMAIN_INFO[f] = ("Social",         "#2B6CB0")
for f in IKS:            DOMAIN_INFO[f] = ("Wellness Practice", "#C8973A")

# Feature order after ColumnTransformer (WIDE first, then NARROW)
TRANSFORMED_ORDER = WIDE_SCALE + NARROW_SCALE  # length 24


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_healthy_baseline(df: pd.DataFrame) -> pd.Series:
    """
    Return mean feature values of low-stress students (stress_level == 0)
    as a Series indexed by ALL_FEATURES.
    """
    low = df[df["stress_level"] == 0][ALL_FEATURES]
    return low.mean()


def get_feature_contributions(
    pipe,
    user_row: dict,
    healthy_baseline: pd.Series,
    top_n: int = 5,
) -> list[dict]:
    """
    Compute signed feature contributions for a single student profile.

    Parameters
    ----------
    pipe             : fitted imblearn / sklearn Pipeline
    user_row         : dict with all 24 feature values
    healthy_baseline : pd.Series — mean values for low-stress students
    top_n            : how many top features to return

    Returns
    -------
    List of dicts (length top_n), sorted by |contribution| descending:
        feature      : raw feature name
        label        : user-facing label
        contribution : float  (+ = raises stress risk, - = protective)
        domain       : domain label string
        color        : hex colour string
        annotation   : one-sentence plain-English explanation
    """
    preprocessor = pipe.named_steps["preprocessor"]

    # Build DataFrames in the ALL_FEATURES column order
    user_df = pd.DataFrame([user_row])[ALL_FEATURES]
    base_df = healthy_baseline[ALL_FEATURES].to_frame().T.reset_index(drop=True)

    # Transform through preprocessing (no SMOTE in transform path)
    user_t = preprocessor.transform(user_df)[0]
    base_t = preprocessor.transform(base_df)[0]

    deviation = user_t - base_t  # shape (24,)

    # Build per-feature weight vector
    model = pipe.named_steps["model"]
    try:
        # Logistic Regression: net coefficient = coef[High] - coef[Low]
        net_coef = model.coef_[2] - model.coef_[0]  # shape (24,)
    except AttributeError:
        try:
            # Tree-based: feature_importances_ has no sign → use deviation sign
            importances = model.feature_importances_
            net_coef = importances * np.sign(deviation + 1e-9)
        except AttributeError:
            # Fallback: unit weights
            net_coef = np.ones(len(deviation))

    raw_contributions = deviation * net_coef  # shape (24,)

    results = []
    for i, feat in enumerate(TRANSFORMED_ORDER):
        contrib = float(raw_contributions[i])
        domain_label, color = DOMAIN_INFO.get(feat, ("Other", "#4A5568"))
        display = FEATURE_DISPLAY.get(feat, feat.replace("_", " ").title())

        if contrib > 0.01:
            annotation = (
                f"Your {display.lower()} is higher than the average "
                f"low-stress student — this is contributing to your stress risk."
            )
        elif contrib < -0.01:
            annotation = (
                f"Your {display.lower()} is better than the low-stress average "
                f"— this is working in your favour."
            )
        else:
            annotation = (
                f"Your {display.lower()} is close to the low-stress baseline — "
                f"minimal impact on your score."
            )

        results.append({
            "feature":     feat,
            "label":       display,
            "contribution": contrib,
            "domain":      domain_label,
            "color":       color,
            "annotation":  annotation,
        })

    results.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return results[:top_n]

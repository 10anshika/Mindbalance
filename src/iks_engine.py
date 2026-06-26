"""
MindBalance v2 | iks_engine.py
Scored IKS Recommendation Engine — not just if-else rules.

Computes a weighted IKS Wellness Score (0–100) per student using
validated IKS variables, then prescribes a tiered intervention plan
with intervention intensity proportional to the score.

Run: python src/iks_engine.py
     python src/iks_engine.py --demo
     python src/iks_engine.py --interactive
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pickle, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pipeline import load_data
from config import DATA_PATH, MODEL_DIR, OUTPUT_DIR, LABEL_NAMES
from recommender import print_lifestyle_report

NAVY  = "#1B3A6B"; GOLD  = "#C8973A"; SLATE = "#4A5568"
LOW_C = "#2B6CB0"; MED_C = "#B7791F"; HIGH_C = "#C53030"; SAGE  = "#2F855A"

# ─────────────────────────────────────────────────────────────────────────────
# IKS WELLNESS SCORE FORMULA
# Scored 0–100. Higher score = better IKS engagement = protective against stress.
# Weights are derived from literature on Yoga/Pranayama efficacy.
#
#  Component                     Weight  Max raw   Rationale
#  Yoga frequency (days/week)     0.35    7        Core IKS practice marker
#  Yoga duration (mins/session)   0.20    60       Dose-response relationship
#  Mindfulness (mins/day)         0.25    60       Widely validated stress reducer
#  Pranayama (binary)             0.20    1        Autonomic regulation
# ─────────────────────────────────────────────────────────────────────────────

IKS_WEIGHTS = {
    "yoga_days_per_week":   (0.35, 7),
    "yoga_duration_mins":   (0.20, 60),
    "mindfulness_mins_day": (0.25, 60),
    "pranayama_practice":   (0.20, 1),
}

def compute_iks_score(row: dict) -> float:
    """Compute IKS Wellness Score 0–100 for a student row."""
    score = 0.0
    for feat, (weight, max_val) in IKS_WEIGHTS.items():
        val    = min(float(row.get(feat, 0)), max_val)
        score += weight * (val / max_val) * 100
    return round(score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# INTERVENTION LIBRARY
# Each tier has a base prescription + intensity multiplier from IKS score.
# ─────────────────────────────────────────────────────────────────────────────

INTERVENTIONS = {
    0: {  # LOW stress
        "tier":  "Low Stress",
        "color": LOW_C,
        "yoga": [
            ("Surya Namaskar (Sun Salutation)", "12 rounds/day — maintains energy and physical balance"),
            ("Tadasana (Mountain Pose)",        "5 min — grounding, posture alignment"),
            ("Vrikshasana (Tree Pose)",         "30 sec each side — builds concentration"),
        ],
        "pranayama": [
            ("Anulom Vilom", "10 min — balances both hemispheres of the nervous system"),
            ("Kapalbhati",   "5 min — clears mental fatigue and improves oxygenation"),
        ],
        "mindfulness": "10 min guided body scan or breath awareness each morning.",
        "ayurveda":    "Vata-pacifying diet: warm cooked meals, warm water, consistent wake time.",
        "frequency":   "3–4 sessions per week to sustain protective effect.",
        "note":        "You are managing stress well. The goal here is maintenance, not intervention."
    },
    1: {  # MODERATE stress
        "tier":  "Moderate Stress",
        "color": MED_C,
        "yoga": [
            ("Balasana (Child's Pose)",         "3–5 min — activates parasympathetic response"),
            ("Paschimottanasana (Seated Fold)", "5 min — calms the nervous system"),
            ("Viparita Karani (Legs Up Wall)",  "10 min before bed — reduces cortisol"),
            ("Marjaryasana-Bitilasana",         "Cat-Cow flow — releases spinal tension"),
        ],
        "pranayama": [
            ("Nadi Shodhana (Alternate Nostril)", "15 min — clinically shown to reduce anxiety scores"),
            ("Bhramari (Humming Bee Breath)",     "10 min — directly targets cognitive fatigue"),
        ],
        "mindfulness": "15 min body scan + 5 min journalling each evening.",
        "ayurveda":    "Reduce caffeine. Consider Ashwagandha (consult practitioner). Avoid cold/raw foods.",
        "frequency":   "5–6 sessions per week. Consistency is the intervention at this tier.",
        "note":        "Structured, consistent practice over 3–4 weeks typically produces measurable reduction."
    },
    2: {  # HIGH stress
        "tier":  "High Stress",
        "color": HIGH_C,
        "yoga": [
            ("Yoga Nidra",                "40 min — most evidence-based IKS tool for severe stress"),
            ("Shavasana (Corpse Pose)",   "20 min guided — full nervous system reset"),
            ("Supta Baddha Konasana",     "10 min — deep restorative, hip and chest opener"),
            ("Viparita Karani",           "15 min — proven cortisol reduction in 4-week trials"),
        ],
        "pranayama": [
            ("Bhramari Pranayama",  "15–20 min — stimulates vagus nerve, reduces cortisol"),
            ("4-7-8 Breathing",     "5 cycles before sleep — clinical sleep inducer"),
            ("Sheetali / Sheetkari","5 min — cooling breath, counteracts physiological arousal"),
        ],
        "mindfulness": "20 min Yoga Nidra or NSDR (Non-Sleep Deep Rest) daily. Non-negotiable at this tier.",
        "ayurveda":    "Eliminate stimulants. Add Brahmi + Shankhpushpi (consult practitioner). Eat by 7pm.",
        "frequency":   "Daily practice. Missing even one day has measurable impact at high stress levels.",
        "note":        "Seek counsellor support alongside IKS practice. This tier warrants professional involvement."
    }
}


def intensity_modifier(iks_score: float, tier: int) -> str:
    """
    Returns a personalised intensity note based on how much IKS practice
    the student already does vs what is needed for their tier.
    """
    needed = {0: 60, 1: 40, 2: 20}  # minimum IKS score to be adequately protected per tier
    gap    = needed[tier] - iks_score
    if gap <= 0:
        return f"Your IKS Wellness Score is {iks_score}/100 — already meeting the threshold for this tier. Focus on consistency."
    elif gap < 20:
        return f"Your IKS Wellness Score is {iks_score}/100 — close to the protective threshold. A small increase in Yoga frequency will bridge the gap."
    else:
        return f"Your IKS Wellness Score is {iks_score}/100 — significantly below the recommended level for your stress tier. Begin with one daily practice and scale up."


def print_recommendation(tier: int, iks_score: float, proba: np.ndarray):
    rec = INTERVENTIONS[tier]
    w   = 65
    print("\n" + "═"*w)
    print(f"  MINDBALANCE RESULT: {rec['tier']}")
    print(f"  IKS Wellness Score: {iks_score}/100")
    print("═"*w)
    print(f"\n  Confidence — Low: {proba[0]:.0%}  Moderate: {proba[1]:.0%}  High: {proba[2]:.0%}")
    print(f"\n  {rec['note']}")
    print(f"\n  {intensity_modifier(iks_score, tier)}")

    print("\n  ── YOGA PRESCRIPTION ─────────────────────────────────")
    for i, (asana, desc) in enumerate(rec["yoga"], 1):
        print(f"  {i}. {asana}")
        print(f"     {desc}")

    print("\n  ── PRANAYAMA (BREATHWORK) ────────────────────────────")
    for i, (name, desc) in enumerate(rec["pranayama"], 1):
        print(f"  {i}. {name}")
        print(f"     {desc}")

    print(f"\n  ── MINDFULNESS PRESCRIPTION ──────────────────────────")
    print(f"  {rec['mindfulness']}")
    print(f"\n  ── AYURVEDA LIFESTYLE ────────────────────────────────")
    print(f"  {rec['ayurveda']}")
    print(f"\n  ── PRACTICE FREQUENCY ────────────────────────────────")
    print(f"  {rec['frequency']}")
    print("\n" + "═"*w + "\n")


def save_report_chart(user: dict, tier: int, iks_score: float, proba: np.ndarray):
    """Generate a 3-panel visual report for a student."""
    rec   = INTERVENTIONS[tier]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Stress tier probability
    ax = axes[0]
    colors = [LOW_C, MED_C, HIGH_C]
    bars = ax.bar(LABEL_NAMES, [p*100 for p in proba], color=colors,
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, proba):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{val:.0%}", ha="center", fontsize=12, fontweight="bold", color=NAVY)
    bars[tier].set_edgecolor(NAVY); bars[tier].set_linewidth(3)
    ax.set_ylim(0, 115); ax.set_title("Stress Tier Probability", fontsize=12, fontweight="bold", color=NAVY)
    ax.set_ylabel("Confidence (%)"); ax.spines[["top","right"]].set_visible(False); ax.set_facecolor("#F7F8FA")

    # Panel 2: IKS Wellness Score gauge
    ax = axes[1]
    score_color = HIGH_C if iks_score < 25 else (MED_C if iks_score < 55 else SAGE)
    ax.barh(["IKS Score"], [iks_score], color=score_color, edgecolor="white", height=0.4)
    ax.barh(["IKS Score"], [100-iks_score], left=[iks_score], color="#E2E8F0", edgecolor="white", height=0.4)
    ax.text(iks_score/2, 0, f"{iks_score:.0f}", ha="center", va="center",
            fontsize=20, fontweight="bold", color="white")
    ax.set_xlim(0, 100); ax.set_title(f"IKS Wellness Score\n(out of 100)",
                                       fontsize=12, fontweight="bold", color=NAVY)
    ax.set_xlabel("Score"); ax.spines[["top","right"]].set_visible(False); ax.set_facecolor("#F7F8FA")

    # Panel 3: IKS component breakdown
    ax = axes[2]
    comp_labels = ["Yoga Frequency\n(days/wk)", "Yoga Duration\n(mins)", "Mindfulness\n(mins/day)", "Pranayama\n(binary)"]
    comp_keys   = ["yoga_days_per_week", "yoga_duration_mins", "mindfulness_mins_day", "pranayama_practice"]
    comp_maxes  = [7, 60, 60, 1]
    comp_colors = [GOLD, GOLD, SAGE, SAGE]
    pcts = [min(user.get(k,0)/m*100, 100) for k,m in zip(comp_keys, comp_maxes)]
    ax.barh(comp_labels, pcts, color=comp_colors, edgecolor="white", height=0.5)
    ax.barh(comp_labels, [100-p for p in pcts], left=pcts, color="#E2E8F0", edgecolor="white", height=0.5)
    for i, (p, v) in enumerate(zip(pcts, [user.get(k,0) for k in comp_keys])):
        ax.text(p/2, i, f"{v}", ha="center", va="center", fontsize=10, fontweight="bold", color="white" if p>15 else NAVY)
    ax.set_xlim(0, 110); ax.set_title("IKS Component Breakdown", fontsize=12, fontweight="bold", color=NAVY)
    ax.set_xlabel("% of recommended dose"); ax.spines[["top","right"]].set_visible(False); ax.set_facecolor("#F7F8FA")

    fig.suptitle(f"MindBalance Report  ·  {rec['tier']}  ·  IKS Score: {iks_score}/100",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_student_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 10_student_report.png saved.")


# ── Population-level IKS analysis ────────────────────────────────────────────
def run_population_analysis():
    """Analyse IKS scores across the full dataset and plot insights."""
    X, y = load_data(DATA_PATH)
    df   = X.copy(); df["stress_level"] = y

    df["iks_score"] = df.apply(compute_iks_score, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: IKS score distribution by stress tier
    ax = axes[0]
    for tier, color, name in zip([0,1,2], [LOW_C, MED_C, HIGH_C], LABEL_NAMES):
        vals = df[df["stress_level"]==tier]["iks_score"]
        ax.hist(vals, bins=20, alpha=0.65, color=color, label=f"{name} (n={len(vals)})", edgecolor="white")
    ax.set_xlabel("IKS Wellness Score (0–100)", fontsize=11)
    ax.set_ylabel("Number of Students", fontsize=11)
    ax.set_title("IKS Score Distribution by Stress Tier",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False); ax.set_facecolor("#F7F8FA")

    # Right: mean IKS score per tier
    ax = axes[1]
    means = df.groupby("stress_level")["iks_score"].mean()
    stds  = df.groupby("stress_level")["iks_score"].std()
    bars  = ax.bar(LABEL_NAMES, means.values, color=[LOW_C, MED_C, HIGH_C],
                   edgecolor="white", width=0.5,
                   yerr=stds.values, capsize=6, error_kw={"color": NAVY, "linewidth": 1.5})
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2,
                f"{val:.1f}", ha="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.set_title("Mean IKS Wellness Score per Stress Tier\n(higher = more protected)",
                 fontsize=12, fontweight="bold", color=NAVY)
    ax.set_ylabel("Mean IKS Score ± SD"); ax.set_ylim(0, 100)
    ax.spines[["top","right"]].set_visible(False); ax.set_facecolor("#F7F8FA")

    plt.suptitle("Population-Level IKS Analysis — MindBalance v2",
                 fontsize=13, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_iks_population_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 11_iks_population_analysis.png")

    # Print summary
    print("\n  IKS Score by Stress Tier:")
    print(df.groupby("stress_level")["iks_score"].agg(["mean","std","min","max"]).round(2).to_string())


# ── Entry points ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open(f"{MODEL_DIR}/best_model.pkl", "rb") as f:
        pipe = pickle.load(f)

    from config import ALL_FEATURES
    run_population_analysis()

    if "--interactive" in sys.argv:
        print("\n  Enter your details (all values are integers):\n")
        user = {}
        prompts = [
            ("anxiety_level",          "Anxiety level (0–20)",       0, 20),
            ("self_esteem",            "Self-esteem score (0–30)",    0, 30),
            ("mental_health_history",  "Mental health history (0/1)", 0, 1),
            ("depression",             "Depression score (0–27)",     0, 27),
            ("headache",               "Headache frequency (0–5)",    0, 5),
            ("blood_pressure",         "Blood pressure level (1–3)",  1, 3),
            ("sleep_quality",          "Sleep quality (0–5)",         0, 5),
            ("breathing_problem",      "Breathing problems (0–5)",    0, 5),
            ("noise_level",            "Noise level (0–5)",           0, 5),
            ("living_conditions",      "Living conditions (0–5)",     0, 5),
            ("safety",                 "Safety (0–5)",                0, 5),
            ("basic_needs",            "Basic needs met (0–5)",       0, 5),
            ("academic_performance",   "Academic performance (0–5)",  0, 5),
            ("study_load",             "Study load (0–5)",            0, 5),
            ("teacher_student_rel",    "Teacher-student rel (0–5)",   0, 5),
            ("future_career_concerns", "Career concerns (0–5)",       0, 5),
            ("social_support",         "Social support (0–3)",        0, 3),
            ("peer_pressure",          "Peer pressure (0–5)",         0, 5),
            ("extracurricular_act",    "Extracurricular (0–5)",       0, 5),
            ("bullying",               "Bullying (0–5)",              0, 5),
            ("yoga_days_per_week",     "Yoga days/week (0–7)",        0, 7),
            ("yoga_duration_mins",     "Yoga session mins (0–60)",    0, 60),
            ("mindfulness_mins_day",   "Mindfulness mins/day (0–60)", 0, 60),
            ("pranayama_practice",     "Pranayama practice (0/1)",    0, 1),
        ]
        for key, prompt, lo, hi in prompts:
            while True:
                try:
                    val = int(input(f"  {prompt}: "))
                    if lo <= val <= hi:
                        user[key] = val; break
                    print(f"    ⚠ Enter {lo}–{hi}")
                except ValueError:
                    print("    ⚠ Integer required")

        X_in  = pd.DataFrame([user])[ALL_FEATURES]
        tier  = int(pipe.predict(X_in)[0])
        proba = pipe.predict_proba(X_in)[0]
        score = compute_iks_score(user)
        print_recommendation(tier, score, proba)
        save_report_chart(user, tier, score, proba)
        print_lifestyle_report(user, tier, score)

    elif "--demo" in sys.argv:
        demos = [
            {"name": "High-Stress Student",
             "data": dict(anxiety_level=18, self_esteem=8, mental_health_history=1, depression=22,
                          headache=4, blood_pressure=3, sleep_quality=1, breathing_problem=4,
                          noise_level=4, living_conditions=2, safety=2, basic_needs=2,
                          academic_performance=2, study_load=5, teacher_student_rel=1,
                          future_career_concerns=5, social_support=0, peer_pressure=5,
                          extracurricular_act=1, bullying=4,
                          yoga_days_per_week=0, yoga_duration_mins=0,
                          mindfulness_mins_day=0, pranayama_practice=0)},
            {"name": "Low-Stress Student",
             "data": dict(anxiety_level=3, self_esteem=28, mental_health_history=0, depression=3,
                          headache=1, blood_pressure=1, sleep_quality=5, breathing_problem=0,
                          noise_level=1, living_conditions=5, safety=5, basic_needs=5,
                          academic_performance=5, study_load=2, teacher_student_rel=5,
                          future_career_concerns=2, social_support=3, peer_pressure=1,
                          extracurricular_act=4, bullying=0,
                          yoga_days_per_week=6, yoga_duration_mins=45,
                          mindfulness_mins_day=30, pranayama_practice=1)},
        ]
        for d in demos:
            X_in  = pd.DataFrame([d["data"]])[ALL_FEATURES]
            tier  = int(pipe.predict(X_in)[0])
            proba = pipe.predict_proba(X_in)[0]
            score = compute_iks_score(d["data"])
            print(f"\n{'─'*65}\n  DEMO: {d['name']}\n{'─'*65}")
            print_recommendation(tier, score, proba)
            save_report_chart(d["data"], tier, score, proba)
            print_lifestyle_report(d["data"], tier, score)
    else:
        print("\n  Usage:")
        print("    python src/iks_engine.py              # population analysis only")
        print("    python src/iks_engine.py --demo       # show 2 demo profiles")
        print("    python src/iks_engine.py --interactive  # enter your own data")

"""
MindBalance v2 | recommender.py
Full Lifestyle Report Generator.

Analyses all six domain scores (sleep, social, academic, physical,
environmental, psychological) from a student profile and emits
concrete, tiered lifestyle recommendations that complement the
IKS / Yoga prescription produced by iks_engine.py.

Run standalone:
    python src/recommender.py --batch          # process full dataset → CSV
    python src/recommender.py --demo           # show 2 sample reports
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from config import DATA_PATH, OUTPUT_DIR, LABEL_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN SCORING  (each domain → 0-100; lower raw score = more at-risk)
# ─────────────────────────────────────────────────────────────────────────────

def _pct(val, max_val, invert=False):
    """Normalise val to 0-100; invert for features where higher = worse."""
    p = min(float(val), max_val) / max_val * 100
    return round(100 - p if invert else p, 1)


def domain_scores(row: dict) -> dict:
    """Return a dict of six domain health percentages (0=very poor, 100=excellent)."""
    return {
        "sleep":         _pct(row.get("sleep_quality", 0),        5),
        "social":        _pct(row.get("social_support", 0),       3),
        "academic":      _pct(row.get("academic_performance", 0), 5),
        "physical":      _pct(row.get("blood_pressure", 1) - 1,   2, invert=True),  # 1→best, 3→worst
        "environmental": _pct(row.get("living_conditions", 0),    5),
        "psychological": _pct(row.get("self_esteem", 0),          30),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION RULES
# Each function receives the raw row dict and returns a list of advice strings.
# ─────────────────────────────────────────────────────────────────────────────

def _sleep_advice(row: dict) -> list[str]:
    sq = row.get("sleep_quality", 3)
    if sq >= 4:
        return ["Sleep quality is good. Maintain a consistent 10 pm–6 am schedule to consolidate gains."]
    elif sq == 3:
        return [
            "Sleep quality is moderate. Avoid screens 60 min before bed.",
            "Keep room temperature between 18–20 °C for deeper sleep cycles.",
            "Consider a 10-min Yoga Nidra recording as a pre-sleep ritual.",
        ]
    else:
        return [
            "Sleep quality is low — this is the single highest-leverage intervention point.",
            "Establish a strict wake time (even weekends) to anchor your circadian rhythm.",
            "Eliminate caffeine after 2 pm; alcohol and late meals disrupt REM sleep.",
            "Practice 4-7-8 breathing (4 s inhale, 7 hold, 8 exhale) for 5 cycles at lights-out.",
            "Track sleep for one week using a free app (e.g. Sleep Cycle) to identify patterns.",
        ]


def _social_advice(row: dict) -> list[str]:
    ss    = row.get("social_support", 1)
    pp    = row.get("peer_pressure", 0)
    bully = row.get("bullying", 0)
    tips  = []
    if ss <= 1:
        tips += [
            "Social support is critically low. Reach out to one trusted person this week.",
            "Consider joining a study group, club, or campus volunteer activity.",
            "Campus counselling services offer peer-support circles — highly effective for isolation.",
        ]
    elif ss == 2:
        tips.append("Social support is moderate. Deepen 1–2 relationships rather than widening the network.")
    else:
        tips.append("Strong social support — a key protective factor. Nurture these connections actively.")

    if pp >= 4:
        tips.append("High peer pressure detected. Practice assertiveness: 'I'm not available for that right now' is a complete sentence.")
    if bully >= 3:
        tips.append("Bullying indicators are elevated. Document incidents and report through the institution's welfare channel.")
    return tips


def _academic_advice(row: dict) -> list[str]:
    ap  = row.get("academic_performance", 3)
    sl  = row.get("study_load", 3)
    fcc = row.get("future_career_concerns", 3)
    tsr = row.get("teacher_student_rel", 3)
    tips = []

    if sl >= 4:
        tips += [
            "Study load is very high. Use the Pomodoro technique: 25 min deep work, 5 min break.",
            "Batch similar tasks (readings together, assignments together) to reduce context-switching.",
        ]
    if ap <= 2:
        tips += [
            "Academic performance is low. Identify the one subject creating the most drag and address it first.",
            "Visit faculty office hours — most professors will provide targeted guidance when approached directly.",
        ]
    if fcc >= 4:
        tips.append("Career anxiety is high. Schedule one concrete career action per week (LinkedIn profile, one application, one informational interview).")
    if tsr <= 2:
        tips.append("Teacher–student relationship is strained. A short, polite email expressing engagement often resets the dynamic.")
    if not tips:
        tips.append("Academic profile is balanced. Continue using active recall and spaced repetition to consolidate learning.")
    return tips


def _physical_advice(row: dict) -> list[str]:
    bp  = row.get("blood_pressure", 1)
    hd  = row.get("headache", 0)
    brp = row.get("breathing_problem", 0)
    tips = []
    if bp >= 3:
        tips += [
            "Blood pressure is elevated. Reduce sodium and processed food intake immediately.",
            "30 min brisk walking daily reduces systolic BP by 5–8 mmHg on average.",
            "Consult a physician if elevation persists beyond two weeks.",
        ]
    elif bp == 2:
        tips.append("Blood pressure is borderline. Monitor weekly; reduce caffeine and maintain hydration (2–3 L/day).")
    else:
        tips.append("Blood pressure is healthy. Sustain aerobic activity 3×/week to maintain this.")

    if hd >= 3:
        tips += [
            "Frequent headaches detected. Ensure 8-glass daily water intake — dehydration is a primary trigger.",
            "Tension headaches: apply cold/warm compress to neck and practise neck rolls (Greeva Sanchalana).",
        ]
    if brp >= 3:
        tips.append("Breathing problems are significant. Kapalbhati (slow variant, 1:2 ratio) strengthens respiratory muscles.")
    return tips


def _environmental_advice(row: dict) -> list[str]:
    nl  = row.get("noise_level", 2)
    lc  = row.get("living_conditions", 3)
    sf  = row.get("safety", 3)
    bn  = row.get("basic_needs", 3)
    tips = []
    if nl >= 4:
        tips.append("Noise level is high. Use foam earplugs or over-ear headphones with brown noise during study hours.")
    if lc <= 2:
        tips.append("Living conditions need attention. Identify one actionable improvement (ventilation, lighting, cleanliness) per week.")
    if sf <= 2:
        tips.append("Safety concerns are present. Contact campus welfare or student union for housing/safety support.")
    if bn <= 2:
        tips.append("Basic needs are not fully met. Explore campus food banks, emergency bursaries, or student hardship funds.")
    if not tips:
        tips.append("Environmental conditions are adequate. Keep your study space tidy — physical order reduces cognitive load.")
    return tips


def _psychological_advice(row: dict) -> list[str]:
    ax  = row.get("anxiety_level", 5)
    se  = row.get("self_esteem", 15)
    dep = row.get("depression", 5)
    mhh = row.get("mental_health_history", 0)
    tips = []

    if dep >= 20:
        tips += [
            "Depression indicators are in the severe range. Please speak to a mental health professional this week.",
            "The PHQ-9 online self-test can help you articulate symptoms to a clinician.",
        ]
    elif dep >= 10:
        tips.append("Moderate depression indicators present. Consider scheduling a GP visit and explore CBT-based self-help resources.")

    if ax >= 15:
        tips += [
            "Anxiety is high. Ground yourself with 5-4-3-2-1 (5 things you see, 4 touch, 3 hear, 2 smell, 1 taste) during acute episodes.",
            "Limit news and social media to one 15-min slot per day.",
        ]
    elif ax >= 8:
        tips.append("Anxiety is moderate. Journalling for 10 min each evening externalises worry and reduces overnight rumination.")

    if se <= 10:
        tips += [
            "Self-esteem is low. Start a weekly 'three wins' journal — write three things that went well each Sunday.",
            "Avoid comparing outputs; compare your own growth trajectory instead.",
        ]
    if mhh == 1:
        tips.append("Mental health history noted. Proactive engagement with campus counselling (even when stable) is strongly recommended.")

    if not tips:
        tips.append("Psychological indicators are within healthy range. Continue self-awareness practices to maintain this.")
    return tips


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REPORT FUNCTION  (callable from iks_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_LABELS = {
    "sleep":         "Sleep",
    "social":        "Social Support",
    "academic":      "Academic",
    "physical":      "Physical Health",
    "environmental": "Environment",
    "psychological": "Psychological",
}

STATUS_ICON = {
    "excellent": "[++]",
    "good":      "[ +]",
    "moderate":  "[~~]",
    "poor":      "[ -]",
    "critical":  "[!!]",
}

def _status(score: float) -> str:
    if score >= 80:   return "excellent"
    elif score >= 60: return "good"
    elif score >= 40: return "moderate"
    elif score >= 20: return "poor"
    else:             return "critical"


def print_lifestyle_report(row: dict, tier: int, iks_score: float):
    """
    Print the full lifestyle report for a student.
    Called at the end of every iks_engine prediction.

    Parameters
    ----------
    row       : dict  — raw feature values for the student
    tier      : int   — predicted stress tier (0/1/2)
    iks_score : float — IKS Wellness Score (0–100)
    """
    w    = 65
    ds   = domain_scores(row)
    tier_name = LABEL_NAMES[tier]

    print("\n" + "━" * w)
    print(f"  LIFESTYLE REPORT  ·  {tier_name} Stress  ·  IKS Score: {iks_score}/100")
    print("━" * w)

    # Domain health summary bar
    print("\n  DOMAIN HEALTH OVERVIEW")
    print(f"  {'Domain':<22} {'Score':>6}   Status")
    print("  " + "─" * (w - 2))
    for key, label in DOMAIN_LABELS.items():
        sc  = ds[key]
        st  = _status(sc)
        bar_fill = int(sc / 5)  # 20 chars wide
        bar = "█" * bar_fill + "░" * (20 - bar_fill)
        icon = STATUS_ICON[st]
        print(f"  {label:<22} {sc:>5.0f}%  {icon} {bar}")

    weakest = min(ds, key=ds.get)
    print(f"\n  Priority domain: {DOMAIN_LABELS[weakest]} ({ds[weakest]:.0f}%) — recommendations below target this first.")

    # Per-domain advice sections
    sections = [
        ("SLEEP HYGIENE",        _sleep_advice(row)),
        ("SOCIAL & RELATIONSHIPS",_social_advice(row)),
        ("ACADEMIC STRATEGIES",  _academic_advice(row)),
        ("PHYSICAL HEALTH",      _physical_advice(row)),
        ("ENVIRONMENT",          _environmental_advice(row)),
        ("PSYCHOLOGICAL HEALTH", _psychological_advice(row)),
    ]

    for title, tips in sections:
        print(f"\n  ── {title} {'─' * max(0, w - len(title) - 7)}")
        for tip in tips:
            # Word-wrap at ~60 chars
            words = tip.split()
            line, lines = [], []
            for w_ in words:
                if sum(len(x)+1 for x in line) + len(w_) > 58:
                    lines.append(" ".join(line))
                    line = [w_]
                else:
                    line.append(w_)
            if line:
                lines.append(" ".join(line))
            print(f"  • {lines[0]}")
            for cont in lines[1:]:
                print(f"    {cont}")

    print("\n" + "━" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH MODE  (python src/recommender.py --batch)
# ─────────────────────────────────────────────────────────────────────────────

def run_batch():
    """Compute domain scores for every student in the dataset, save CSV."""
    import pickle
    from config import MODEL_DIR, ALL_FEATURES

    df = pd.read_csv(DATA_PATH)

    with open(f"{MODEL_DIR}/best_model.pkl", "rb") as f:
        pipe = pickle.load(f)

    X   = df[ALL_FEATURES]
    df["predicted_tier"] = pipe.predict(X)

    score_rows = df.apply(lambda r: domain_scores(r.to_dict()), axis=1)
    scores_df  = pd.DataFrame(list(score_rows))
    scores_df.columns = [f"domain_{c}" for c in scores_df.columns]

    from iks_engine import compute_iks_score
    df["iks_score"] = df.apply(lambda r: compute_iks_score(r.to_dict()), axis=1)

    out = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
    out_path = f"{OUTPUT_DIR}/lifestyle_report_batch.csv"
    out.to_csv(out_path, index=False)
    print(f"  ✓ Batch lifestyle report saved → {out_path}  ({len(out)} students)")

    # Summary statistics
    print("\n  Domain Health Summary (mean % across all students):")
    domain_cols = [c for c in scores_df.columns]
    summary = scores_df.mean().rename(lambda c: c.replace("domain_", "")).to_frame("mean_%")
    summary["status"] = summary["mean_%"].apply(_status)
    print(summary.to_string())

    print("\n  Mean IKS Score by Predicted Tier:")
    print(out.groupby("predicted_tier")["iks_score"].mean().rename(
        index={0: "Low", 1: "Moderate", 2: "High"}).to_frame("mean_iks").to_string())


# ─────────────────────────────────────────────────────────────────────────────
# DEMO  (python src/recommender.py --demo)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_PROFILES = [
    {
        "name": "High-Stress Student",
        "tier": 2,
        "data": dict(
            anxiety_level=18, self_esteem=8, mental_health_history=1, depression=22,
            headache=4, blood_pressure=3, sleep_quality=1, breathing_problem=4,
            noise_level=4, living_conditions=2, safety=2, basic_needs=2,
            academic_performance=2, study_load=5, teacher_student_rel=1,
            future_career_concerns=5, social_support=0, peer_pressure=5,
            extracurricular_act=1, bullying=4,
            yoga_days_per_week=0, yoga_duration_mins=0,
            mindfulness_mins_day=0, pranayama_practice=0,
        ),
    },
    {
        "name": "Low-Stress Student",
        "tier": 0,
        "data": dict(
            anxiety_level=3, self_esteem=28, mental_health_history=0, depression=3,
            headache=1, blood_pressure=1, sleep_quality=5, breathing_problem=0,
            noise_level=1, living_conditions=5, safety=5, basic_needs=5,
            academic_performance=5, study_load=2, teacher_student_rel=5,
            future_career_concerns=2, social_support=3, peer_pressure=1,
            extracurricular_act=4, bullying=0,
            yoga_days_per_week=6, yoga_duration_mins=45,
            mindfulness_mins_day=30, pranayama_practice=1,
        ),
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindBalance Lifestyle Recommender")
    parser.add_argument("--batch", action="store_true", help="Process full dataset and save CSV")
    parser.add_argument("--demo",  action="store_true", help="Show lifestyle reports for demo profiles")
    args = parser.parse_args()

    if args.batch:
        print("\n  Running batch lifestyle analysis …")
        run_batch()
    elif args.demo:
        from iks_engine import compute_iks_score
        for p in DEMO_PROFILES:
            print(f"\n{'─'*65}\n  DEMO: {p['name']}\n{'─'*65}")
            iks = compute_iks_score(p["data"])
            print_lifestyle_report(p["data"], p["tier"], iks)
    else:
        print("Usage:")
        print("  python src/recommender.py --batch   # full dataset analysis")
        print("  python src/recommender.py --demo    # show sample reports")

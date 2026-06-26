"""
MindBalance v2 | app.py
Streamlit web application — run with:  streamlit run app.py
"""

import sys, os, pickle, json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import (ALL_FEATURES, LABEL_NAMES, DATA_PATH,
                    MODEL_DIR, OUTPUT_DIR, WIDE_SCALE, NARROW_SCALE,
                    PSYCHOLOGICAL, PHYSIOLOGICAL, IKS)
from iks_engine import compute_iks_score, INTERVENTIONS
from recommender import (
    domain_scores, DOMAIN_LABELS,
    _sleep_advice, _social_advice, _academic_advice,
    _physical_advice, _environmental_advice, _psychological_advice,
)
from fast_explainer import (
    compute_healthy_baseline, get_feature_contributions, FEATURE_DISPLAY,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindBalance — Student Stress Assessment",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TIER_COLORS  = {"Low": "#2B6CB0", "Moderate": "#B7791F", "High": "#C53030"}
TIER_BG      = {"Low": "#EBF4FF", "Moderate": "#FFFAF0", "High": "#FFF5F5"}
GOLD         = "#C8973A"
NAVY         = "#1B3A6B"
PAGES        = [
    "🏠  Take the Assessment",
    "📊  My Stress Report",
    "📋  My Action Plan",
    "ℹ️  About MindBalance",
]

FOUR_WEEK_PLANS = {
    0: [
        {"week": "Week 1", "title": "Build Your Foundation",
         "practice": "Surya Namaskar (12 rounds) + Anulom Vilom (10 min), 3×/week",
         "duration": "25 min/session",
         "expect": "Improved energy within 3–4 days"},
        {"week": "Week 2", "title": "Add Variety",
         "practice": "Add Vrikshasana & Tadasana. Introduce 10-min body scan at bedtime.",
         "duration": "30 min/session",
         "expect": "Shorter time to fall asleep"},
        {"week": "Week 3", "title": "Deepen Mindfulness",
         "practice": "Full sequence + 10-min journalling. Try Kapalbhati (5 min).",
         "duration": "35 min/session",
         "expect": "Clearer thinking, less mental clutter"},
        {"week": "Week 4", "title": "Sustain & Review",
         "practice": "Maintain 4× routine. Self-rate sleep, energy, and mood vs. Week 1.",
         "duration": "30–40 min/session",
         "expect": "Consolidated protective habit"},
    ],
    1: [
        {"week": "Week 1", "title": "Interrupt the Pattern",
         "practice": "Daily Balasana (5 min) + Nadi Shodhana (15 min). Non-negotiable.",
         "duration": "20 min/day",
         "expect": "Reduced cortisol spike within 5 days"},
        {"week": "Week 2", "title": "Build Consistency",
         "practice": "Add Viparita Karani (10 min before bed). Reduce caffeine to 1 cup/day.",
         "duration": "30 min/session",
         "expect": "Noticeably improved sleep quality"},
        {"week": "Week 3", "title": "Stabilise",
         "practice": "Full sequence + 15-min body scan + 5-min journalling each evening.",
         "duration": "40 min/session",
         "expect": "Measurable anxiety reduction (self-rate before/after)"},
        {"week": "Week 4", "title": "Consolidate Gains",
         "practice": "Review which practices helped most. Anchor the top 2 permanently.",
         "duration": "30 min/session",
         "expect": "Sustainable long-term routine in place"},
    ],
    2: [
        {"week": "Week 1", "title": "Emergency Protocol",
         "practice": "Yoga Nidra 40 min daily. 4-7-8 breathing at lights-out. No exceptions.",
         "duration": "45 min/day",
         "expect": "Sleep latency (time to fall asleep) decreases"},
        {"week": "Week 2", "title": "Build Your Foundation",
         "practice": "Add Bhramari (15 min) each morning. Use 5-4-3-2-1 grounding at anxious moments.",
         "duration": "60 min/day",
         "expect": "Acute anxiety episodes become shorter"},
        {"week": "Week 3", "title": "Expand the Practice",
         "practice": "Add Shavasana + Supta Baddha Konasana. Eliminate all caffeine and alcohol.",
         "duration": "60–75 min/day",
         "expect": "Baseline anxiety trending noticeably down"},
        {"week": "Week 4", "title": "Seek Support & Review",
         "practice": "Schedule a counsellor appointment. Bring your 4-week practice log.",
         "duration": "Ongoing",
         "expect": "Professional support pathway established"},
    ],
}

TRACKING_TABLE = [
    ["😴 Sleep hours per night",   "Phone notes / any habit app", "7–9 hours",        "< 6 hours consistently"],
    ["🧘 Yoga days per week",      "Mark on wall calendar",       "4 or more days",   "< 2 days"],
    ["🧠 Daily mindfulness (min)", "Timer app",                   "15 min or more",   "Skipping more than 2 days"],
    ["😰 Morning anxiety (0–10)",  "Rate in notes after waking",  "Trending below 5", "Staying above 7"],
    ["⚡ Evening energy (0–10)",   "Rate before bed",             "Above 6 most days","Below 4 most days"],
]

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Section headings */
.section-header {
    border-left: 4px solid #C8973A;
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.75rem 0;
    font-size: 1.1rem;
    font-weight: 700;
    color: #1B3A6B;
}

/* Stress tier banner */
.tier-banner {
    border-radius: 10px;
    padding: 1.4rem 2rem;
    margin-bottom: 1rem;
}
.tier-banner h1 { margin: 0; font-size: 2rem; }
.tier-banner p  { margin: 0.4rem 0 0 0; font-size: 1rem; opacity: 0.85; }

/* Metric card */
.metric-card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);
    border-left: 4px solid #C8973A;
    margin-bottom: 0.5rem;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #718096;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-card .value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #1A202C;
    line-height: 1.2;
}
.metric-card .sub { font-size: 0.85rem; color: #4A5568; margin-top: 0.2rem; }

/* Action plan card */
.action-card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.09);
    margin-bottom: 0.75rem;
    border-top: 3px solid #C8973A;
    min-height: 110px;
}
.action-card .icon { font-size: 1.6rem; }
.action-card .title { font-weight: 700; font-size: 0.95rem; color: #1B3A6B; margin: 0.3rem 0 0.2rem 0; }
.action-card .desc  { font-size: 0.84rem; color: #4A5568; line-height: 1.4; }

/* Week card */
.week-card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.09);
    margin-bottom: 0.75rem;
    border-left: 5px solid #1B3A6B;
}
.week-card .week-label { font-size: 0.72rem; font-weight: 700; color: #C8973A;
                          text-transform: uppercase; letter-spacing: 0.08em; }
.week-card .week-title { font-size: 1rem; font-weight: 700; color: #1B3A6B; margin: 0.15rem 0; }
.week-card .week-body  { font-size: 0.85rem; color: #4A5568; line-height: 1.45; }
.week-card .week-expect{ font-size: 0.8rem; color: #2F855A; font-weight: 600; margin-top: 0.4rem; }

/* Domain advice section */
.domain-section {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.domain-section h4 { margin: 0 0 0.5rem 0; color: #1B3A6B; font-size: 0.95rem; }
.domain-section p  { margin: 0.2rem 0; font-size: 0.85rem; color: #4A5568; line-height: 1.45; }
.domain-good { color: #2F855A !important; font-weight: 600; }

/* Warning box */
.warning-box {
    border: 2px solid #C53030;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    background: #FFF5F5;
    margin-top: 1rem;
}
.warning-box h4 { color: #C53030; margin: 0 0 0.5rem 0; }
.warning-box p  { font-size: 0.85rem; color: #4A5568; margin: 0.2rem 0; }

/* IKS formula visual */
.formula-row {
    display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.75rem 0;
}
.formula-chip {
    border-radius: 8px; padding: 0.6rem 0.9rem;
    font-size: 0.82rem; font-weight: 600;
    display: inline-flex; flex-direction: column; align-items: center;
    min-width: 130px; text-align: center;
}
.disclaimer-box {
    background: #EBF8FF;
    border-left: 3px solid #2B6CB0;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.8rem;
    color: #2C5282;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _retrain_and_save(path: str):
    """Retrain Logistic Regression with the locally installed sklearn and save."""
    import warnings
    warnings.filterwarnings("ignore")
    from pipeline import build_pipeline, load_data
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = load_data(DATA_PATH)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipe.fit(X_train, y_train)
    with open(path, "wb") as f:
        pickle.dump(pipe, f)
    return pipe


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = os.path.join(ROOT, "models", "best_model.pkl")

    pipe = None
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                pipe = pickle.load(f)
            # Validate with a dummy prediction — version mismatches often only
            # surface during inference (not during pickle.load), so this catches
            # errors like "LogisticRegression has no attribute 'multi_class'".
            _dummy = pd.DataFrame(
                [{f: (1 if f == "blood_pressure" else 0) for f in ALL_FEATURES}]
            )
            pipe.predict_proba(_dummy)
        except Exception:
            pipe = None  # fall through to retrain

    if pipe is None:
        st.toast("Rebuilding model for your sklearn version — one moment…", icon="⚙️")
        pipe = _retrain_and_save(path)

    return pipe


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = os.path.join(ROOT, "models", "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    path = os.path.join(ROOT, "outputs", "model_comparison.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_baseline(_df: pd.DataFrame) -> pd.Series:
    return compute_healthy_baseline(_df)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _metric_card(label: str, value: str, sub: str = "", accent: str = GOLD) -> str:
    style = f"border-left-color:{accent}"
    return (
        f'<div class="metric-card" style="{style}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'{"<div class=sub>" + sub + "</div>" if sub else ""}'
        f"</div>"
    )


def _action_card(icon: str, title: str, desc: str, accent: str = GOLD) -> str:
    return (
        f'<div class="action-card" style="border-top-color:{accent}">'
        f'<div class="icon">{icon}</div>'
        f'<div class="title">{title}</div>'
        f'<div class="desc">{desc}</div>'
        f"</div>"
    )


def _week_card(week: str, title: str, practice: str, duration: str, expect: str) -> str:
    return (
        f'<div class="week-card">'
        f'<div class="week-label">{week}</div>'
        f'<div class="week-title">{title}</div>'
        f'<div class="week-body"><strong>Practice:</strong> {practice}<br>'
        f'<strong>Time:</strong> {duration}</div>'
        f'<div class="week-expect">✓ What to expect: {expect}</div>'
        f"</div>"
    )


def _slider_row(label: str, key: str, lo: int, hi: int, default: int,
                emoji_lo: str, emoji_hi: str) -> int:
    st.markdown(f"**{label}**")
    c1, c2, c3 = st.columns([1, 10, 1])
    c1.markdown(f"<div style='text-align:center;font-size:1.1rem;padding-top:0.5rem'>{emoji_lo}</div>",
                unsafe_allow_html=True)
    with c2:
        val = st.slider(key, lo, hi, default, label_visibility="collapsed", key=f"sl_{key}")
    c3.markdown(f"<div style='text-align:center;font-size:1.1rem;padding-top:0.5rem'>{emoji_hi}</div>",
                unsafe_allow_html=True)
    return val


def _section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def _run_prediction(pipe, df: pd.DataFrame, user: dict) -> dict:
    X_in  = pd.DataFrame([user])[ALL_FEATURES]
    tier  = int(pipe.predict(X_in)[0])
    proba = pipe.predict_proba(X_in)[0]
    iks   = compute_iks_score(user)
    ds    = domain_scores(user)
    baseline = get_baseline(df)
    contribs = get_feature_contributions(pipe, user, baseline, top_n=5)
    # Stress risk 0–100 (weighted tier probability)
    risk_score = float(0 * proba[0] + 50 * proba[1] + 100 * proba[2])
    # Top risk factor
    pos_contribs = [c for c in contribs if c["contribution"] > 0]
    top_risk = pos_contribs[0]["label"] if pos_contribs else contribs[0]["label"]
    return {
        "tier": tier,
        "tier_name": LABEL_NAMES[tier],
        "proba": proba.tolist(),
        "iks_score": iks,
        "domain_scores": ds,
        "contributions": contribs,
        "risk_score": risk_score,
        "top_risk": top_risk,
        "user": user,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
def page_assessment(pipe, df: pd.DataFrame):
    st.markdown("## 🧘 MindBalance — Student Stress Assessment")
    st.markdown(
        "Answer honestly. Takes about **3 minutes**. "
        "All data stays on your device — nothing is stored or sent anywhere."
    )
    st.divider()

    # ── Section 1: Feelings ──────────────────────────────────────────────────
    with st.expander("😔  How are you feeling?", expanded=True):
        st.caption("Rate how you've been feeling over the past two weeks.")

        anxiety_level = _slider_row(
            "How anxious do you feel day-to-day?",
            "anxiety_level", 0, 20, 5, "😌 Calm", "😰 Very anxious")

        self_esteem = _slider_row(
            "How confident and positive do you feel about yourself?",
            "self_esteem", 0, 30, 15, "😔 Low", "😊 Very confident")

        depression = _slider_row(
            "How low, hopeless, or unmotivated have you been feeling?",
            "depression", 0, 27, 5, "😊 Rarely", "😞 Very often")

        st.markdown("**Have you experienced mental health difficulties in the past?**")
        mhh_val = st.selectbox("mental_health_history_sel",
                                ["No (0)", "Yes (1)"], index=0,
                                label_visibility="collapsed", key="sl_mental_health_history")
        mental_health_history = 0 if "No" in mhh_val else 1

    # ── Section 2: Sleep & Body ──────────────────────────────────────────────
    with st.expander("💤  Your sleep & body"):
        st.caption("Your physical health has a direct link to your stress level.")

        sleep_quality = _slider_row(
            "Rate your overall sleep quality",
            "sleep_quality", 0, 5, 3, "😴 Very poor", "✨ Excellent")

        headache = _slider_row(
            "How often do you get headaches?",
            "headache", 0, 5, 1, "Never", "Every day")

        st.markdown("**Blood pressure level**")
        bp_val = st.select_slider(
            "blood_pressure_sel",
            options=["1 — Normal", "2 — Borderline high", "3 — High"],
            value="1 — Normal", label_visibility="collapsed", key="sl_blood_pressure")
        blood_pressure = int(bp_val[0])

        breathing_problem = _slider_row(
            "Breathing difficulties (chest tightness, shortness of breath)",
            "breathing_problem", 0, 5, 0, "None", "Severe")

    # ── Section 3: Environment ───────────────────────────────────────────────
    with st.expander("🏠  Your environment"):
        st.caption("Where you live and study shapes how you feel.")

        noise_level = _slider_row(
            "How noisy is your study / living space?",
            "noise_level", 0, 5, 2, "🤫 Very quiet", "📢 Very noisy")

        living_conditions = _slider_row(
            "Rate your current living conditions overall",
            "living_conditions", 0, 5, 3, "😟 Poor", "🏠 Excellent")

        safety = _slider_row(
            "How safe do you feel where you live and study?",
            "safety", 0, 5, 3, "😨 Unsafe", "🛡️ Very safe")

        basic_needs = _slider_row(
            "Are your basic needs (food, housing, heating) met?",
            "basic_needs", 0, 5, 4, "😢 Barely", "✅ Fully")

    # ── Section 4: Academic life ─────────────────────────────────────────────
    with st.expander("📚  Academic life"):
        st.caption("Academic pressure is one of the largest drivers of student stress.")

        academic_performance = _slider_row(
            "How well are you performing academically right now?",
            "academic_performance", 0, 5, 3, "📉 Struggling", "📈 Excellent")

        study_load = _slider_row(
            "How heavy is your current study workload?",
            "study_load", 0, 5, 3, "📖 Light", "📚 Overwhelming")

        teacher_student_rel = _slider_row(
            "How would you rate your relationship with your teachers / lecturers?",
            "teacher_student_rel", 0, 5, 3, "😤 Strained", "🤝 Excellent")

        future_career_concerns = _slider_row(
            "How worried are you about your future career?",
            "future_career_concerns", 0, 5, 2, "😌 Not at all", "😰 Extremely")

        social_support = _slider_row(
            "How much social support do you have? (friends, family, mentors)",
            "social_support", 0, 3, 1, "😶 None", "🤗 Strong")

        peer_pressure = _slider_row(
            "How much peer pressure do you experience?",
            "peer_pressure", 0, 5, 2, "😌 None", "😰 Intense")

        extracurricular_act = _slider_row(
            "Level of extracurricular activity involvement",
            "extracurricular_act", 0, 5, 2, "None", "Very active")

        bullying = _slider_row(
            "Have you experienced bullying or harassment?",
            "bullying", 0, 5, 0, "Never", "Severely")

    # ── Section 5: Wellness practice ─────────────────────────────────────────
    with st.expander("🌿  Your wellness practice"):
        st.caption(
            "This is what makes MindBalance different — we measure your IKS "
            "(Indian Knowledge Systems) practice as a protective factor."
        )

        yoga_days_per_week = _slider_row(
            "How many days per week do you practise yoga?",
            "yoga_days_per_week", 0, 7, 0, "0 days", "Every day 🧘")

        yoga_duration_mins = _slider_row(
            "How long is a typical yoga session? (minutes)",
            "yoga_duration_mins", 0, 60, 0, "0 min", "60 min")

        mindfulness_mins_day = _slider_row(
            "How much time do you spend on mindfulness / meditation each day? (minutes)",
            "mindfulness_mins_day", 0, 60, 0, "0 min", "60 min 🧠")

        st.markdown("**Do you practise pranayama (breathing exercises)?**")
        pr_val = st.selectbox("pranayama_sel",
                               ["No (0)", "Yes (1)"], index=0,
                               label_visibility="collapsed", key="sl_pranayama_practice")
        pranayama_practice = 0 if "No" in pr_val else 1

    # ── Submit ────────────────────────────────────────────────────────────────
    st.divider()
    col_btn, col_note = st.columns([2, 5])
    with col_btn:
        submitted = st.button("🔍  Get My Report", type="primary", use_container_width=True)
    with col_note:
        st.caption(
            "We'll analyse your answers across 24 factors and build a personalised "
            "stress report with an IKS wellness plan — all in under a second."
        )

    if submitted:
        user = dict(
            anxiety_level=anxiety_level, self_esteem=self_esteem,
            mental_health_history=mental_health_history, depression=depression,
            headache=headache, blood_pressure=blood_pressure,
            sleep_quality=sleep_quality, breathing_problem=breathing_problem,
            noise_level=noise_level, living_conditions=living_conditions,
            safety=safety, basic_needs=basic_needs,
            academic_performance=academic_performance, study_load=study_load,
            teacher_student_rel=teacher_student_rel,
            future_career_concerns=future_career_concerns,
            social_support=social_support, peer_pressure=peer_pressure,
            extracurricular_act=extracurricular_act, bullying=bullying,
            yoga_days_per_week=yoga_days_per_week,
            yoga_duration_mins=yoga_duration_mins,
            mindfulness_mins_day=mindfulness_mins_day,
            pranayama_practice=pranayama_practice,
        )
        with st.spinner("Analysing your profile…"):
            results = _run_prediction(pipe, df, user)
        st.session_state["results"] = results
        st.session_state["current_page"] = PAGES[1]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — STRESS REPORT
# ─────────────────────────────────────────────────────────────────────────────
def page_report():
    if "results" not in st.session_state:
        st.info("Complete the assessment first to see your personalised report.")
        if st.button("Go to Assessment →"):
            st.session_state["current_page"] = PAGES[0]
            st.rerun()
        return

    r         = st.session_state["results"]
    tier_name = r["tier_name"]
    tier      = r["tier"]
    proba     = r["proba"]
    iks       = r["iks_score"]
    risk      = r["risk_score"]
    contribs  = r["contributions"]
    ds        = r["domain_scores"]
    user      = r["user"]
    inter     = INTERVENTIONS[tier]

    color  = TIER_COLORS[tier_name]
    bg     = TIER_BG[tier_name]

    SUMMARIES = {
        "Low":      "Your stress is well-managed. The focus now is on maintaining what's working.",
        "Moderate": "You're feeling the pressure, but structured daily practice can measurably reduce it within 3–4 weeks.",
        "High":     "Your stress level needs immediate attention. Daily IKS practice and professional support are both recommended.",
    }

    # ── Tier banner ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="tier-banner" style="background:{bg};border-left:6px solid {color}">'
        f'<h1 style="color:{color}">'
        f'{"🟢" if tier == 0 else "🟡" if tier == 1 else "🔴"} {tier_name} Stress Level'
        f"</h1>"
        f'<p style="color:#1A202C">{SUMMARIES[tier_name]}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Three metric cards ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            _metric_card("Stress Tier", tier_name,
                         f"Confidence: {proba[tier]*100:.0f}%", color),
            unsafe_allow_html=True,
        )
    with c2:
        iks_status = "Low — needs attention" if iks < 25 else ("Building — keep going" if iks < 55 else "Strong — protective")
        st.markdown(
            _metric_card("IKS Wellness Score", f"{iks:.0f} / 100", iks_status, GOLD),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _metric_card("Primary Risk Factor", r["top_risk"],
                         "The factor deviating most from a healthy baseline", "#C53030"),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    # ── IKS progress bar ───────────────────────────────────────────────────────
    _section("Your IKS Wellness Score")
    iks_color = "#C53030" if iks < 25 else ("#B7791F" if iks < 55 else "#2F855A")
    st.markdown(
        f"Your **IKS Wellness Score is {iks:.0f}/100** — this measures how much your "
        f"yoga, mindfulness, and pranayama practice is protecting you from stress. "
        f"The higher the score, the more buffered you are against stressors."
    )
    st.progress(int(iks), text=f"{iks:.0f}%")
    st.caption(
        "Score bands: 0–24 = Needs significant improvement · "
        "25–54 = Building protective habits · 55–100 = Strong IKS engagement"
    )

    # ── Gauge + class probabilities ────────────────────────────────────────────
    _section("Stress Risk Level")
    gcol, pcol = st.columns([1, 1])

    with gcol:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk,
            title={"text": "Overall Stress Risk Score", "font": {"size": 14, "color": NAVY}},
            delta={"reference": 33, "decreasing": {"color": "#2F855A"},
                   "increasing": {"color": "#C53030"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4A5568"},
                "bar":  {"color": color, "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 33],  "color": "#EBF4FF"},
                    {"range": [33, 67], "color": "#FFFAF0"},
                    {"range": [67, 100],"color": "#FFF5F5"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": risk,
                },
            },
            number={"suffix": "/100", "font": {"color": color, "size": 28}},
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20),
                                 paper_bgcolor="#FFFFFF", font={"family": "sans-serif"})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with pcol:
        st.markdown("**How confident is the model in each category?**")
        prob_df = pd.DataFrame({
            "Stress Level": LABEL_NAMES,
            "Confidence": [p * 100 for p in proba],
        })
        fig_bar = px.bar(
            prob_df, x="Confidence", y="Stress Level", orientation="h",
            color="Stress Level",
            color_discrete_map=TIER_COLORS,
            text="Confidence",
            range_x=[0, 100],
        )
        fig_bar.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig_bar.update_layout(
            showlegend=False, height=240,
            margin=dict(t=10, b=10, l=10, r=40),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F7F8FA",
            xaxis_title="Confidence (%)", yaxis_title="",
            font={"family": "sans-serif", "color": "#1A202C"},
        )
        fig_bar.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(
            f"The model predicts **{tier_name} Stress** with **{proba[tier]*100:.0f}% confidence**, "
            f"based on Logistic Regression trained on 1,100 student records (ROC-AUC = 0.889)."
        )

    # ── Feature contribution chart ─────────────────────────────────────────────
    _section("What's Driving Your Stress")
    st.markdown(
        '<div class="disclaimer-box">💡 <strong>Note:</strong> Approximate feature contribution '
        "based on deviation from healthy student baseline (low-stress cohort mean). "
        "Positive bars = raising your risk. Negative bars = protective factors.</div>",
        unsafe_allow_html=True,
    )

    contrib_df = pd.DataFrame(contribs)
    contrib_df["abs"] = contrib_df["contribution"].abs()
    contrib_df["direction"] = contrib_df["contribution"].apply(
        lambda x: "Raising stress risk" if x > 0 else "Protective factor"
    )
    contrib_df["bar_val"] = contrib_df["contribution"]

    fig_contrib = go.Figure()
    for _, row in contrib_df.iterrows():
        fig_contrib.add_trace(go.Bar(
            y=[row["label"]],
            x=[row["bar_val"]],
            orientation="h",
            marker_color=row["color"],
            marker_line_color="white",
            marker_line_width=1,
            name=row["domain"],
            showlegend=False,
        ))

    fig_contrib.add_vline(x=0, line_width=1.5, line_color="#4A5568")
    fig_contrib.update_layout(
        height=260,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F7F8FA",
        xaxis_title="Contribution (positive = increases risk)",
        yaxis_title="",
        font={"family": "sans-serif", "color": "#1A202C"},
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", zeroline=False),
        yaxis=dict(autorange="reversed"),
        barmode="overlay",
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

    # Annotations under chart
    for c in contribs:
        direction_icon = "⬆️" if c["contribution"] > 0 else "✅"
        st.markdown(f"{direction_icon} **{c['label']}** — {c['annotation']}")

    # ── Domain health overview ─────────────────────────────────────────────────
    _section("Your Wellbeing at a Glance")
    ds_labels = {
        "sleep": "😴 Sleep Quality", "social": "👥 Social Support",
        "academic": "📚 Academic", "physical": "💪 Physical Health",
        "environmental": "🏠 Environment", "psychological": "🧠 Psychological",
    }
    col_pairs = list(ds.items())
    cols = st.columns(3)
    for idx, (key, score) in enumerate(col_pairs):
        color_d = "#2F855A" if score >= 60 else ("#B7791F" if score >= 35 else "#C53030")
        with cols[idx % 3]:
            st.markdown(
                _metric_card(ds_labels.get(key, key), f"{score:.0f}%", "", color_d),
                unsafe_allow_html=True,
            )

    # ── CTA ───────────────────────────────────────────────────────────────────
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📋  See My Full Action Plan →", type="primary", use_container_width=True):
            st.session_state["current_page"] = PAGES[2]
            st.rerun()
    with c2:
        if st.button("🔄  Reset and Take Again", use_container_width=True):
            for k in ["results", "current_page"]:
                st.session_state.pop(k, None)
            st.session_state["current_page"] = PAGES[0]
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ACTION PLAN
# ─────────────────────────────────────────────────────────────────────────────
def page_action_plan():
    if "results" not in st.session_state:
        st.info("Complete the assessment first to unlock your personalised action plan.")
        if st.button("Go to Assessment →"):
            st.session_state["current_page"] = PAGES[0]
            st.rerun()
        return

    r         = st.session_state["results"]
    tier      = r["tier"]
    tier_name = r["tier_name"]
    iks       = r["iks_score"]
    user      = r["user"]
    inter     = INTERVENTIONS[tier]
    ds        = r["domain_scores"]
    color     = TIER_COLORS[tier_name]

    st.markdown(f"## 📋 Your Action Plan — {tier_name} Stress")
    st.markdown(
        f"*IKS Wellness Score: **{iks:.0f}/100**. "
        f"Plan intensity is set for your **{tier_name.lower()} stress** tier.*"
    )
    st.markdown(f'<div style="color:{color};font-weight:600;margin-bottom:1rem">{inter["note"]}</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⚡ This Week", "📅 4-Week Plan", "🌱 Lifestyle Changes"])

    # ── Tab 1: This Week ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("Start with these practices **today**. Even one session creates measurable change.")
        st.divider()

        _section("🧘 Yoga Prescription")
        yoga_poses = inter["yoga"]
        cols = st.columns(2)
        yoga_icons = ["🌟", "🌿", "🌸", "💫"]
        for i, (asana, desc) in enumerate(yoga_poses):
            with cols[i % 2]:
                st.markdown(
                    _action_card(yoga_icons[i % 4], asana, desc, color),
                    unsafe_allow_html=True,
                )

        _section("🌬️ Breathwork (Pranayama)")
        prana_poses = inter["pranayama"]
        cols2 = st.columns(2)
        prana_icons = ["💨", "🌊", "🌬️", "✨"]
        for i, (name, desc) in enumerate(prana_poses):
            with cols2[i % 2]:
                st.markdown(
                    _action_card(prana_icons[i % 4], name, desc, GOLD),
                    unsafe_allow_html=True,
                )

        _section("🧠 Mindfulness")
        st.markdown(
            f'<div class="domain-section"><p>{inter["mindfulness"]}</p></div>',
            unsafe_allow_html=True,
        )

        _section("🌿 Ayurveda & Diet")
        st.markdown(
            f'<div class="domain-section"><p>{inter["ayurveda"]}</p></div>',
            unsafe_allow_html=True,
        )

        _section("📆 Practice Frequency")
        st.markdown(
            f'<div class="domain-section"><p>{inter["frequency"]}</p></div>',
            unsafe_allow_html=True,
        )

    # ── Tab 2: 4-Week Plan ────────────────────────────────────────────────────
    with tab2:
        st.markdown(
            "A structured 4-week progression. Each week builds on the last. "
            "Consistency matters more than duration — 20 minutes daily beats 2 hours once a week."
        )
        st.divider()

        plan = FOUR_WEEK_PLANS[tier]
        week_colors = [NAVY, "#2B6CB0", GOLD, "#2F855A"]
        for i, w in enumerate(plan):
            wc = f'<div class="week-card" style="border-left-color:{week_colors[i]}">'
            wc += f'<div class="week-label">{w["week"]}</div>'
            wc += f'<div class="week-title">{w["title"]}</div>'
            wc += f'<div class="week-body"><strong>Practice:</strong> {w["practice"]}<br>'
            wc += f'<strong>Time commitment:</strong> {w["duration"]}</div>'
            wc += f'<div class="week-expect">✓ What to expect: {w["expect"]}</div></div>'
            st.markdown(wc, unsafe_allow_html=True)

    # ── Tab 3: Lifestyle Changes ───────────────────────────────────────────────
    with tab3:
        st.markdown(
            "Personalised advice for each area of your life. "
            "Domains where you're already doing well are highlighted in green."
        )
        st.divider()

        DOMAIN_RENDER = [
            ("🌙 Sleep Hygiene",       "sleep",         _sleep_advice(user)),
            ("👥 Social & Relationships","social",       _social_advice(user)),
            ("📚 Academic Strategies", "academic",      _academic_advice(user)),
            ("💪 Physical Health",     "physical",      _physical_advice(user)),
            ("🏠 Environment",         "environmental", _environmental_advice(user)),
            ("🧠 Psychological Health","psychological", _psychological_advice(user)),
        ]

        for icon_title, domain_key, tips in DOMAIN_RENDER:
            score = ds.get(domain_key, 50)
            if score >= 70:
                status_html = f'<span class="domain-good">✓ Already good ({score:.0f}%)</span>'
            else:
                status_html = f'<span style="color:#C53030">Needs attention ({score:.0f}%)</span>'

            html = (
                f'<div class="domain-section">'
                f"<h4>{icon_title} &nbsp; {status_html}</h4>"
            )
            for tip in tips:
                html += f"<p>• {tip}</p>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

        # Warning signs
        WARNINGS = {
            0: ["Anxiety rising above your usual baseline for more than 3 consecutive days",
                "Sleep quality dropping noticeably without an obvious cause",
                "Withdrawal from social contact"],
            1: ["Persistent inability to concentrate for more than 10 minutes",
                "Missing meals or sleep regularly",
                "Feeling hopeless about your academic situation"],
            2: ["Any thoughts of self-harm — please call a crisis line immediately",
                "Complete inability to attend classes or complete basic tasks",
                "Panic attacks occurring more than twice per week"],
        }
        st.markdown(
            '<div class="warning-box">'
            "<h4>🚨 Warning Signs to Watch</h4>"
            "<p>If you experience any of the following, seek professional support immediately:</p>"
            + "".join(f"<p>⚠️ {w}</p>" for w in WARNINGS[tier])
            + "</div>",
            unsafe_allow_html=True,
        )

        # Tracking table
        _section("📊 Track These 5 Things Weekly")
        st.caption("Simple self-monitoring accelerates recovery. Review once a week — Sunday evening works well.")
        tracking_df = pd.DataFrame(
            TRACKING_TABLE,
            columns=["What to Track", "How", "Good Sign", "Concern"],
        )
        st.dataframe(tracking_df, use_container_width=True, hide_index=True)

    st.divider()
    if st.button("🔄  Reset and Take Again", use_container_width=False):
        for k in ["results", "current_page"]:
            st.session_state.pop(k, None)
        st.session_state["current_page"] = PAGES[0]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
def page_about(df: pd.DataFrame):
    st.markdown("## ℹ️ About MindBalance v2")

    st.markdown("""
MindBalance is a student stress prediction and intervention tool built on a dataset
of **1,100 student records** spanning 24 variables across five domains:
psychological, physiological, environmental, academic, and social.

What makes it different from generic stress apps is the **IKS (Indian Knowledge Systems)**
component — we measure and reward yoga, pranayama, and mindfulness practice as evidence-based
protective factors against student stress, not just as add-ons.
""")

    # ── Why IKS? ──────────────────────────────────────────────────────────────
    _section("Why Indian Knowledge Systems (IKS)?")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Yoga** — Clinical trials consistently show yoga reduces cortisol (the primary stress hormone)
by 15–30% after 4–8 weeks of regular practice.

**Pranayama** — Alternate-nostril breathing (Nadi Shodhana) has been shown to reduce
anxiety scores (HAM-A) by up to 44% in a 12-week controlled trial.
""")
    with col_b:
        st.markdown("""
**Yoga Nidra** — Delivers physiological rest equivalent to 3–4 hours of sleep in 40 minutes.
The most evidence-backed IKS tool for high stress.

**Mindfulness** — 15 minutes daily reduces perceived stress scale (PSS) scores by an average
of 6 points over 8 weeks, across multiple meta-analyses.
""")

    # ── IKS Formula ───────────────────────────────────────────────────────────
    _section("How Your IKS Wellness Score is Calculated")
    st.markdown("Four practice components are scored and weighted:")

    formula_items = [
        ("Yoga Days/Week",       "35%", "0–7 days",  "#2B6CB0"),
        ("Yoga Duration (mins)", "20%", "0–60 min",  "#2F855A"),
        ("Mindfulness (mins/day)","25%","0–60 min",  "#C8973A"),
        ("Pranayama Practice",   "20%", "Yes / No",  NAVY),
    ]
    cols = st.columns(4)
    for i, (label, weight, scale, col_color) in enumerate(formula_items):
        with cols[i]:
            st.markdown(
                f'<div class="formula-chip" style="background:{col_color}20;border:2px solid {col_color}">'
                f'<span style="font-size:1.3rem;font-weight:800;color:{col_color}">{weight}</span>'
                f'<span style="color:#1A202C;font-size:0.82rem">{label}</span>'
                f'<span style="color:#718096;font-size:0.75rem">{scale}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.caption(
        "Formula: IKS Score = Σ (component_value / max_value) × weight × 100. "
        "Weights are derived from the relative efficacy shown in peer-reviewed IKS literature."
    )

    # ── Model performance ──────────────────────────────────────────────────────
    _section("Model Transparency")
    metrics = load_metrics()
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best Model",  metrics.get("best_model", "—"))
        m2.metric("Accuracy",    f"{metrics.get('accuracy', 0)*100:.1f}%")
        m3.metric("ROC-AUC",     f"{metrics.get('roc_auc', 0):.3f}")
        m4.metric("Cohen's κ",   f"{metrics.get('cohen_kappa', 0):.3f}")
        st.caption("Evaluated on a held-out test set (20% of 1,100 records, stratified).")

    comparison_df = load_model_comparison()
    if comparison_df is not None:
        st.markdown("**All models compared (ranked by ROC-AUC):**")
        display_df = comparison_df.copy()
        for col in ["Accuracy", "F1 (weighted)", "ROC-AUC", "Cohen Kappa", "CV Accuracy"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Bar chart of model comparison
        plot_df = comparison_df.copy()
        fig_models = px.bar(
            plot_df.melt(id_vars="Model", value_vars=["Accuracy", "F1 (weighted)", "ROC-AUC", "Cohen Kappa"],
                         var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric",
            barmode="group",
            color_discrete_sequence=[NAVY, GOLD, "#2F855A", "#C53030"],
            range_y=[0, 1.05],
        )
        fig_models.update_layout(
            height=340,
            margin=dict(t=20, b=40, l=0, r=0),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F7F8FA",
            font={"family": "sans-serif", "color": "#1A202C"},
            legend=dict(orientation="h", y=-0.25),
            xaxis=dict(tickangle=-15),
        )
        fig_models.update_xaxes(showgrid=False)
        fig_models.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
        st.plotly_chart(fig_models, use_container_width=True)

    # ── SHAP global bar image ──────────────────────────────────────────────────
    _section("Feature Importance — What Drives Student Stress Most?")
    shap_img = os.path.join(ROOT, "outputs", "06_shap_global_bar.png")
    if os.path.exists(shap_img):
        st.image(shap_img, caption="Top 10 features by SHAP importance (Logistic Regression)")
    else:
        st.caption("Run `python src/shap_explainer.py` to generate the feature importance chart.")

    st.info(
        "🧘 **Key finding:** Yoga Days/Week ranks in the **top 5 of all 24 features** "
        "in terms of predictive importance — ahead of many conventional stress factors. "
        "This confirms that IKS practice is not just a lifestyle add-on but a clinically meaningful variable."
    )

    # ── Data note ──────────────────────────────────────────────────────────────
    _section("Data & Methodology")
    st.markdown("""
| Detail | Value |
|---|---|
| Dataset size | 1,100 student records |
| Feature count | 24 (5 domains + 4 IKS variables) |
| Target variable | Stress level (Low / Moderate / High) |
| Class balance | Handled with SMOTE oversampling |
| Train/test split | 80% / 20%, stratified |
| Cross-validation | 5-fold stratified |
| Best model | Logistic Regression (CV Accuracy: 70.1%, ROC-AUC: 88.9%) |

*MindBalance is a research prototype. It is not a clinical diagnostic tool.
Please consult a qualified mental health professional for medical advice.*
""")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
def build_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f'<div style="text-align:center;padding:1rem 0">'
            f'<span style="font-size:2.2rem">🧘</span><br>'
            f'<span style="font-weight:800;font-size:1.2rem;color:{NAVY}">MindBalance</span><br>'
            f'<span style="font-size:0.78rem;color:#718096">Student Stress Assessment v2</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        if "current_page" not in st.session_state:
            st.session_state["current_page"] = PAGES[0]

        current_idx = PAGES.index(st.session_state["current_page"]) \
            if st.session_state["current_page"] in PAGES else 0

        page = st.radio(
            "Navigate", PAGES,
            index=current_idx,
            label_visibility="collapsed",
        )
        st.session_state["current_page"] = page

        st.divider()

        # Assessment status
        if "results" in st.session_state:
            r = st.session_state["results"]
            tn = r["tier_name"]
            tc = TIER_COLORS[tn]
            st.markdown(
                f'<div style="background:{TIER_BG[tn]};border-left:3px solid {tc};'
                f'border-radius:6px;padding:0.6rem 0.9rem;margin-bottom:0.5rem">'
                f'<div style="font-size:0.72rem;color:#718096;font-weight:600">LAST RESULT</div>'
                f'<div style="font-weight:700;color:{tc}">{tn} Stress</div>'
                f'<div style="font-size:0.78rem;color:#4A5568">IKS Score: {r["iks_score"]:.0f}/100</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("🔄 Reset", use_container_width=True):
                for k in ["results", "current_page"]:
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            st.caption("No assessment yet. Complete the assessment to see your report.")

        st.divider()
        st.markdown(
            '<div style="font-size:0.74rem;color:#A0AEC0;text-align:center">'
            "Built with Streamlit · scikit-learn · IKS principles<br>"
            "Data stays on your device. Nothing is stored."
            "</div>",
            unsafe_allow_html=True,
        )

    return page


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    page = build_sidebar()

    # Load resources
    pipe = load_model()

    if pipe is None and page != PAGES[3]:
        st.error(
            "**Model not found.**  \n"
            "Run `python src/train.py` first to train and save the model, "
            "then restart the app."
        )
        st.code("python src/train.py", language="bash")
        return

    df = load_dataset()

    if page == PAGES[0]:
        page_assessment(pipe, df)
    elif page == PAGES[1]:
        page_report()
    elif page == PAGES[2]:
        page_action_plan()
    else:
        page_about(df)


if __name__ == "__main__":
    main()

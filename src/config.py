"""Central configuration for MindBalance v2."""
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "StressLevelDataset.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_DIR   = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

TARGET      = "stress_level"
LABELS      = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_NAMES = ["Low", "Moderate", "High"]

# Feature groups
PSYCHOLOGICAL  = ["anxiety_level","self_esteem","mental_health_history","depression"]
PHYSIOLOGICAL  = ["headache","blood_pressure","sleep_quality","breathing_problem"]
ENVIRONMENTAL  = ["noise_level","living_conditions","safety","basic_needs"]
ACADEMIC       = ["academic_performance","study_load","teacher_student_rel","future_career_concerns"]
SOCIAL         = ["social_support","peer_pressure","extracurricular_act","bullying"]
IKS            = ["yoga_days_per_week","yoga_duration_mins","mindfulness_mins_day","pranayama_practice"]

ALL_FEATURES = PSYCHOLOGICAL + PHYSIOLOGICAL + ENVIRONMENTAL + ACADEMIC + SOCIAL + IKS

WIDE_SCALE   = ["anxiety_level", "self_esteem", "depression",
                "yoga_duration_mins", "mindfulness_mins_day"]
NARROW_SCALE = [f for f in ALL_FEATURES if f not in WIDE_SCALE]

# Palette
NAVY    = "#1B3A6B"
GOLD    = "#C8973A"
SLATE   = "#4A5568"
LOW_C   = "#2B6CB0"
MED_C   = "#B7791F"
HIGH_C  = "#C53030"
SAGE    = "#2F855A"
PALETTE = [LOW_C, MED_C, HIGH_C]

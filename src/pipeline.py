"""
MindBalance v2 | pipeline.py
Sklearn Pipeline: preprocessing → SMOTE → model
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from config import ALL_FEATURES, PSYCHOLOGICAL, PHYSIOLOGICAL, ENVIRONMENTAL, ACADEMIC, SOCIAL, IKS

# Features with larger scale need StandardScaler
WIDE_SCALE = ["anxiety_level", "self_esteem", "depression",
              "yoga_duration_mins", "mindfulness_mins_day"]
NARROW_SCALE = [f for f in ALL_FEATURES if f not in WIDE_SCALE]

def build_preprocessor():
    """
    ColumnTransformer:
    - wide-scale features (0-30 range): StandardScaler
    - narrow-scale features (0-5 range): MinMaxScaler
    - all: SimpleImputer(median) first
    """
    wide = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    narrow = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  MinMaxScaler())
    ])
    return ColumnTransformer([
        ("wide",   wide,   WIDE_SCALE),
        ("narrow", narrow, NARROW_SCALE),
    ], remainder="drop")


def build_pipeline(model, use_smote=True):
    """
    Build full ImbPipeline with preprocessing + optional SMOTE + model.
    Returns an imblearn Pipeline (supports SMOTE in fit step only).
    """
    steps = [("preprocessor", build_preprocessor())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=5)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def load_data(data_path):
    """Load dataset, return X, y."""
    df = pd.read_csv(data_path)
    X  = df[ALL_FEATURES]
    y  = df["stress_level"]
    return X, y

# loader.py  — load models, data, bias correction
import os
import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st
from config import (
    MODEL_PATH, MODEL_P10_PATH, MODEL_P90_PATH,
    FEATURES_PATH, BIAS_PATH, PINCODE_PATH,
    FEATURES_DF, STOCK_PATH
)


# ── load models ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        FEATURE_COLS = pickle.load(f)
    model_p10 = None
    model_p90 = None
    if os.path.exists(MODEL_P10_PATH):
        with open(MODEL_P10_PATH, 'rb') as f:
            model_p10 = pickle.load(f)
    if os.path.exists(MODEL_P90_PATH):
        with open(MODEL_P90_PATH, 'rb') as f:
            model_p90 = pickle.load(f)
    return model, model_p10, model_p90, FEATURE_COLS


# ── load data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ts_df    = pd.read_csv(FEATURES_DF)
    selected = pd.read_csv(PINCODE_PATH)
    ts_df['datetime'] = pd.to_datetime(ts_df['datetime'])

    if os.path.exists(STOCK_PATH):
        stock_df = pd.read_csv(STOCK_PATH)
    else:
        avg = (ts_df.groupby(['product_id','product_name','pincode'])['demand']
               .mean().reset_index())
        avg['avg_hourly_real'] = np.expm1(avg['demand'])
        avg['avg_daily_real']  = avg['avg_hourly_real'] * 24

        np.random.seed(42)
        avg['current_stock'] = (
            avg['avg_daily_real'] *
            np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5],
                             size=len(avg), p=[0.10, 0.25, 0.35, 0.20, 0.10])
            + np.random.normal(0, 2, len(avg))
        ).clip(lower=0).round(0)

        stock_df = avg[['product_id','product_name','pincode','current_stock']].copy()
        stock_df.to_csv(STOCK_PATH, index=False)

    return ts_df, selected, stock_df


# ── load bias correction ──────────────────────────────────────────────
def load_bias(bias_path):
    if os.path.exists(bias_path):
        with open(bias_path) as f:
            return json.load(f)['bias_factor']
    return 1.0
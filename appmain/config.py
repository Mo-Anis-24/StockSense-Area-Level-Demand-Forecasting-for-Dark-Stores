# config.py  — all constants, paths, and live weather fetch
import numpy as np
import requests
import streamlit as st
from datetime import datetime

# ── paths ─────────────────────────────────────────────────────────────
MODEL_PATH     = r"F:\projectss\personal\blinkit_project\noteboook\models\lgbm_best.pkl"
MODEL_P10_PATH = r"F:\projectss\personal\blinkit_project\noteboook\models\lgbm_p10.pkl"
MODEL_P90_PATH = r"F:\projectss\personal\blinkit_project\noteboook\models\lgbm_p90.pkl"
FEATURES_PATH  = r"F:\projectss\personal\blinkit_project\noteboook\models\feature_cols.pkl"
BIAS_PATH      = r"F:\projectss\personal\blinkit_project\noteboook\models\bias_factor.json"
PINCODE_PATH   = r"F:\projectss\personal\blinkit_project\data\external\selected_ahmedabad_pincodes.csv"
FEATURES_DF    = r"F:\projectss\personal\blinkit_project\data\processed\features_df_fixed.csv"
STOCK_PATH     = r"F:\projectss\personal\blinkit_project\data\processed\stock_levels.csv"

# ── area density weights ───────────────────────────────────────────────
AREA_DENSITY = {
    380009: 0.18,
    380015: 0.17,
    380051: 0.15,
    380006: 0.14,
    380013: 0.12,
    380019: 0.10,
    380021: 0.08,
    382210: 0.03,
    382330: 0.02,
    382435: 0.01,
}

# ── auto detect today ─────────────────────────────────────────────────
NOW        = datetime.now()
TODAY_DOW  = NOW.weekday()
TODAY_HOUR = NOW.hour
TODAY_WEEK = NOW.isocalendar()[1]
TODAY_NAME = ['Monday','Tuesday','Wednesday','Thursday',
              'Friday','Saturday','Sunday'][TODAY_DOW]
TODAY_DATE = NOW.strftime('%d %B %Y')
DAY_NAMES  = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

# ── fetch live weather ────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_live_weather():
    try:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude"       : 23.0225,
            "longitude"      : 72.5714,
            "current_weather": True,
            "hourly"         : "relativehumidity_2m,precipitation",
            "timezone"       : "Asia/Kolkata",
            "forecast_days"  : 1,
        }
        r        = requests.get(url, params=params, timeout=5).json()
        curr     = r['current_weather']
        humidity = r['hourly']['relativehumidity_2m'][TODAY_HOUR]
        rainfall = r['hourly']['precipitation'][TODAY_HOUR]
        return {
            'temperature': curr['temperature'],
            'windspeed'  : curr['windspeed'],
            'humidity'   : humidity,
            'rainfall'   : rainfall,
            'is_raining' : rainfall > 0.5,
            'is_hot'     : curr['temperature'] > 35,
            'source'     : 'live'
        }
    except Exception:
        return {
            'temperature': 32.0,
            'windspeed'  : 10.0,
            'humidity'   : 60.0,
            'rainfall'   : 0.0,
            'is_raining' : False,
            'is_hot'     : False,
            'source'     : 'fallback'
        }
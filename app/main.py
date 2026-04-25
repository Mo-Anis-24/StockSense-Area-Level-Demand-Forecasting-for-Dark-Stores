# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = FastAPI(title="DemandPulse API", version="1.0")

# ── load model and features on startup ───────────────────────────────
with open('models/lgbm_best.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/feature_cols.pkl', 'rb') as f:
    FEATURE_COLS = pickle.load(f)

selected = pd.read_csv('data/external/selected_ahmedabad_pincodes.csv')
PINCODES = selected['pincode'].tolist()

# load weather stats for fallback
weather_df = pd.read_csv('data/external/ahmedabad_weather_2024.csv')
AVG_TEMP   = weather_df['temperature'].median()
AVG_HUM    = weather_df['humidity'].median()

# ── input schema ──────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    pincode    : int
    product_id : int
    hour       : int   # 0-23
    dow        : int   # 0-6
    week       : int   # 1-52
    lag_1      : float = 50.0
    lag_24     : float = 50.0
    lag_48     : float = 50.0
    lag_168    : float = 50.0
    temperature: float = 32.0
    rainfall   : float = 0.0
    humidity   : float = 60.0

# ── helper — build feature vector ─────────────────────────────────────
def build_features(req: PredictRequest) -> pd.DataFrame:
    hour = req.hour
    dow  = req.dow
    week = req.week

    row = {
        'hour_sin'         : np.sin(2 * np.pi * hour / 24),
        'hour_cos'         : np.cos(2 * np.pi * hour / 24),
        'dow_sin'          : np.sin(2 * np.pi * dow  / 7),
        'dow_cos'          : np.cos(2 * np.pi * dow  / 7),
        'week_sin'         : np.sin(2 * np.pi * week / 52),
        'week_cos'         : np.cos(2 * np.pi * week / 52),
        'month'            : datetime.now().month,
        'is_weekend'       : int(dow in [5, 6]),
        'is_morning'       : int(6  <= hour <= 11),
        'is_afternoon'     : int(12 <= hour <= 17),
        'is_evening'       : int(18 <= hour <= 22),
        'is_night'         : int(hour in [23,0,1,2,3,4,5]),
        'is_festival'      : 0,
        'lag_1'            : req.lag_1,
        'lag_24'           : req.lag_24,
        'lag_48'           : req.lag_48,
        'lag_168'          : req.lag_168,
        'rolling_mean_24'  : req.lag_24,
        'rolling_mean_168' : req.lag_168,
        'rolling_mean_720' : req.lag_168,
        'rolling_std_24'   : 5.0,
        'rolling_std_168'  : 8.0,
        'rolling_max_24'   : req.lag_1 * 1.2,
        'rolling_max_168'  : req.lag_168 * 1.2,
        'ewma_24'          : req.lag_24,
        'ewma_168'         : req.lag_168,
        'temperature'      : req.temperature,
        'rainfall'         : req.rainfall,
        'humidity'         : req.humidity,
        'is_raining'       : int(req.rainfall > 0.5),
        'is_hot'           : int(req.temperature > 35),
    }

    return pd.DataFrame([row])[FEATURE_COLS]

# ── routes ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "lgbm_best", "features": len(FEATURE_COLS)}

@app.post("/predict")
def predict(req: PredictRequest):
    features   = build_features(req)
    prediction = float(np.clip(model.predict(features)[0], 0, None))
    return {
        "pincode"         : req.pincode,
        "product_id"      : req.product_id,
        "hour"            : req.hour,
        "predicted_demand": round(prediction, 2),
        "unit"            : "orders"
    }

@app.get("/pincodes")
def get_pincodes():
    df = pd.read_csv('data/external/selected_ahmedabad_pincodes.csv')
    return df.to_dict(orient='records')
# predictor.py  — feature builder + all prediction functions
import numpy as np
import pandas as pd
import streamlit as st
from config import NOW, AREA_DENSITY


# ── feature builder ───────────────────────────────────────────────────
def build_features(hour, dow, week, lag_1, lag_24, lag_168,
                   temperature, rainfall, humidity,
                   area_density=0.10, feature_cols=None):
    row = {
        'hour_sin'         : np.sin(2 * np.pi * hour / 24),
        'hour_cos'         : np.cos(2 * np.pi * hour / 24),
        'dow_sin'          : np.sin(2 * np.pi * dow  / 7),
        'dow_cos'          : np.cos(2 * np.pi * dow  / 7),
        'week_sin'         : np.sin(2 * np.pi * week / 52),
        'week_cos'         : np.cos(2 * np.pi * week / 52),
        'month'            : NOW.month,
        'is_weekend'       : int(dow in [5, 6]),
        'is_morning'       : int(6  <= hour <= 11),
        'is_afternoon'     : int(12 <= hour <= 17),
        'is_evening'       : int(18 <= hour <= 22),
        'is_night'         : int(hour in [23,0,1,2,3,4,5]),
        'is_festival'      : 0,
        'lag_1'            : lag_1,
        'lag_24'           : lag_24,
        'lag_48'           : lag_24,
        'lag_168'          : lag_168,
        'rolling_mean_24'  : lag_24,
        'rolling_mean_168' : lag_168,
        'rolling_mean_720' : lag_168,
        'rolling_std_24'   : 5.0,
        'rolling_std_168'  : 8.0,
        'rolling_max_24'   : lag_1   * 1.2,
        'rolling_max_168'  : lag_168 * 1.2,
        'ewma_24'          : lag_24,
        'ewma_168'         : lag_168,
        'temperature'      : temperature,
        'rainfall'         : rainfall,
        'humidity'         : humidity,
        'is_raining'       : int(rainfall > 0.5),
        'is_hot'           : int(temperature > 35),
        'area_density'     : area_density,
    }
    return pd.DataFrame([row])[[c for c in feature_cols if c in row]]


# ── model output → real hourly orders ────────────────────────────────
def predict_hourly(raw_pred_log, bias_factor=1.0):
    return np.expm1(float(np.clip(raw_pred_log, 0, None))) * bias_factor


# ── stock status ──────────────────────────────────────────────────────
def get_stock_status(hourly_demand, daily_stock):
    hourly_stock = daily_stock / 24.0
    gap = hourly_demand - hourly_stock
    if gap > hourly_stock * 0.5        : return gap, '🔴 CRITICAL'
    elif gap > 0                        : return gap, '🟡 LOW'
    elif abs(gap) < hourly_stock * 0.2  : return gap, '🟢 OK'
    else                                : return gap, '🔵 SURPLUS'


# ── helper — get lags for area ────────────────────────────────────────
def get_area_lags(pincode, ts_df):
    hist    = ts_df[ts_df['pincode'] == pincode].sort_values('datetime')
    lag_1   = float(hist['demand'].iloc[-1])   if len(hist) >= 1   else 1.0
    lag_24  = float(hist['demand'].iloc[-24])  if len(hist) >= 24  else 1.0
    lag_168 = float(hist['demand'].iloc[-168]) if len(hist) >= 168 else 1.0
    return lag_1, lag_24, lag_168


# ── predict one pincode for one specific hour ─────────────────────────
def predict_pincode(pincode, hour, dow, week,
                    temperature, rainfall, humidity,
                    ts_df, stock_df, feature_cols, model,
                    model_p10=None, model_p90=None,
                    bias_factor=1.0, with_intervals=False):

    HAS_QUANTILE = model_p10 is not None and model_p90 is not None

    EMPTY = pd.DataFrame({
        'product_id'      : pd.Series(dtype='int'),
        'product_name'    : pd.Series(dtype='str'),
        'predicted_demand': pd.Series(dtype='float'),
        'current_stock'   : pd.Series(dtype='float'),
        'stock_gap'       : pd.Series(dtype='float'),
        'status'          : pd.Series(dtype='str'),
    })

    if pincode not in ts_df['pincode'].values:
        return EMPTY

    products = ts_df[ts_df['pincode'] == pincode][
        ['product_id','product_name']
    ].drop_duplicates()

    if products.empty:
        return EMPTY

    density = AREA_DENSITY.get(pincode, 0.10)
    results = []

    for _, row in products.iterrows():
        hist = ts_df[
            (ts_df['product_id'] == row['product_id']) &
            (ts_df['pincode']    == pincode)
        ].sort_values('datetime')

        if hist.empty:
            continue

        lag_1   = float(hist['demand'].iloc[-1])   if len(hist) >= 1   else 1.0
        lag_24  = float(hist['demand'].iloc[-24])  if len(hist) >= 24  else 1.0
        lag_168 = float(hist['demand'].iloc[-168]) if len(hist) >= 168 else 1.0

        features = build_features(
            hour, dow, week, lag_1, lag_24, lag_168,
            temperature, rainfall, humidity,
            area_density=density, feature_cols=feature_cols
        )

        predicted = predict_hourly(model.predict(features)[0], bias_factor)

        stock_match = stock_df[
            (stock_df['product_id'] == row['product_id']) &
            (stock_df['pincode']    == pincode)
        ]
        current_stock = float(stock_match['current_stock'].values[0]) \
                        if not stock_match.empty else round(predicted * 48, 0)

        gap, status = get_stock_status(predicted, current_stock)

        entry = {
            'product_id'      : row['product_id'],
            'product_name'    : row['product_name'],
            'predicted_demand': round(predicted,     2),
            'current_stock'   : round(current_stock, 0),
            'stock_gap'       : round(gap,           2),
            'status'          : status,
        }

        if with_intervals and HAS_QUANTILE:
            p10 = predict_hourly(model_p10.predict(features)[0], bias_factor)
            p90 = predict_hourly(model_p90.predict(features)[0], bias_factor)
            entry['demand_low']  = round(p10, 2)
            entry['demand_high'] = round(p90, 2)
            entry['range']       = f"{p10:.1f} – {p90:.1f}"

        results.append(entry)

    if not results:
        return EMPTY

    return pd.DataFrame(results).sort_values(
        'predicted_demand', ascending=False
    ).reset_index(drop=True)


# ── predict all areas for one specific hour ───────────────────────────
@st.cache_data(ttl=900)
def predict_all_areas(_model, hour, dow, week,
                      temperature, rainfall, humidity,
                      _ts_df, _stock_df, _selected,
                      _feature_cols, _bias_factor):
    area_results = []
    for _, area_row in _selected.iterrows():
        pin = area_row['pincode']
        if pin not in _ts_df['pincode'].values:
            continue

        df     = predict_pincode(pin, hour, dow, week,
                                 temperature, rainfall, humidity,
                                 _ts_df, _stock_df, _feature_cols, _model,
                                 bias_factor=_bias_factor)
        total  = float(df['predicted_demand'].sum()) if not df.empty else 0.0
        top3   = df['product_name'].head(3).tolist()     if not df.empty else []
        top3_d = df['predicted_demand'].head(3).tolist() if not df.empty else []
        while len(top3)  < 3: top3.append('-')
        while len(top3_d)< 3: top3_d.append(0)

        critical = int((df['status'] == '🔴 CRITICAL').sum()) if not df.empty else 0
        low      = int((df['status'] == '🟡 LOW').sum())      if not df.empty else 0

        area_results.append({
            'pincode'       : pin,
            'area_name'     : area_row['area_name'],
            'latitude'      : float(area_row['latitude']),
            'longitude'     : float(area_row['longitude']),
            'total_demand'  : round(total, 1),
            'top_product_1' : top3[0],
            'top_product_2' : top3[1],
            'top_product_3' : top3[2],
            'top_demand_1'  : top3_d[0],
            'top_demand_2'  : top3_d[1],
            'top_demand_3'  : top3_d[2],
            'critical_items': critical,
            'low_items'     : low,
        })

    if not area_results:
        return pd.DataFrame(columns=[
            'pincode','area_name','latitude','longitude',
            'total_demand','top_product_1','top_product_2','top_product_3',
            'top_demand_1','top_demand_2','top_demand_3',
            'critical_items','low_items'
        ])
    return pd.DataFrame(area_results).sort_values(
        'total_demand', ascending=False
    ).reset_index(drop=True)


# ── predict full 24h curve for one area ──────────────────────────────
def predict_area_hourly(pincode, dow, week, temperature, rainfall, humidity,
                        ts_df, feature_cols, model, bias_factor=1.0):
    density = AREA_DENSITY.get(pincode, 0.10)
    lag_1, lag_24, lag_168 = get_area_lags(pincode, ts_df)
    hourly = []
    for h in range(24):
        f = build_features(h, dow, week, lag_1, lag_24, lag_168,
                           temperature, rainfall, humidity,
                           area_density=density, feature_cols=feature_cols)
        p = predict_hourly(model.predict(f)[0], bias_factor)
        hourly.append({'hour': h, 'demand': round(p, 2)})
    return pd.DataFrame(hourly)
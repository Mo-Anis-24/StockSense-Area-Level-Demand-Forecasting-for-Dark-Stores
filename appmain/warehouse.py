# warehouse.py  ← RUN THIS FILE:  streamlit run warehouse.py
import streamlit as st

st.set_page_config(
    page_title="DemandPulse — Warehouse",
    page_icon ="🏭",
    layout    ="wide"
)

import warnings
warnings.filterwarnings('ignore')

# ── import from the 4 helper files ───────────────────────────────────
from config   import (
    TODAY_NAME, TODAY_DATE, TODAY_DOW, TODAY_HOUR, TODAY_WEEK,
    DAY_NAMES, NOW, fetch_live_weather, BIAS_PATH
)
from loader   import load_model, load_data, load_bias
from predictor import predict_all_areas
from tab_views import (
    render_tab_map,
    render_tab_pack,
    render_tab_alerts,
    render_tab_overview,
    render_tab_simulator,
    render_tab_roi,
)

# ── load everything ───────────────────────────────────────────────────
model, model_p10, model_p90, FEATURE_COLS = load_model()
ts_df, selected, stock_df                 = load_data()
weather                                    = fetch_live_weather()
BIAS_FACTOR                               = load_bias(BIAS_PATH)
HAS_QUANTILE                              = model_p10 is not None and model_p90 is not None

# ── pincode mismatch check ────────────────────────────────────────────
ts_pincodes       = set(ts_df['pincode'].unique())
selected_pincodes = set(selected['pincode'].values)
missing_pins      = selected_pincodes - ts_pincodes

# ══════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════
st.title("🏭 DemandPulse — Warehouse Manager")

col_d, col_t, col_w, col_r, col_h, col_ws = st.columns(6)
col_d.metric("📅 Today",        TODAY_NAME)
col_t.metric("🕐 Current hour", f"{TODAY_HOUR:02d}:00")
col_w.metric("🌡️ Temperature",  f"{weather['temperature']}°C")
col_r.metric("🌧️ Rainfall",     f"{weather['rainfall']} mm")
col_h.metric("💧 Humidity",     f"{weather['humidity']}%")
col_ws.metric("💨 Wind",        f"{weather['windspeed']} km/h")

if weather['source'] == 'live':
    st.success("✅ Live weather data from Open-Meteo — Ahmedabad")
else:
    st.warning("⚠️ Using fallback weather — check internet connection")
if weather['is_raining']:
    st.info("🌧️ It is raining — expect higher demand for essentials")
if weather['is_hot']:
    st.info("🌡️ High temperature — expect higher demand for beverages and dairy")
if not HAS_QUANTILE:
    st.warning("⚠️ Quantile models not found — run Cells 13–16 in Notebook 03")
if missing_pins:
    st.sidebar.error(
        f"⚠️ {len(missing_pins)} pincodes not in training data: {missing_pins}"
    )
if BIAS_FACTOR != 1.0:
    st.sidebar.info(f"📐 Bias correction: ×{BIAS_FACTOR:.3f}")

st.markdown("---")

# ── sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Forecast Settings")
    st.markdown(f"**Auto-detected: {TODAY_NAME}, {TODAY_DATE}**")
    st.markdown("Override if needed:")

    hour = st.slider("Hour", 0, 23, TODAY_HOUR)
    dow  = st.selectbox(
        "Day",
        options    =[0,1,2,3,4,5,6],
        index      =TODAY_DOW,
        format_func=lambda x: DAY_NAMES[x],
        key        ="wh_dow"
    )
    week = TODAY_WEEK

    st.markdown("---")
    st.markdown("**🌤️ Weather (auto-fetched)**")
    temperature = st.slider("Temperature °C", 10, 48, int(weather['temperature']))
    rainfall    = st.slider("Rainfall mm",     0,  50, int(weather['rainfall']))
    humidity    = st.slider("Humidity %",      10, 100, int(weather['humidity']))

    st.markdown("---")
    st.button("🔄 Refresh Forecast", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption(
        f"Week {week} of year  \n"
        f"Showing **hourly** forecast for {hour:02d}:00  \n"
        f"Each number = orders expected in that 1-hour window"
    )

# ── run predictions ───────────────────────────────────────────────────
with st.spinner(f"🔮 Predicting demand for {hour:02d}:00 across all areas..."):
    all_areas_df = predict_all_areas(
        model, hour, dow, week, temperature, rainfall, humidity,
        ts_df, stock_df, selected, FEATURE_COLS, BIAS_FACTOR
    )

# ── tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️  Area Demand Map",
    "📋  Pack Priority List",
    "⚠️  Stock Alerts",
    "📊  Demand Overview",
    "🎮  What-if Simulator",
    "💰  ROI Calculator",
])

with tab1:
    render_tab_map(all_areas_df, hour)

with tab2:
    render_tab_pack(
        all_areas_df, hour, dow, week,
        temperature, rainfall, humidity,
        ts_df, stock_df, FEATURE_COLS,
        model, model_p10, model_p90,
        BIAS_FACTOR, HAS_QUANTILE
    )

with tab3:
    render_tab_alerts(
        all_areas_df, hour, dow, week,
        temperature, rainfall, humidity,
        ts_df, stock_df, FEATURE_COLS,
        model, BIAS_FACTOR
    )

with tab4:
    render_tab_overview(
        all_areas_df, hour, dow, week,
        temperature, rainfall, humidity,
        ts_df, selected, FEATURE_COLS,
        model, BIAS_FACTOR
    )

with tab5:
    render_tab_simulator(
        hour, dow, week, weather,
        ts_df, selected, FEATURE_COLS,
        model, model_p10, model_p90,
        BIAS_FACTOR, HAS_QUANTILE
    )

with tab6:
    render_tab_roi()

# ── footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"DemandPulse · Showing hourly forecast for {hour:02d}:00 · "
    f"Each number = predicted orders in that 1-hour window · "
    f"Weather refreshes every 30 mins · "
    f"Model: LightGBM · WAPE 20.86% · Bias ×{BIAS_FACTOR:.3f} · "
    f"Last updated: {NOW.strftime('%d %b %Y %H:%M')}"
)
# app/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title = "DemandPulse — HyperLocal Demand Forecasting",
    page_icon  = "📦",
    layout     = "wide"
)

# ── load assets ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(r'F:\projectss\personal\blinkit_project\noteboook\models\lgbm_best.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(r'F:\projectss\personal\blinkit_project\noteboook\models\feature_cols.pkl', 'rb') as f:
        FEATURE_COLS = pickle.load(f)
    return model, FEATURE_COLS
@st.cache_data
def load_data():
    ts_df      = pd.read_csv(r'F:\projectss\personal\blinkit_project\data\processed\features_df_fixed.csv')
    selected   = pd.read_csv(r'F:\projectss\personal\blinkit_project\data\external\selected_ahmedabad_pincodes.csv')
    results    = pd.read_csv(r'F:\projectss\personal\blinkit_project\data\processed\model_results_final.csv')
    ts_df['datetime'] = pd.to_datetime(ts_df['datetime'])
    return ts_df, selected, results

model, FEATURE_COLS = load_model()
ts_df, selected, results = load_data()

# ── helper — build features for prediction ────────────────────────────
def build_features(hour, dow, week, lag_1, lag_24, lag_168,
                   temperature, rainfall, humidity):
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
        'lag_1'            : lag_1,
        'lag_24'           : lag_24,
        'lag_48'           : lag_24,
        'lag_168'          : lag_168,
        'rolling_mean_24'  : lag_24,
        'rolling_mean_168' : lag_168,
        'rolling_mean_720' : lag_168,
        'rolling_std_24'   : 5.0,
        'rolling_std_168'  : 8.0,
        'rolling_max_24'   : lag_1 * 1.2,
        'rolling_max_168'  : lag_168 * 1.2,
        'ewma_24'          : lag_24,
        'ewma_168'         : lag_168,
        'temperature'      : temperature,
        'rainfall'         : rainfall,
        'humidity'         : humidity,
        'is_raining'       : int(rainfall > 0.5),
        'is_hot'           : int(temperature > 35),
    }
    return pd.DataFrame([row])[FEATURE_COLS]

# ── header ────────────────────────────────────────────────────────────
st.title("📦 DemandPulse")
st.markdown("#### HyperLocal Inventory Demand Forecasting — Ahmedabad")
st.markdown("---")

# ── tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Area Demand Map",
    "🔮 Live Prediction",
    "📊 Model Performance",
    "📈 Demand Trends"
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — area demand map
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Demand heatmap by Ahmedabad area")

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("**Filter**")
        selected_dow  = st.selectbox(
            "Day of week",
            options=[0,1,2,3,4,5,6],
            format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x],
            key="map_dow"  # ← add this
        )
        selected_hour = st.slider("Hour of day", 0, 23, 10)

    # compute average demand per pincode for selected dow+hour
    filtered = ts_df[
        (ts_df['order_dow'] == selected_dow) &
        (ts_df['order_hour_of_day'] == selected_hour)
    ].groupby(['pincode','area_name','latitude','longitude'])['demand'].mean().reset_index()

    # merge with selected pincodes to ensure all 10 show
    map_df = selected.merge(
        filtered[['pincode','demand']],
        on='pincode', how='left'
    ).fillna({'demand': 0})

    with col1:
        # build folium map centered on ahmedabad
        m = folium.Map(
            location=[23.0225, 72.5714],
            zoom_start=11,
            tiles='CartoDB positron'
        )

        max_demand = map_df['demand'].max() or 1

        for _, row in map_df.iterrows():
            demand    = row['demand']
            intensity = demand / max_demand
            radius    = 10 + intensity * 30

            # color: green low → red high
            r = int(255 * intensity)
            g = int(255 * (1 - intensity))
            color = f'#{r:02x}{g:02x}40'

            folium.CircleMarker(
                location  = [row['latitude'], row['longitude']],
                radius    = radius,
                color     = '#333',
                weight    = 1,
                fill      = True,
                fill_color= color,
                fill_opacity = 0.7,
                tooltip   = (f"<b>{row['area_name']}</b><br>"
                             f"Pincode: {row['pincode']}<br>"
                             f"Avg Demand: {demand:.0f} orders")
            ).add_to(m)

            folium.Marker(
                location = [row['latitude'], row['longitude']],
                icon     = folium.DivIcon(
                    html=f'<div style="font-size:9px;font-weight:500;'
                         f'color:#333;white-space:nowrap">'
                         f'{row["area_name"]}</div>'
                )
            ).add_to(m)

        st_folium(m, width=700, height=450)

    # demand table below map
    st.markdown("**Demand by area**")
    display_df = map_df[['area_name','pincode','demand']].copy()
    display_df.columns = ['Area','Pincode','Avg Demand']
    display_df['Avg Demand'] = display_df['Avg Demand'].round(1)
    display_df = display_df.sort_values('Avg Demand', ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — live prediction
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Predict demand for any product + area + time")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Location & Time**")
        pincode_options = dict(zip(selected['area_name'], selected['pincode']))
        area_name  = st.selectbox("Area", list(pincode_options.keys()))
        pincode    = pincode_options[area_name]
        hour = st.slider("Hour of day", 0, 23, 10, key="hour_slider")
       
        dow = st.selectbox(
            "Day of week",
            options=[0,1,2,3,4,5,6],
            format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x],
            key="predict_dow"  # ← add this
       )
        week = st.slider("Week of year", 1, 52, 20)

    with col2:
        st.markdown("**Recent demand (lag values)**")
        lag_1   = st.number_input("Demand 1 hour ago",   value=50.0, step=1.0)
        lag_24  = st.number_input("Demand 24 hours ago", value=45.0, step=1.0)
        lag_168 = st.number_input("Demand 1 week ago",   value=48.0, step=1.0)

    with col3:
        st.markdown("**Weather conditions**")
        temperature = st.slider("Temperature (°C)", 10, 48, 32)
        rainfall    = st.slider("Rainfall (mm)",     0,  50,  0)
        humidity    = st.slider("Humidity (%)",      10, 100, 60)

    st.markdown("---")

    if st.button("🔮 Predict Demand", type="primary"):
        features   = build_features(
            hour, dow, week, lag_1, lag_24, lag_168,
            temperature, rainfall, humidity
        )
        prediction = float(np.clip(model.predict(features)[0], 0, None))

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("📦 Predicted Demand", f"{prediction:.0f} orders")
        col_b.metric("📍 Area",             area_name)
        col_c.metric("🕐 Time",             f"{hour:02d}:00")
        col_d.metric("🌡️ Temperature",      f"{temperature}°C")

        # show demand range context
        st.markdown("**Context — how does this compare?**")
        area_stats = ts_df[ts_df['pincode'] == pincode]['demand']
        low  = area_stats.quantile(0.25)
        mid  = area_stats.median()
        high = area_stats.quantile(0.75)

        if prediction < low:
            st.info(f"🟢 Low demand period — predicted {prediction:.0f} vs area median {mid:.0f}")
        elif prediction < high:
            st.success(f"🟡 Normal demand period — predicted {prediction:.0f} vs area median {mid:.0f}")
        else:
            st.warning(f"🔴 High demand period — predicted {prediction:.0f} vs area median {mid:.0f}. Stock up!")

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — model performance
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model performance comparison")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("**Key metrics — LightGBM v2**")
        st.metric("MAPE",  "~24%",   delta="vs 30%+ baseline")
        st.metric("WAPE",  "~10%",   delta="industry standard")
        st.metric("R²",    "0.985",  delta="98.5% variance explained")
        st.metric("CV Std","1.42%",  delta="stable across folds")

    with col1:
        # model comparison bar chart
        if not results.empty:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            metrics  = ['mape','wape','rmse']
            titles   = ['MAPE %','WAPE %','RMSE']
            colors   = ['#378ADD','#1D9E75']

            for i, (metric, title) in enumerate(zip(metrics, titles)):
                vals   = results[metric].values
                models = results['model'].apply(
                    lambda x: 'Baseline' if 'Baseline' in x else 'LightGBM'
                ).values
                axes[i].bar(models, vals, color=colors, edgecolor='none')
                axes[i].set_title(title, fontsize=11)
                axes[i].set_ylabel(title)
                for j, v in enumerate(vals):
                    axes[i].text(j, v + 0.1, f'{v:.2f}', ha='center',
                                 fontsize=10, fontweight='500')

            plt.suptitle('Baseline vs LightGBM — metric comparison',
                         fontsize=13, fontweight='500')
            plt.tight_layout()
            st.pyplot(fig)

    # CV results table
    st.markdown("**5-Fold TimeSeriesSplit CV results**")
    cv_data = {
        'Fold'  : [1, 2, 3, 4, 5, 'Mean'],
        'MAPE %': [23.11, 25.15, 26.30, 26.12, 23.73, 24.88],
        'RMSE'  : [14.35, 16.17,  8.51,  8.09, 10.22, 11.47],
    }
    st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)

    # shap plot if exists
    import os
    if os.path.exists('notebooks/plot_shap_beeswarm.png'):
        st.markdown("**SHAP — feature impact on predictions**")
        st.image('notebooks/plot_shap_beeswarm.png', use_column_width=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 4 — demand trends
# ════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Historical demand trends")

    col1, col2 = st.columns(2)

    with col1:
        selected_area = st.selectbox(
            "Select area",
            selected['area_name'].tolist(),
            key="trend_area"
        )

    with col2:
        selected_product = st.selectbox(
            "Select product",
            ts_df['product_name'].unique()[:20].tolist(),
            key="trend_product"
        )

    pin = selected[selected['area_name'] == selected_area]['pincode'].values[0]

    trend_df = ts_df[
        (ts_df['pincode']      == pin) &
        (ts_df['product_name'] == selected_product)
    ].sort_values('datetime')

    if not trend_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # plot 1 — demand over time
        axes[0,0].plot(trend_df['datetime'], trend_df['demand'],
                       color='#378ADD', linewidth=1, alpha=0.8)
        axes[0,0].set_title(f'Demand over time — {selected_area}', fontsize=11)
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Demand')
        axes[0,0].tick_params(axis='x', rotation=30)

        # plot 2 — hourly pattern
        hourly = trend_df.groupby('order_hour_of_day')['demand'].mean()
        axes[0,1].bar(hourly.index, hourly.values,
                      color='#1D9E75', edgecolor='none')
        axes[0,1].set_title('Average demand by hour', fontsize=11)
        axes[0,1].set_xlabel('Hour of day')
        axes[0,1].set_ylabel('Avg demand')

        # plot 3 — day of week pattern
        day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        daily = trend_df.groupby('order_dow')['demand'].mean()
        colors = ['#E24B4A' if d in [5,6] else '#378ADD' for d in daily.index]
        axes[1,0].bar([day_names[d] for d in daily.index],
                      daily.values, color=colors, edgecolor='none')
        axes[1,0].set_title('Average demand by day of week', fontsize=11)
        axes[1,0].set_xlabel('Day')
        axes[1,0].set_ylabel('Avg demand')

        # plot 4 — demand distribution
        axes[1,1].hist(trend_df['demand'], bins=30,
                       color='#7F77DD', edgecolor='none', alpha=0.8)
        axes[1,1].axvline(trend_df['demand'].mean(), color='#E24B4A',
                          linestyle='--', label=f"Mean: {trend_df['demand'].mean():.1f}")
        axes[1,1].set_title('Demand distribution', fontsize=11)
        axes[1,1].set_xlabel('Demand')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()

        plt.suptitle(f'{selected_product} — {selected_area}',
                     fontsize=13, fontweight='500')
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.info("No data for this product + area combination. Try another.")

# ── footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Built with LightGBM + Streamlit · "
    "Data: Instacart + Open-Meteo + Govt of India Pincodes · "
    "**DemandPulse** — HyperLocal Demand Forecasting"
)
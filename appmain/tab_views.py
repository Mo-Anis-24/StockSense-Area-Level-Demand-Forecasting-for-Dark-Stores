# tab_views.py  — all 6 tab UI functions (map, pack, alerts, overview, simulator, roi)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import streamlit as st

from config import AREA_DENSITY, TODAY_NAME, TODAY_DATE
from predictor import (
    predict_pincode, predict_area_hourly,
    get_area_lags, build_features, predict_hourly
)


# ════════════════════════════════════════════════════════════════════
# TAB 1 — area demand map
# ════════════════════════════════════════════════════════════════════
def render_tab_map(all_areas_df, hour):
    st.subheader(f"🗺️ Hourly demand forecast — {TODAY_NAME} {hour:02d}:00")
    st.caption(
        f"Each number = predicted orders in the **{hour:02d}:00 – {(hour+1)%24:02d}:00** window only. "
        f"Change hour in sidebar to see other slots."
    )

    if all_areas_df.empty:
        st.error("No predictions generated — check pincode mismatch in sidebar")
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown(f"**📍 Area ranking — {hour:02d}:00**")
        for i, row in all_areas_df.reset_index(drop=True).iterrows():
            intensity = row['total_demand'] / (all_areas_df['total_demand'].max() or 1)
            bar       = '█' * int(intensity * 10)
            alert = ""
            if row['critical_items'] > 0:
                alert = f" 🔴{row['critical_items']}"
            elif row['low_items'] > 0:
                alert = f" 🟡{row['low_items']}"
            st.markdown(
                f"**{i+1}. {row['area_name']}**{alert}  \n"
                f"`{bar}` {row['total_demand']:.1f} orders/hr  \n"
                f"📦 {row['top_product_1'][:22]} ({row['top_demand_1']:.1f}/hr)"
            )
            st.markdown("---")

    with col1:
        m          = folium.Map(location=[23.0225, 72.5714],
                                zoom_start=11, tiles='CartoDB positron')
        max_demand = all_areas_df['total_demand'].max() or 1

        for _, row in all_areas_df.iterrows():
            demand    = row['total_demand']
            intensity = demand / max_demand
            radius    = 15 + intensity * 35
            r_val     = int(255 * intensity)
            g_val     = int(255 * (1 - intensity * 0.7))
            fill_color = f'#{r_val:02x}{g_val:02x}28'

            stock_badge = ""
            if row['critical_items'] > 0:
                stock_badge = f"<br>🔴 {row['critical_items']} critical items"
            elif row['low_items'] > 0:
                stock_badge = f"<br>🟡 {row['low_items']} low stock items"
            else:
                stock_badge = "<br>🟢 Stock OK"

            popup_html = f"""
            <div style='font-family:sans-serif;min-width:180px'>
              <b style='font-size:14px'>{row['area_name']}</b><br>
              <span style='color:#888'>Pincode: {row['pincode']}</span><br>
              <hr style='margin:6px 0'>
              <b>🕐 {hour:02d}:00 — {demand:.1f} orders/hr predicted</b>
              {stock_badge}<br>
              <hr style='margin:6px 0'>
              <b>Top products this hour:</b><br>
              📦 {row['top_product_1'][:28]} — {row['top_demand_1']:.1f}/hr<br>
              📦 {row['top_product_2'][:28]} — {row['top_demand_2']:.1f}/hr<br>
              📦 {row['top_product_3'][:28]} — {row['top_demand_3']:.1f}/hr
            </div>"""

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius, color='#333', weight=1,
                fill=True, fill_color=fill_color, fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"{row['area_name']} — {demand:.1f} orders/hr at {hour:02d}:00"
            ).add_to(m)

            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(
                    html=(f'<div style="font-size:10px;font-weight:600;'
                          f'color:#1a1a2e;white-space:nowrap;'
                          f'text-shadow:1px 1px 2px white">'
                          f'{row["area_name"]}<br>'
                          f'{demand:.1f}/hr @ {hour:02d}:00</div>'),
                    icon_size=(130, 30)
                )
            ).add_to(m)

        st_folium(m, width=750, height=500)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — pack priority list
# ════════════════════════════════════════════════════════════════════
def render_tab_pack(all_areas_df, hour, dow, week,
                    temperature, rainfall, humidity,
                    ts_df, stock_df, feature_cols,
                    model, model_p10, model_p90,
                    bias_factor, HAS_QUANTILE):

    st.subheader(f"📋 Pack Priority List — {hour:02d}:00")
    st.caption(
        f"Each row = predicted **hourly** orders at {hour:02d}:00 for that product.  "
        f"'Daily Stock' = total units on hand. 'Hourly Gap' = shortfall for this 1-hour slot only."
    )

    if all_areas_df.empty:
        st.error("No areas loaded — check pincode mismatch")
        return

    selected_area = st.selectbox(
        "Select area",
        all_areas_df['area_name'].tolist(),
        key="pack_area"
    )

    pin = all_areas_df[
        all_areas_df['area_name'] == selected_area
    ]['pincode'].values[0]

    with st.spinner(f"Loading {hour:02d}:00 pack list for {selected_area}..."):
        pack_df = predict_pincode(
            pin, hour, dow, week,
            temperature, rainfall, humidity,
            ts_df, stock_df, feature_cols, model,
            model_p10, model_p90, bias_factor,
            with_intervals=True
        )

    if pack_df.empty:
        st.warning(f"No products found for {selected_area}")
        return

    pack_df['priority'] = range(1, len(pack_df) + 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🕐 Hour",      f"{hour:02d}:00")
    c2.metric("🔴 Critical",  len(pack_df[pack_df['status']=='🔴 CRITICAL']))
    c3.metric("🟡 Low",       len(pack_df[pack_df['status']=='🟡 LOW']))
    c4.metric("🟢 OK",        len(pack_df[pack_df['status']=='🟢 OK']))
    c5.metric("📦 Total/hr",  f"{pack_df['predicted_demand'].sum():.1f} orders")

    st.markdown("---")

    hourly_df = predict_area_hourly(
        pin, dow, week, temperature, rainfall, humidity,
        ts_df, feature_cols, model, bias_factor
    )
    fig_h, ax_h = plt.subplots(figsize=(12, 3))
    bar_colors  = ['#E24B4A' if h == hour else '#378ADD'
                   for h in hourly_df['hour']]
    ax_h.bar(hourly_df['hour'], hourly_df['demand'],
             color=bar_colors, edgecolor='none', width=0.7)
    ax_h.axvline(x=hour, color='#E24B4A', linewidth=2,
                 linestyle='--', alpha=0.5)
    ax_h.set_title(
        f'{selected_area} — 24h hourly demand forecast '
        f'(red = selected hour {hour:02d}:00)',
        fontsize=11
    )
    ax_h.set_xlabel('Hour of day')
    ax_h.set_ylabel('Predicted orders / hour')
    ax_h.set_xticks(range(0, 24))
    plt.tight_layout()
    st.pyplot(fig_h)

    st.markdown("---")

    def color_row(val):
        if '🔴' in str(val): return 'background-color:#FAECE7'
        if '🟡' in str(val): return 'background-color:#FAEEDA'
        if '🔵' in str(val): return 'background-color:#E6F1FB'
        return 'background-color:#E1F5EE'

    if HAS_QUANTILE and 'range' in pack_df.columns:
        display = pack_df[[
            'priority','product_name','predicted_demand',
            'range','current_stock','stock_gap','status'
        ]].copy()
        display.columns = [
            'Priority','Product',f'Orders/hr at {hour:02d}:00',
            'Range/hr (p10–p90)','Daily Stock','Hourly Gap','Status'
        ]
    else:
        display = pack_df[[
            'priority','product_name','predicted_demand',
            'current_stock','stock_gap','status'
        ]].copy()
        display.columns = [
            'Priority','Product',f'Orders/hr at {hour:02d}:00',
            'Daily Stock','Hourly Gap','Status'
        ]

    st.dataframe(
        display.style.applymap(color_row, subset=['Status']),
        use_container_width=True,
        hide_index=True,
        height=480
    )

    csv = display.to_csv(index=False)
    st.download_button(
        f"⬇️ Download {hour:02d}:00 Pack List", csv,
        file_name=f"pack_{selected_area}_{TODAY_NAME}_{hour:02d}h.csv",
        mime="text/csv"
    )


# ════════════════════════════════════════════════════════════════════
# TAB 3 — stock alerts
# ════════════════════════════════════════════════════════════════════
def render_tab_alerts(all_areas_df, hour, dow, week,
                      temperature, rainfall, humidity,
                      ts_df, stock_df, feature_cols,
                      model, bias_factor):

    st.subheader(f"⚠️ Stock Alerts — {hour:02d}:00")
    st.markdown(
        f"Comparing **hourly demand at {hour:02d}:00** vs hourly stock allocation "
        f"(daily stock ÷ 24). Gap = units short for this 1-hour window.  "
        f"{TODAY_NAME}, {TODAY_DATE}."
    )

    if all_areas_df.empty:
        st.error("No areas loaded")
        return

    total_critical = int(all_areas_df['critical_items'].sum())
    total_low      = int(all_areas_df['low_items'].sum())
    sa1, sa2, sa3  = st.columns(3)
    sa1.metric("🔴 Critical across all areas", total_critical)
    sa2.metric("🟡 Low across all areas",      total_low)
    sa3.metric("🕐 Forecast hour",             f"{hour:02d}:00")

    st.markdown("---")

    for _, area_row in all_areas_df.iterrows():
        pin   = area_row['pincode']
        area  = area_row['area_name']
        total = area_row['total_demand']
        crit  = area_row['critical_items']
        low_c = area_row['low_items']

        badge = "🔴" if crit > 0 else ("🟡" if low_c > 0 else "🟢")

        with st.expander(
            f"{badge} {area} — {total:.1f} orders/hr at {hour:02d}:00 — "
            f"Top: {area_row['top_product_1'][:25]}"
        ):
            df = predict_pincode(
                pin, hour, dow, week,
                temperature, rainfall, humidity,
                ts_df, stock_df, feature_cols, model,
                bias_factor=bias_factor
            )

            if df.empty:
                st.info("No products found")
                continue

            critical = df[df['status'] == '🔴 CRITICAL']
            low      = df[df['status'] == '🟡 LOW']
            ok       = df[df['status'] == '🟢 OK']
            surplus  = df[df['status'] == '🔵 SURPLUS']

            if not critical.empty:
                st.error(f"🔴 {len(critical)} critically low — pack immediately")
                for _, r in critical.iterrows():
                    st.markdown(
                        f"**{r['product_name']}** — "
                        f"Predicted {r['predicted_demand']:.1f} orders/hr, "
                        f"Daily stock {r['current_stock']:.0f} units, "
                        f"**Hourly shortfall: {r['stock_gap']:.1f} units**"
                    )

            if not low.empty:
                st.warning(f"🟡 {len(low)} running low")
                for _, r in low.iterrows():
                    st.markdown(
                        f"**{r['product_name']}** — "
                        f"Predicted {r['predicted_demand']:.1f}/hr, "
                        f"Daily stock {r['current_stock']:.0f}"
                    )

            if not surplus.empty:
                st.info(f"🔵 {len(surplus)} overstocked")

            if critical.empty and low.empty:
                st.success(f"✅ {len(ok)} products adequately stocked for {hour:02d}:00")


# ════════════════════════════════════════════════════════════════════
# TAB 4 — demand overview
# ════════════════════════════════════════════════════════════════════
def render_tab_overview(all_areas_df, hour, dow, week,
                        temperature, rainfall, humidity,
                        ts_df, selected, feature_cols,
                        model, bias_factor):

    st.subheader("📊 Demand Overview")

    if all_areas_df.empty:
        st.error("No predictions available")
        return

    st.markdown(f"### 🕐 This hour — {hour:02d}:00")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors  = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(all_areas_df)))
        ax.barh(
            all_areas_df['area_name'][::-1],
            all_areas_df['total_demand'][::-1],
            color=colors, edgecolor='none'
        )
        ax.set_title(f'Orders per area at {hour:02d}:00 (orders/hr)', fontsize=12)
        ax.set_xlabel('Predicted orders / hour')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown(f"**Top products across all areas at {hour:02d}:00**")
        summary_tbl = all_areas_df[[
            'area_name','total_demand',
            'top_product_1','top_demand_1',
            'critical_items','low_items'
        ]].copy()
        summary_tbl.columns = [
            'Area', f'Orders/hr at {hour:02d}:00',
            'Top Product','Top Demand/hr',
            '🔴 Critical','🟡 Low'
        ]
        summary_tbl.index = range(1, len(summary_tbl)+1)
        st.dataframe(summary_tbl, use_container_width=True, height=380)

    st.markdown("---")
    st.markdown("### 📅 Full day — all hours (orders/hr per area)")

    heatmap_data = []
    for _, area_row in selected.iterrows():
        pin  = area_row['pincode']
        area = area_row['area_name']
        if pin not in ts_df['pincode'].values:
            continue

        row_data               = {'area': area}
        lag_1, lag_24, lag_168 = get_area_lags(pin, ts_df)
        density                = AREA_DENSITY.get(pin, 0.10)

        for h in range(24):
            f = build_features(h, dow, week, lag_1, lag_24, lag_168,
                               temperature, rainfall, humidity,
                               area_density=density, feature_cols=feature_cols)
            p = predict_hourly(model.predict(f)[0], bias_factor)
            row_data[f'{h:02d}'] = round(p, 1)

        heatmap_data.append(row_data)

    hm_df = pd.DataFrame(heatmap_data).set_index('area')

    fig2, ax2 = plt.subplots(figsize=(16, 5))
    sns.heatmap(hm_df, cmap='YlOrRd', linewidths=0.3,
                linecolor='white', ax=ax2, fmt='.1f',
                annot=True, annot_kws={'size': 7})
    ax2.add_patch(plt.Rectangle(
        (hour, 0), 1, len(hm_df),
        fill=False, edgecolor='#1a1a2e', lw=2.5
    ))
    ax2.set_title(
        f'Hourly demand heatmap (orders/hr) — {TODAY_NAME} '
        f'(black box = selected hour {hour:02d}:00)',
        fontsize=12
    )
    ax2.set_xlabel('Hour of day (00 to 23)')
    ax2.set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")
    st.markdown("### 📈 Hourly demand curve — select area")

    area_sel = st.selectbox(
        "Select area for hourly curve",
        selected['area_name'].tolist(),
        key="hourly_area_sel"
    )
    pin_sel = selected[selected['area_name'] == area_sel]['pincode'].values[0]

    if pin_sel in ts_df['pincode'].values:
        hourly_area = predict_area_hourly(
            pin_sel, dow, week, temperature, rainfall, humidity,
            ts_df, feature_cols, model, bias_factor
        )

        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.fill_between(hourly_area['hour'], hourly_area['demand'],
                         alpha=0.2, color='#378ADD')
        ax3.plot(hourly_area['hour'], hourly_area['demand'],
                 color='#378ADD', linewidth=2, marker='o', markersize=4)
        ax3.axvline(x=hour, color='#E24B4A', linewidth=2,
                    linestyle='--', label=f'Now — {hour:02d}:00')
        now_demand = hourly_area.loc[hourly_area['hour']==hour, 'demand'].values
        if len(now_demand):
            ax3.scatter([hour], [now_demand[0]],
                        color='#E24B4A', s=80, zorder=5)
        ax3.set_title(
            f'{area_sel} — hourly demand forecast (orders/hr) — {TODAY_NAME}',
            fontsize=12
        )
        ax3.set_xlabel('Hour of day')
        ax3.set_ylabel('Predicted orders / hour')
        ax3.set_xticks(range(0, 24))
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3)


# ════════════════════════════════════════════════════════════════════
# TAB 5 — what-if simulator
# ════════════════════════════════════════════════════════════════════
def render_tab_simulator(hour, dow, week, weather,
                         ts_df, selected, feature_cols,
                         model, model_p10, model_p90,
                         bias_factor, HAS_QUANTILE):

    st.subheader("🎮 What-if Scenario Simulator")
    st.markdown(
        f"Showing impact on **hourly demand at {hour:02d}:00** across all areas. "
        f"Adjust conditions and watch demand change instantly."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🌤️ Change weather**")
        sim_temp     = st.slider("Temperature °C", 10, 48,
                                 int(weather['temperature']), key="sim_temp")
        sim_rain     = st.slider("Rainfall mm",     0,  50,
                                 int(weather['rainfall']),    key="sim_rain")
        sim_humidity = st.slider("Humidity %",      10, 100,
                                 int(weather['humidity']),    key="sim_hum")

    with col2:
        st.markdown("**🎉 Change events**")
        sim_festival = st.toggle("Festival today (Diwali / Navratri)", value=False)
        sim_cricket  = st.toggle("IPL match tonight",                  value=False)
        sim_holiday  = st.toggle("Public holiday",                     value=False)
        sim_weekend  = st.toggle("Weekend override", value=dow in [5, 6])

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{sim_temp}°C",
              delta=f"{sim_temp - int(weather['temperature']):+.0f}°C")
    c2.metric("Rainfall",    f"{sim_rain} mm",
              delta=f"{sim_rain - int(weather['rainfall']):+.0f} mm")
    c3.metric("Festival",    "Yes" if sim_festival else "No")
    c4.metric("Cricket",     "Yes" if sim_cricket  else "No")

    st.markdown("---")

    sim_dow          = 5 if sim_weekend else dow
    event_multiplier = 1.0
    if sim_festival: event_multiplier *= 1.35
    if sim_cricket : event_multiplier *= 1.20
    if sim_holiday : event_multiplier *= 1.15

    areas_list    = []
    base_totals   = []
    sim_totals    = []
    base_p10_list = []
    base_p90_list = []
    sim_p10_list  = []
    sim_p90_list  = []

    for _, area_row in selected.iterrows():
        pin  = area_row['pincode']
        area = area_row['area_name']
        if pin not in ts_df['pincode'].values:
            continue

        lag_1, lag_24, lag_168 = get_area_lags(pin, ts_df)
        density = AREA_DENSITY.get(pin, 0.10)

        f_base = build_features(
            hour, dow, week, lag_1, lag_24, lag_168,
            weather['temperature'], weather['rainfall'], weather['humidity'],
            area_density=density, feature_cols=feature_cols
        )
        f_sim = build_features(
            hour, sim_dow, week, lag_1, lag_24, lag_168,
            sim_temp, sim_rain, sim_humidity,
            area_density=density, feature_cols=feature_cols
        )

        base_mean = predict_hourly(model.predict(f_base)[0], bias_factor)
        sim_mean  = predict_hourly(model.predict(f_sim)[0], bias_factor) * event_multiplier

        areas_list.append(area)
        base_totals.append(round(base_mean, 1))
        sim_totals.append(round(sim_mean,   1))

        if HAS_QUANTILE:
            bp10 = predict_hourly(model_p10.predict(f_base)[0], bias_factor)
            bp90 = predict_hourly(model_p90.predict(f_base)[0], bias_factor)
            sp10 = predict_hourly(model_p10.predict(f_sim)[0], bias_factor) * event_multiplier
            sp90 = predict_hourly(model_p90.predict(f_sim)[0], bias_factor) * event_multiplier
            base_p10_list.append(round(bp10, 1))
            base_p90_list.append(round(bp90, 1))
            sim_p10_list.append(round(sp10,  1))
            sim_p90_list.append(round(sp90,  1))

    if not areas_list:
        st.error("No area predictions available")
        return

    deltas  = [s - b for s, b in zip(sim_totals, base_totals)]
    pct_chg = [(d/b*100) if b > 0 else 0
               for d, b in zip(deltas, base_totals)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x     = range(len(areas_list))
    width = 0.35

    axes[0].bar([i-width/2 for i in x], base_totals,
                width, label=f'Current {hour:02d}:00', color='#378ADD', alpha=0.8)
    axes[0].bar([i+width/2 for i in x], sim_totals,
                width, label='Scenario',               color='#E24B4A', alpha=0.8)
    axes[0].set_title(f'Current vs scenario — {hour:02d}:00 (orders/hr)', fontsize=12)
    axes[0].set_ylabel('Predicted orders / hour')
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(areas_list, rotation=30, ha='right', fontsize=9)
    axes[0].legend()

    bar_colors = ['#1D9E75' if d >= 0 else '#E24B4A' for d in pct_chg]
    axes[1].bar(list(x), pct_chg, color=bar_colors, edgecolor='none')
    axes[1].axhline(y=0, color='#888', linewidth=0.8)
    axes[1].set_title('Demand change % vs baseline', fontsize=12)
    axes[1].set_ylabel('Change %')
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(areas_list, rotation=30, ha='right', fontsize=9)

    y_span = (max(pct_chg) - min(pct_chg)) or 1
    pad    = y_span * 0.04
    for i, v in enumerate(pct_chg):
        axes[1].text(
            i,
            v + (pad if v >= 0 else -pad),
            f'{v:+.1f}%',
            ha='center',
            va='bottom' if v >= 0 else 'top',
            fontsize=9, fontweight='500'
        )
    ymin, ymax = axes[1].get_ylim()
    axes[1].set_ylim(ymin - y_span * 0.12, ymax + y_span * 0.12)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Detailed breakdown**")
    sim_rows = []
    for i, area in enumerate(areas_list):
        entry = {
            'Area'                        : area,
            f'Current/hr ({hour:02d}:00)' : f"{base_totals[i]:.1f}",
            'Scenario/hr'                 : f"{sim_totals[i]:.1f}",
            'Change'                      : f"{deltas[i]:+.1f}",
            'Change %'                    : f"{pct_chg[i]:+.1f}%",
        }
        if HAS_QUANTILE and base_p10_list:
            entry['Current range/hr']  = f"{base_p10_list[i]:.1f} – {base_p90_list[i]:.1f}"
            entry['Scenario range/hr'] = f"{sim_p10_list[i]:.1f} – {sim_p90_list[i]:.1f}"
        sim_rows.append(entry)

    st.dataframe(pd.DataFrame(sim_rows),
                 use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# TAB 6 — ROI calculator
# ════════════════════════════════════════════════════════════════════
def render_tab_roi():
    st.subheader("💰 ROI & Wastage Savings Calculator")
    st.markdown(
        "Converts MAPE improvement into actual rupee savings vs naive baseline."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📦 Product economics**")
        avg_product_cost = st.number_input("Average product cost per unit (₹)", value=45,  step=5)
        avg_order_value  = st.number_input("Average order value (₹)",           value=380, step=10)
        avg_daily_orders = st.number_input("Average daily orders per dark store",value=500, step=50)
        num_skus         = st.number_input("Number of active SKUs",              value=50,  step=5)

    with col2:
        st.markdown("**💸 Cost parameters**")
        overstock_cost_pct = st.slider("Overstock cost % of product cost", 1, 30, 8,
                                       help="Wastage + storage + markdown")
        stockout_cost_pct  = st.slider("Stockout cost % of order value",   1, 30, 12,
                                       help="Lost revenue + customer loss")
        num_dark_stores    = st.number_input("Number of dark stores", value=1, step=1)

    st.markdown("---")

    BASELINE_MAPE = 133.91
    MODEL_MAPE    = 30.58
    IMPROVEMENT   = BASELINE_MAPE - MODEL_MAPE

    st.markdown("**📊 Model performance**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline MAPE", f"{BASELINE_MAPE:.1f}%")
    m2.metric("LightGBM MAPE", f"{MODEL_MAPE:.2f}%",
              delta=f"-{IMPROVEMENT:.2f}% improvement")
    m3.metric("WAPE", "20.86%")

    st.markdown("---")

    avg_units_per_sku     = avg_daily_orders / num_skus
    baseline_overstock    = avg_units_per_sku * (BASELINE_MAPE / 100)
    model_overstock       = avg_units_per_sku * (MODEL_MAPE    / 100)
    overstock_units_saved = (baseline_overstock - model_overstock) * num_skus
    overstock_cost_saved  = overstock_units_saved * avg_product_cost * overstock_cost_pct / 100
    baseline_stockouts    = avg_daily_orders * (BASELINE_MAPE / 100)
    model_stockouts       = avg_daily_orders * (MODEL_MAPE    / 100)
    stockout_saved        = baseline_stockouts - model_stockouts
    stockout_rev_saved    = stockout_saved * avg_order_value * stockout_cost_pct / 100
    daily_savings         = overstock_cost_saved + stockout_rev_saved
    monthly_savings       = daily_savings * 30 * num_dark_stores
    yearly_savings        = monthly_savings * 12

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Daily / store",   f"₹{daily_savings:,.0f}")
    r2.metric("Monthly / store", f"₹{daily_savings*30:,.0f}")
    r3.metric("Monthly total",   f"₹{monthly_savings:,.0f}",
              delta=f"{num_dark_stores} store(s)")
    r4.metric("Annual total",    f"₹{yearly_savings:,.0f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Overstock reduction**")
        st.markdown(f"- Baseline: **{baseline_overstock*num_skus:.0f} units/day** wasted")
        st.markdown(f"- Model:    **{model_overstock*num_skus:.0f} units/day** wasted")
        st.markdown(f"- Saved:    **{overstock_units_saved:.0f} units/day**")
        st.markdown(f"- Value:    **₹{overstock_cost_saved:,.0f}/day**")
    with col_b:
        st.markdown("**Stockout recovery**")
        st.markdown(f"- Baseline: **{baseline_stockouts:.0f} stockouts/day**")
        st.markdown(f"- Model:    **{model_stockouts:.0f} stockouts/day**")
        st.markdown(f"- Recovered:**{stockout_saved:.0f} orders/day**")
        st.markdown(f"- Value:    **₹{stockout_rev_saved:,.0f}/day**")

    st.markdown("---")

    months      = list(range(1, 13))
    cum_savings = [monthly_savings * m for m in months]
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(month_names, cum_savings, color='#1D9E75', alpha=0.8, edgecolor='none')
    ax.plot(month_names, cum_savings, color='#085041',
            linewidth=2, marker='o', markersize=5)
    for i, v in enumerate(cum_savings):
        ax.text(i, v + monthly_savings * 0.02,
                f'₹{v/1e5:.1f}L', ha='center', fontsize=9, fontweight='500')
    ax.set_title(
        f'Cumulative savings — {num_dark_stores} dark store(s) — 12 months',
        fontsize=12
    )
    ax.set_ylabel('Cumulative savings (₹)')
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'₹{x/1e5:.1f}L')
    )
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.success(
        f"**Summary:**  \n"
        f"DemandPulse reduces forecasting error from {BASELINE_MAPE:.0f}% to "
        f"{MODEL_MAPE:.2f}% MAPE, saving approximately "
        f"₹{monthly_savings:,.0f}/month across {num_dark_stores} dark store(s) "
        f"through reduced overstock wastage and stockout revenue recovery."
    )
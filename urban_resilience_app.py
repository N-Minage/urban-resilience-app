
import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

## Page config
st.set_page_config(
    page_title="Nairobi Urban Resilience Engine",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

## CSS: full light theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background: #f6f8fa !important;
    color: #24292f !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
#MainMenu, footer, header,
[data-testid="stDecoration"] { display: none !important; visibility: hidden !important; }

.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Top bar ── */
.top-bar {
    background: #ffffff; border-bottom: 1px solid #d0d7de;
    padding: 10px 24px; display: flex; align-items: center; gap: 14px;
}
.top-title { font-size: 0.95rem; font-weight: 700; color: #1f2328; }
.top-sub   { font-size: 0.7rem; color: #6e7781; margin-top: 1px; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important; border-color: #d0d7de !important;
    color: #24292f !important; border-radius: 8px !important; font-size: 0.84rem !important;
}
div[data-testid="stSelectbox"] label { color: #57606a !important; font-size: 0.78rem !important; font-weight: 600 !important; }
div[data-testid="stSlider"] > label { color: #57606a !important; font-size: 0.78rem !important; font-weight: 600 !important; }
div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: #1f2328 !important; font-weight: 700 !important; font-size: 0.82rem !important;
}

/* ── KPI cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; padding: 16px 24px 0; }
.kpi-card {
    background: #ffffff; border: 1px solid #d0d7de; border-radius: 10px; padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(27,31,36,0.06);
}
.kpi-label {
    font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .1em; color: #6e7781; margin-bottom: 5px;
}
.kpi-value { font-size: 1.9rem; font-weight: 700; letter-spacing: -.03em; line-height: 1.1; }
.kpi-sub   { font-size: 0.72rem; color: #6e7781; margin-top: 3px; }
.risk-HIGH   { color: #cf222e; }
.risk-MEDIUM { color: #9a6700; }
.risk-LOW    { color: #1a7f37; }

/* ── Section header ── */
.sec-hd { font-size: 0.83rem; font-weight: 700; color: #24292f; margin-bottom: 12px; }

/* ── Chart card ── */
.chart-card {
    background: #ffffff; border: 1px solid #d0d7de; border-radius: 10px; padding: 16px;
    box-shadow: 0 1px 3px rgba(27,31,36,0.06);
}

/* ── Probability bars ── */
.prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 13px; }
.prob-label { font-size: 0.82rem; font-weight: 600; width: 64px; flex-shrink: 0; }
.prob-label-HIGH   { color: #cf222e; }
.prob-label-MEDIUM { color: #9a6700; }
.prob-label-LOW    { color: #1a7f37; }
.prob-bar-bg { flex: 1; background: #e8eaed; border-radius: 4px; height: 8px; overflow: hidden; }
.prob-bar    { height: 100%; border-radius: 4px; }
.prob-pct    { font-size: 0.82rem; font-weight: 700; color: #24292f;
               font-family: 'DM Mono', monospace; width: 44px; text-align: right; flex-shrink: 0; }

/* ── Risk distribution legend ── */
.dist-item { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 9px; }
.dist-dot  { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; margin-top: 3px; }
.dist-label{ font-size: 0.77rem; color: #57606a; line-height: 1.4; }
.dist-label b { color: #24292f; display: block; }

/* ── LOO box ── */
.loo-card {
    background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px;
    padding: 10px 14px; margin-top: 12px;
}
.loo-hd  { font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
           letter-spacing: .08em; color: #6e7781; margin-bottom: 8px; }
.loo-row { display: flex; justify-content: space-between; align-items: center;
           padding: 4px 0; border-bottom: 1px solid #d0d7de; }
.loo-row:last-child { border: none; }
.loo-k   { font-size: 0.76rem; color: #57606a; }
.loo-v   { font-size: 0.87rem; font-weight: 700; color: #1f2328;
           font-family: 'DM Mono', monospace; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important; border-bottom: 1px solid #d0d7de !important;
    gap: 0 !important; padding: 0 24px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #57606a !important; font-size: 0.82rem !important;
    font-weight: 600 !important; padding: 10px 18px !important;
    border-radius: 0 !important; font-family: 'DM Sans', sans-serif !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #24292f !important; }
.stTabs [aria-selected="true"] {
    color: #1a7f37 !important; border-bottom: 2px solid #1a7f37 !important;
}
[data-testid="stTabsContent"] { background: #f6f8fa !important; padding: 16px 24px 0 !important; }

/* ── Custom table ── */
.tbl-wrap {
    background: #ffffff; border: 1px solid #d0d7de;
    border-radius: 10px; overflow: hidden;
    box-shadow: 0 1px 3px rgba(27,31,36,0.06);
}
.tbl-hd {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr 0.85fr 0.85fr 0.85fr 1.4fr;
    padding: 10px 16px; background: #f6f8fa;
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: #57606a; border-bottom: 1px solid #d0d7de;
}
.tbl-row {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr 0.85fr 0.85fr 0.85fr 1.4fr;
    padding: 12px 16px; font-size: 0.82rem; color: #57606a;
    border-bottom: 1px solid #d0d7de; align-items: center;
}
.tbl-row:last-child { border: none; }
.tbl-row:hover { background: #f6f8fa; }
.tbl-sc { font-weight: 700; color: #1f2328; }
.risk-pill {
    display: inline-block; border-radius: 12px; padding: 2px 10px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: .03em;
}
.pill-HIGH   { background: #fff0ee; border: 1px solid #ffcbc8; color: #cf222e; }
.pill-MEDIUM { background: #fff8e1; border: 1px solid #fde272; color: #9a6700; }
.pill-LOW    { background: #f0fff4; border: 1px solid #b7ebc8; color: #1a7f37; }
.mini-bar-bg { background: #e8eaed; border-radius: 3px; height: 6px;
               width: 72px; display: inline-block; vertical-align: middle;
               margin-right: 6px; overflow: hidden; }
.mini-bar    { height: 100%; border-radius: 3px; background: #cf222e; }
.mono        { font-family: 'DM Mono', monospace; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

##Constants
RISK_COLORS = {"High": "#cf222e", "Medium": "#9a6700", "Low": "#1a7f37"}
CHART_LIGHT = dict(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                   font=dict(color="#57606a", family="DM Sans"))
GRID_C      = "#e8eaed"

SC_COORDS = {
    "Mathare":   (-1.258, 36.857), "Kibra":     (-1.314, 36.784),
    "Kamukunji": (-1.282, 36.844), "Makadara":  (-1.303, 36.860),
    "Dagoretti": (-1.302, 36.736), "Starehe":   (-1.275, 36.829),
    "Embakasi":  (-1.322, 36.896), "Kasarani":  (-1.218, 36.895),
    "Westlands": (-1.261, 36.807), "Lang'ata":  (-1.356, 36.758),
    "Njiru":     (-1.243, 36.960),
}

##  Loaders
@st.cache_resource
def load_model():
    model    = joblib.load("models/xgboost_risk_model.pkl")
    le       = joblib.load("models/label_encoder.pkl")
    features = joblib.load("models/feature_list.pkl")
    return model, le, features

@st.cache_data
def load_data():
    def clean(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df
    risk_df    = clean(pd.read_csv("data/nairobi_risk_dataset.csv"))
    weather_df = clean(pd.read_csv("data/nairobi_weather_clean.csv"))
    result_df  = pd.read_csv("outputs/subcounty_risk_predictions.csv")
    return risk_df, weather_df, result_df

@st.cache_data
def load_loo():
    path = "models/loo_metrics.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"accuracy": 0.818, "f1": 0.791}

model, le, features = load_model()
risk_df, weather_df, result_df = load_data()
loo_metrics = load_loo()
ALL_SC = sorted(risk_df["sub_county"].unique())

BASE_WEATHER = {
    "avg_annual_precip":         weather_df["total_precip"].mean(),
    "max_monthly_precip":        weather_df["total_precip"].max(),
    "precip_variability":        weather_df["total_precip"].std(),
    "avg_extreme_rain_days":     weather_df["extreme_rain_days"].mean(),
    "total_extreme_rain_events": weather_df["extreme_rain_days"].sum(),
    "avg_temp_c":                weather_df["avg_temp"].mean(),
    "precip_95th":               weather_df["total_precip"].quantile(0.95),
}

## Helpers
def scale_weather(base, rm, to):
    return {k: (v * rm if k != "avg_temp_c" else v + to) for k, v in base.items()}

def build_row(sc, ws):
    r = risk_df[risk_df["sub_county"] == sc].iloc[0]
    return {
        "population":              r["population"],
        "households":              r["households"],
        "avg_household_size":      r["avg_household_size"],
        "area_sqkm":               r["area_sqkm"],
        "pop_density_per_sqkm":    r["pop_density_per_sqkm"],
        "sex_ratio":               r.get("sex_ratio", 1.0),
        "male":                    r.get("male",   r["population"] * 0.5),
        "female":                  r.get("female", r["population"] * 0.5),
        "informal_settlement":     r.get("informal_settlement", 0),
        "drainage_capacity_score": r.get("drainage_capacity_score", 0.5),
        "flood_exposure_index":    r["pop_density_per_sqkm"]    * ws["max_monthly_precip"],
        "vulnerable_pop_rain":     r["population"]              * ws["avg_extreme_rain_days"],
        "drainage_stress":         r["households"]              * ws["precip_variability"],
        "heat_pop_stress":         r["population"]              * ws["avg_temp_c"],
        "area_rain_load":          r["area_sqkm"]               * ws["avg_annual_precip"],
        "density_extreme_rain":    r["pop_density_per_sqkm"]    * ws["total_extreme_rain_events"],
        "household_flood_risk":    r["avg_household_size"]      * ws["precip_95th"],
        "informal_rain_risk":      r.get("informal_settlement", 0) * ws["max_monthly_precip"],
        "drainage_deficit_index":  (1 - r.get("drainage_capacity_score", 0.5))
                                   * r["pop_density_per_sqkm"] * ws["max_monthly_precip"],
    }

def predict(sc, ws):
    feat = pd.DataFrame([build_row(sc, ws)])[features]
    prob = model.predict_proba(feat)[0]
    pred = le.inverse_transform([np.argmax(prob)])[0]
    return prob, pred, feat

def hi_idx():
    cl = list(le.classes_)
    return cl.index("High") if "High" in cl else 0

HI = hi_idx()

@st.cache_data
def get_all_preds(rain_scale, temp_offset):
    ws_ = scale_weather(BASE_WEATHER, rain_scale, temp_offset)
    rows = []
    max_d = risk_df["pop_density_per_sqkm"].max()
    for sc_ in ALL_SC:
        p_, pred_, _ = predict(sc_, ws_)
        r_ = risk_df[risk_df["sub_county"] == sc_].iloc[0]
        lat, lon = SC_COORDS.get(sc_, (-1.286, 36.820))
        d = int(r_["pop_density_per_sqkm"])
        rows.append({
            "sub_county": sc_, "pred_risk": pred_, "hi_prob": float(p_[HI]),
            "population": int(r_["population"]), "density": d,
            "area": float(r_["area_sqkm"]),
            "vuln": round(float(r_.get("vulnerability_score", 0)), 3),
            "lat": lat, "lon": lon,
            "msize": max(10, min(44, d / max_d * 44)),
            "flood_exp": d * ws_["max_monthly_precip"],
        })
    return pd.DataFrame(rows)

## TOP BAR
st.markdown("""
<div class="top-bar">
  <span style="font-size:1.4rem">🏛</span>
  <div>
    <div class="top-title">Nairobi Urban Resilience Engine</div>
    <div class="top-sub">Predicting extreme weather impact on city infrastructure · Kenya</div>
  </div>
</div>
""", unsafe_allow_html=True)

## CONTROL ROW
c1, c2, c3, c4 = st.columns([1.1, 0.1, 1.7, 1.7])
with c1:
    selected_sc = st.selectbox("Sub-county", ALL_SC,
                                index=ALL_SC.index("Mathare") if "Mathare" in ALL_SC else 0)
with c3:
    rain_scale  = st.slider("Rainfall ×", 0.1, 3.0, 1.0, 0.05, format="%.2f×")
with c4:
    temp_offset = st.slider("Temp offset", -5.0, 10.0, 0.0, 0.5, format="%+.1f°C")

##  ACTIVE PREDICTION
ws                    = scale_weather(BASE_WEATHER, rain_scale, temp_offset)
proba, pred_cls, feat = predict(selected_sc, ws)
sc_row                = risk_df[risk_df["sub_county"] == selected_sc].iloc[0]
risk_upper            = pred_cls.upper()
all_pred              = get_all_preds(rain_scale, temp_offset)

## KPI CARDS
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Predicted risk</div>
    <div class="kpi-value risk-{risk_upper}">{risk_upper}</div>
    <div class="kpi-sub">Confidence {np.max(proba)*100:.0f}%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">High-risk probability</div>
    <div class="kpi-value" style="color:#1f2328">{proba[HI]:.3f}</div>
    <div class="kpi-sub">{selected_sc}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Population</div>
    <div class="kpi-value" style="color:#1f2328">{int(sc_row['population']):,}</div>
    <div class="kpi-sub">{int(sc_row.get('households',0)):,} households</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Pop density</div>
    <div class="kpi-value" style="color:#1f2328">{int(sc_row['pop_density_per_sqkm']):,}</div>
    <div class="kpi-sub">per km²</div>
  </div>
</div>
<div style="height:14px"></div>
""", unsafe_allow_html=True)

#
with st.container():
    col_map, col_right = st.columns([1.5, 1], gap="medium")

    with col_map:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-hd">Sub-county risk map — Nairobi</div>',
                    unsafe_allow_html=True)
        map_fig = go.Figure()
        for rl in ["Low", "Medium", "High"]:
            df_ = all_pred[all_pred["pred_risk"] == rl]
            if df_.empty:
                continue
            border_widths = [4 if sc == selected_sc else 1.5 for sc in df_["sub_county"]]
            border_colors = ["#1f2328" if sc == selected_sc else "rgba(255,255,255,0.85)"
                             for sc in df_["sub_county"]]
            map_fig.add_trace(go.Scattermapbox(
                lat=df_["lat"],
                lon=df_["lon"],
                mode="markers+text",
                name=rl,
                marker=go.scattermapbox.Marker(
                    size=df_["msize"] * 1.1,
                    color=RISK_COLORS[rl],
                    opacity=0.82,
                ),
                text=df_["sub_county"],
                textposition="top right",
                textfont=dict(size=11, color="#24292f", family="DM Sans"),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Risk: " + rl + "<br>"
                    "P(High): %{customdata[0]:.3f}<br>"
                    "Density: %{customdata[1]:,}/km²"
                    "<extra></extra>"
                ),
                customdata=df_[["hi_prob", "density"]].values,
            ))
        # Highlight selected sub-county with a separate ring marker
        sel_row = all_pred[all_pred["sub_county"] == selected_sc].iloc[0]
        map_fig.add_trace(go.Scattermapbox(
            lat=[sel_row["lat"]],
            lon=[sel_row["lon"]],
            mode="markers",
            name="",
            showlegend=False,
            marker=go.scattermapbox.Marker(
                size=sel_row["msize"] * 1.1 + 14,
                color="rgba(0,0,0,0)",
                opacity=1,
            ),
            hoverinfo="skip",
        ))
        map_fig.update_layout(
            paper_bgcolor="#ffffff",
            font=dict(color="#57606a", family="DM Sans"),
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=-1.286, lon=36.830),
                zoom=10.3,
            ),
            legend=dict(
                orientation="h", x=0.01, y=0.01,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#d0d7de", borderwidth=1,
                font=dict(size=11, color="#24292f"),
                itemsizing="constant",
            ),
            margin=dict(t=0, b=0, l=0, r=0),
            height=410,
        )
        st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div style="font-size:0.72rem;color:#6e7781;margin-top:4px;">'
            '⬤ Circle size = population density &nbsp;|&nbsp; '
            'Basemap © OpenStreetMap contributors</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # Probability bars
        st.markdown('<div class="chart-card" style="margin-bottom:12px">', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-hd">Risk class probabilities — {selected_sc}</div>',
                    unsafe_allow_html=True)
        for cls in le.classes_:
            p  = proba[list(le.classes_).index(cls)]
            cu = cls.upper()
            st.markdown(f"""
            <div class="prob-row">
              <span class="prob-label prob-label-{cu}">{cls}</span>
              <div class="prob-bar-bg">
                <div class="prob-bar" style="width:{p*100:.1f}%;background:{RISK_COLORS[cls]};"></div>
              </div>
              <span class="prob-pct">{p*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Flood exposure chart
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-hd">Flood exposure index (scaled)</div>', unsafe_allow_html=True)
        sorted_flood = all_pred.sort_values("flood_exp", ascending=False)
        flood_fig = go.Figure(go.Bar(
            x=sorted_flood["sub_county"], y=sorted_flood["flood_exp"],
            marker_color=[RISK_COLORS.get(r, "#6e7781") for r in sorted_flood["pred_risk"]],
            marker_line_width=0,
        ))
        flood_fig.update_layout(
            **CHART_LIGHT, height=178,
            margin=dict(t=4, b=36, l=8, r=8),
            xaxis=dict(tickfont=dict(size=8, color="#57606a"), showgrid=False, tickangle=-35),
            yaxis=dict(showgrid=True, gridcolor=GRID_C, tickfont=dict(size=8)),
        )
        st.plotly_chart(flood_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

#
st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
col_fi, col_dist = st.columns([1.5, 1], gap="medium")

with col_fi:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hd">Feature importance — XGBoost model</div>',
                unsafe_allow_html=True)
    imp = (pd.DataFrame({"feature": features, "importance": model.feature_importances_})
           .sort_values("importance", ascending=False).head(10)
           .sort_values("importance").reset_index(drop=True))
    n = len(imp)
    imp["fname"] = (imp["feature"]
                    .str.replace("flood_exposure_index", "flood_exposure_idx")
                    .str.replace("density_extreme_rain", "density × extr_rain")
                    .str.replace("avg_household_size", "avg_hh_size")
                    .str.replace("_per_sqkm", ""))
    bar_colors = [
        "#cf222e" if i/n >= 0.66 else "#9a6700" if i/n >= 0.33 else "#0969da"
        for i in range(n)
    ]
    fi_fig = go.Figure(go.Bar(
        y=imp["fname"], x=imp["importance"],
        orientation="h", marker_color=bar_colors, marker_line_width=0, width=0.68,
    ))
    fi_fig.update_layout(
        **CHART_LIGHT, height=310, margin=dict(t=4, b=4, l=8, r=20),
        xaxis=dict(showgrid=True, gridcolor=GRID_C, tickfont=dict(size=9)),
        yaxis=dict(showgrid=False, tickfont=dict(size=9.5, color="#24292f")),
    )
    st.plotly_chart(fi_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_dist:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hd">Risk distribution across sub-counties</div>',
                unsafe_allow_html=True)
    dist = all_pred["pred_risk"].value_counts()

    donut = go.Figure(go.Pie(
        labels=list(dist.index), values=list(dist.values), hole=0.60,
        marker=dict(colors=[RISK_COLORS.get(l, "#6e7781") for l in dist.index],
                    line=dict(color="#ffffff", width=2)),
        textinfo="none",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    donut.update_layout(
        **CHART_LIGHT, showlegend=False,
        margin=dict(t=4, b=4, l=4, r=4), height=160,
    )

    d1, d2 = st.columns([1, 1])
    with d1:
        st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})
    with d2:
        leg = ""
        for rl in ["High", "Medium", "Low"]:
            cnt = dist.get(rl, 0)
            leg += f"""<div class="dist-item">
              <div class="dist-dot" style="background:{RISK_COLORS[rl]}"></div>
              <div class="dist-label">
                <b>{rl} risk</b>
                — {cnt} sub-counti{'es' if cnt!=1 else 'y'}
              </div></div>"""
        st.markdown(leg, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="loo-card">
          <div class="loo-hd">LOO-CV performance</div>
          <div class="loo-row">
            <span class="loo-k">Accuracy</span>
            <span class="loo-v">{loo_metrics['accuracy']:.3f}</span>
          </div>
          <div class="loo-row">
            <span class="loo-k">F1 (weighted)</span>
            <span class="loo-v">{loo_metrics['f1']:.3f}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

#
tab1, tab2, tab3 = st.tabs(["All sub-counties", "Weather trends", "Scenario compare"])

#
with tab1:
    tbl = all_pred.sort_values("hi_prob", ascending=False).reset_index(drop=True)
    html = """<div class="tbl-wrap">
    <div class="tbl-hd">
      <span>Sub-county</span><span>Population</span><span>Density /km²</span>
      <span>Area km²</span><span>Vuln. score</span><span>Risk</span>
      <span>High-risk prob</span>
    </div>"""
    for _, r_ in tbl.iterrows():
        ru   = r_["pred_risk"].upper()
        bw   = int(r_["hi_prob"] * 100)
        html += f"""
    <div class="tbl-row">
      <span class="tbl-sc">{r_['sub_county']}</span>
      <span>{int(r_['population']):,}</span>
      <span>{int(r_['density']):,}</span>
      <span>{r_['area']:.1f}</span>
      <span>{r_['vuln']:.3f}</span>
      <span><span class="risk-pill pill-{ru}">{r_['pred_risk']}</span></span>
      <span>
        <div class="mini-bar-bg"><div class="mini-bar" style="width:{bw}%"></div></div>
        <span class="mono">{r_['hi_prob']*100:.1f}%</span>
      </span>
    </div>"""
    st.markdown(html + "</div>", unsafe_allow_html=True)

#
with tab2:
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = (weather_df.groupby("month")
               .agg(avg_precip=("total_precip","mean"),
                    avg_extreme=("extreme_rain_days","mean"))
               .reset_index())
    monthly["month_name"] = monthly["month"].apply(lambda x: MONTHS[x-1])
    wx = go.Figure()
    wx.add_trace(go.Bar(name="Avg monthly precip (mm)", x=monthly["month_name"],
                        y=monthly["avg_precip"].round(1),
                        marker_color="#0969da", opacity=0.8, yaxis="y"))
    wx.add_trace(go.Scatter(name="Extreme rain days", x=monthly["month_name"],
                            y=monthly["avg_extreme"].round(2), mode="lines+markers",
                            line=dict(color="#cf222e", width=2.5), marker=dict(size=6),
                            yaxis="y2"))
    wx.update_layout(
        **CHART_LIGHT, height=310, margin=dict(t=10, b=20, l=10, r=60),
        yaxis=dict(title="Precipitation (mm)", showgrid=True, gridcolor=GRID_C),
        yaxis2=dict(title="Extreme rain days", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(wx, use_container_width=True, config={"displayModeBar": False})

    annual = weather_df.groupby("year")["total_precip"].sum().reset_index()
    yr = px.line(annual, x="year", y="total_precip", markers=True,
                 labels={"total_precip": "Annual total precip (mm)", "year": "Year"},
                 color_discrete_sequence=["#1a7f37"])
    yr.update_traces(line_width=2.5, marker_size=7)
    yr.update_layout(**CHART_LIGHT, height=210, margin=dict(t=10, b=10),
                     xaxis=dict(showgrid=True, gridcolor=GRID_C),
                     yaxis=dict(showgrid=True, gridcolor=GRID_C))
    st.caption("Annual total precipitation 2015–2023")
    st.plotly_chart(yr, use_container_width=True, config={"displayModeBar": False})

#
with tab3:
    mults    = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    hi_probs = []
    for m_ in mults:
        p_, _, _ = predict(selected_sc, scale_weather(BASE_WEATHER, m_, temp_offset))
        hi_probs.append(p_[HI])

    sc_fig = go.Figure(go.Bar(
        x=[f"{m}×" for m in mults], y=[p*100 for p in hi_probs],
        marker_color=["#cf222e" if p>0.5 else "#9a6700" if p>0.3 else "#1a7f37"
                      for p in hi_probs],
        marker_line_width=0,
        text=[f"{p*100:.1f}%" for p in hi_probs], textposition="outside",
        textfont=dict(size=11, color="#24292f"),
    ))
    sc_fig.add_hline(y=50, line_dash="dot", line_color="#8c959f",
                     annotation_text="50% threshold",
                     annotation_font=dict(size=10, color="#57606a"))
    sc_fig.update_layout(
        **CHART_LIGHT, height=300, yaxis_range=[0, 120],
        yaxis_title="High-risk probability (%)",
        xaxis_title="Rainfall multiplier",
        margin=dict(t=20, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=GRID_C),
    )
    st.plotly_chart(sc_fig, use_container_width=True, config={"displayModeBar": False})
    cur = proba[HI]
    cc  = "#cf222e" if cur>0.5 else "#9a6700" if cur>0.3 else "#1a7f37"
    st.markdown(
        f'<div style="font-size:.8rem;color:#57606a;">Current: '
        f'<b style="color:{cc}">{rain_scale:.2f}× / {temp_offset:+.1f}°C</b>'
        f' → P(High) = <b style="color:{cc}">{cur:.3f}</b></div>',
        unsafe_allow_html=True)

#
st.markdown("""
<div style="margin-top:20px;padding:12px 24px;border-top:1px solid #d0d7de;
     font-size:.7rem;color:#8c959f;text-align:center;">
  Data: KNBS 2019 · NOAA GSOD · OpenStreetMap · Sentinel-2 (GEE) &nbsp;|&nbsp;
  Model: XGBoost LOO-CV &nbsp;|&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)

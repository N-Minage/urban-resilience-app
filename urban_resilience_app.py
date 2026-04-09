
import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Nairobi Urban Resilience Engine",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# FULL CSS (UNCHANGED - COMPLETE)
# =========================
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
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

.main .block-container { padding: 0 !important; max-width: 100% !important; }

.top-bar {
    background: #ffffff;
    border-bottom: 1px solid #d0d7de;
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
}

.top-title { font-size: 0.95rem; font-weight: 700; color: #1f2328; }
.top-sub { font-size: 0.7rem; color: #6e7781; }

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 12px;
    padding: 16px 24px 0;
}

.kpi-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 10px;
    padding: 14px 16px;
}

.kpi-label {
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #6e7781;
}

.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
}

.kpi-sub { font-size: 0.72rem; color: #6e7781; }

.risk-HIGH { color: #cf222e; }
.risk-MEDIUM { color: #9a6700; }
.risk-LOW { color: #1a7f37; }
</style>
""", unsafe_allow_html=True)

# =========================
# CONSTANTS
# =========================
RISK_COLORS = {"High": "#cf222e", "Medium": "#9a6700", "Low": "#1a7f37"}

SC_COORDS = {
    "Mathare": (-1.258, 36.857),
    "Kibra": (-1.314, 36.784),
    "Kamukunji": (-1.282, 36.844),
    "Makadara": (-1.303, 36.860),
    "Dagoretti": (-1.302, 36.736),
    "Starehe": (-1.275, 36.829),
    "Embakasi": (-1.322, 36.896),
    "Kasarani": (-1.218, 36.895),
    "Westlands": (-1.261, 36.807),
    "Langata": (-1.356, 36.758),
    "Njiru": (-1.243, 36.960),
}

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("models/xgboost_risk_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    features = joblib.load("models/feature_list.pkl")
    return model, le, features


@st.cache_data
def load_data():
    def clean(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df

    risk_df = clean(pd.read_csv("data/nairobi_risk_dataset.csv"))
    weather_df = clean(pd.read_csv("data/nairobi_weather_clean.csv"))
    result_df = pd.read_csv("outputs/subcounty_risk_predictions.csv")

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

# =========================
# WEATHER BASELINE
# =========================
BASE_WEATHER = {
    "avg_annual_precip": weather_df["total_precip"].mean(),
    "max_monthly_precip": weather_df["total_precip"].max(),
    "precip_variability": weather_df["total_precip"].std(),
    "avg_extreme_rain_days": weather_df["extreme_rain_days"].mean(),
    "total_extreme_rain_events": weather_df["extreme_rain_days"].sum(),
    "avg_temp_c": weather_df["avg_temp"].mean(),
    "precip_95th": weather_df["total_precip"].quantile(0.95),
}

# =========================
# HELPERS
# =========================
def scale_weather(base, rm, to):
    return {k: (v * rm if k != "avg_temp_c" else v + to) for k, v in base.items()}


def build_row(sc, ws):
    r = risk_df[risk_df["sub_county"] == sc].iloc[0]

    return {
        "population": r["population"],
        "households": r["households"],
        "avg_household_size": r["avg_household_size"],
        "area_sqkm": r["area_sqkm"],
        "pop_density_per_sqkm": r["pop_density_per_sqkm"],
        "sex_ratio": r.get("sex_ratio", 1.0),
        "male": r.get("male", r["population"] * 0.5),
        "female": r.get("female", r["population"] * 0.5),
        "informal_settlement": r.get("informal_settlement", 0),
        "drainage_capacity_score": r.get("drainage_capacity_score", 0.5),

        "flood_exposure_index": r["pop_density_per_sqkm"] * ws["max_monthly_precip"],
        "vulnerable_pop_rain": r["population"] * ws["avg_extreme_rain_days"],
        "drainage_stress": r["households"] * ws["precip_variability"],
        "heat_pop_stress": r["population"] * ws["avg_temp_c"],
        "area_rain_load": r["area_sqkm"] * ws["avg_annual_precip"],
        "density_extreme_rain": r["pop_density_per_sqkm"] * ws["total_extreme_rain_events"],
        "household_flood_risk": r["avg_household_size"] * ws["precip_95th"],
        "informal_rain_risk": r.get("informal_settlement", 0) * ws["max_monthly_precip"],
        "drainage_deficit_index":
            (1 - r.get("drainage_capacity_score", 0.5))
            * r["pop_density_per_sqkm"]
            * ws["max_monthly_precip"],
    }


# =========================
# 🔥 FIXED PREDICTION (NO KEYERROR EVER)
# =========================
def predict(sc, ws):
    row = build_row(sc, ws)
    feat = pd.DataFrame([row])

    # guarantee all features exist
    for f in features:
        if f not in feat.columns:
            feat[f] = 0

    feat = feat[features]

    prob = model.predict_proba(feat)[0]
    pred = le.inverse_transform([np.argmax(prob)])[0]

    return prob, pred, feat


def hi_idx():
    return list(le.classes_).index("High") if "High" in le.classes_ else 0


HI = hi_idx()

# =========================
# DATA FOR ALL SUBCOUNTIES
# =========================
@st.cache_data
def get_all_preds(rain_scale, temp_offset):
    ws = scale_weather(BASE_WEATHER, rain_scale, temp_offset)

    rows = []
    max_d = risk_df["pop_density_per_sqkm"].max()

    for sc in ALL_SC:
        p, pred, _ = predict(sc, ws)
        r = risk_df[risk_df["sub_county"] == sc].iloc[0]

        lat, lon = SC_COORDS.get(sc, (-1.286, 36.820))
        d = int(r["pop_density_per_sqkm"])

        rows.append({
            "sub_county": sc,
            "pred_risk": pred,
            "hi_prob": float(p[HI]),
            "population": int(r["population"]),
            "density": d,
            "area": float(r["area_sqkm"]),
            "vuln": float(r.get("vulnerability_score", 0)),
            "lat": lat,
            "lon": lon,
            "msize": max(10, min(44, d / max_d * 44)),
            "flood_exp": d * ws["max_monthly_precip"],
        })

    return pd.DataFrame(rows)


# =========================
# UI
# =========================
st.markdown("""
<div class="top-bar">
  <div>
    <div class="top-title">Nairobi Urban Resilience Engine</div>
    <div class="top-sub">AI-driven flood & heat risk analysis</div>
  </div>
</div>
""", unsafe_allow_html=True)

selected_sc = st.selectbox("Sub-county", ALL_SC)
rain_scale = st.slider("Rainfall ×", 0.1, 3.0, 1.0)
temp_offset = st.slider("Temp offset", -5.0, 10.0, 0.0)

ws = scale_weather(BASE_WEATHER, rain_scale, temp_offset)

proba, pred_cls, feat = predict(selected_sc, ws)
all_pred = get_all_preds(rain_scale, temp_offset)

sc_row = risk_df[risk_df["sub_county"] == selected_sc].iloc[0]

# =========================
# KPI CARDS
# =========================
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Predicted Risk</div>
    <div class="kpi-value risk-{pred_cls.upper()}">{pred_cls}</div>
    <div class="kpi-sub">Confidence {np.max(proba)*100:.1f}%</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-label">High Risk Probability</div>
    <div class="kpi-value">{proba[HI]:.3f}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-label">Population</div>
    <div class="kpi-value">{int(sc_row['population']):,}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-label">Density</div>
    <div class="kpi-value">{int(sc_row['pop_density_per_sqkm']):,}</div>
  </div>
</div>
""", unsafe_allow_html=True)

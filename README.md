# 🏛️ Urban Resilience Engine — Nairobi

A Machine Learning decision support system that helps city planners understand and predict the impact of extreme weather events (floods, heatwaves, storms) across Nairobi’s sub-counties.

The system combines **machine learning, geospatial data, and satellite imagery analysis** to produce risk scores and interactive visual insights.

---

## 🌍 Project Overview

Urban areas are increasingly vulnerable to climate-related disasters. This project builds an intelligent system that:

- Predicts **flood and heat risk levels**
- Identifies **high-risk sub-counties in Nairobi**
- Visualizes risk patterns on an interactive dashboard
- Supports **data-driven urban planning and disaster preparedness**

---

## 🧠 Key Idea

The system turns raw environmental and demographic data into actionable insights:

> “Which parts of the city are most at risk if rainfall increases or temperatures rise?”


## 🏗️ System Architecture (4 Phases)

### 1. 📊 Data Collection & Cleaning
We collect and preprocess data from multiple sources:

- 🌦️ Weather data (rainfall, temperature, extreme events)
- 🧍 Census & population data (KNBS)
- 🗺️ Geographic sub-county data
- 🌱 Infrastructure indicators (drainage, density, settlement type)

All datasets are cleaned, standardized, and merged into a unified analytical dataset.

### 2. 🛰️ Satellite Image Analysis (CNN)
We use Convolutional Neural Networks (CNNs) to analyze satellite imagery to detect:

- Urban density patterns
- Green space coverage
- Informal settlement expansion
- Environmental degradation signals

This helps enrich traditional datasets with **visual environmental intelligence**.


### 3. 🤖 Risk Prediction Model
A machine learning model (XGBoost) is trained to predict risk levels:

- 🟥 High Risk
- 🟧 Medium Risk
- 🟩 Low Risk

### Features include:
- Population density
- Flood exposure indicators
- Rainfall variability
- Temperature stress
- Infrastructure resilience scores

The model outputs:
- Risk class prediction
- Probability of high-risk conditions


### 4. 📊 Interactive Dashboard + Ethics Report
A Streamlit-based dashboard visualizes:

- 🗺️ Risk map of Nairobi sub-counties
- 📈 Probability breakdown per region
- 🌧️ Weather scenario simulations
- 📊 Feature importance (model explainability)
- 🧾 Risk distribution analysis

### Ethics Component:
The project evaluates fairness by analyzing whether:

- Poorer or informal settlements are disproportionately labeled as high-risk
- Model bias exists due to population density or infrastructure inequality
- Predictions reinforce or mitigate inequality


## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** — data processing
- **Scikit-learn / XGBoost** — machine learning
- **Streamlit** — dashboard interface
- **Plotly** — interactive visualizations
- **Joblib** — model serialization
- **Google Earth Engine (optional)** — satellite data


## 📁 Project Structure
urban-resilience-engine/
│
├── app.py                          # Streamlit application (main entry point)
│
├── data/
│   ├── nairobi_risk_dataset.csv
│   ├── nairobi_weather_clean.csv
│   ├── subcounty_risk_predictions.csv
│
├── models/
│   ├── xgboost_risk_model.pkl
│   ├── label_encoder.pkl
│   ├── feature_list.pkl
│   ├── loo_metrics.json
│
├── requirements.txt
├── README.md
└── .gitignore

## 🛠️ View the Live App
https://urban-resilience-app-a4p8sxdx74kdzaw6xjchg5.streamlit.app/


## 🚀 How to Run the Project
### 1. Install dependencies
```bash
pip install -r requirements.txt

streamlit run urban_resilience_app.py


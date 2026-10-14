# ⚡ Wafrha Tenwarha

**Smart Electricity Consumption Analyzer** — a machine learning project that analyzes household electricity usage, detects anomalies, finds peak consumption periods, forecasts future usage, classifies consumption levels, and clusters similar days.

The project includes a full Jupyter notebook pipeline and an interactive Streamlit dashboard.

---

## 📦 What's Included

| File | Description |
|------|-------------|
| `wafrha_tenwarha.ipynb` | Full notebook: data prep, EDA, forecasting, anomaly detection, classification, clustering |
| `app.py` | Streamlit dashboard with 6 tabs and sidebar controls |
| `requirements.txt` | All Python dependencies |

----

## 🚀 How to Run

```bash
pip install -r requirements.txt
```

1. Run `wafrha_tenwarha.ipynb` top to bottom.  
   The last cells save `hourly_power.csv`, which the dashboard uses as default data.

2. Run the dashboard in the same folder:
```bash
streamlit run app.py
```

---

## 🧪 What Was Tested

- Every notebook code cell was run end to end.
- No internet was available during development, so **synthetic minute-level data** shaped like the UCI file was used.
- The dashboard's data and model functions were tested, including a CSV upload in UCI format.
- **Streamlit itself was not installed**, so the dashboard's layout (tabs, sliders, charts) has not been visually verified.
- If you hit an error on first launch, report it and it will be fixed.

---

## 🔍 What Each Part Does

### Data Prep
- Minute-level readings become **hourly kW**.
- Daily kWh = average kW × 24.
- Days with fewer than 20 valid hours are dropped.

### Analysis & Peaks
- Compares last 7 and 30 days with the period before.
- Shows peak hours, weekday and month patterns, and a weekday × hour heatmap.

### Forecasting
- **Linear Regression** and **Random Forest** compared against a "same as yesterday" baseline.
- Evaluated on the last 20% of days.
- Best model gives next-day and 7-day forecasts.

### Anomalies
- A day is flagged if it exceeds the **previous 30-day mean + 2 standard deviations**.
- Only high-consumption days are flagged.

### Classification
- Days labeled **Low / Medium / High** using the 33rd and 66th percentiles of training days.
- A **Random Forest** predicts the level from previous days and calendar features.
- The model never sees the day it is predicting.

### Clustering
- Each day becomes 24 hourly values for **K-Means**.
- Includes elbow and silhouette plots.
- Default: 3 clusters.

### Dashboard
- 6 tabs + sidebar sliders for:
  - Forecast days
  - Anomaly sensitivity
  - Number of clusters

---

## 📂 Custom CSV Upload

In the sidebar:
1. Select **Upload my own CSV**.
2. Choose:
   - Date column
   - Optional time column
   - Power column (kW)
3. All models retrain automatically on your file.

**Requirements for custom data:**
- Must be **hourly or finer** (daily-only data can't give peak hours or clusters).
- Needs **at least ~4 months** of data.

---

## 📊 Model Notes

On the synthetic test data:
- **Random Forest** had the lower **MAE**.
- **Linear Regression** had the lower **RMSE**.

So the best model can change depending on the data. The notebook picks the best model by **RMSE**.

---

## ⚠️ Disclaimer

Wafrha Tenwarha is an **analytical and educational project**. It is not a validated system for guaranteeing electricity savings or generalizing consumption patterns to every household. Predictions and classifications depend on the quality and characteristics of the available data.




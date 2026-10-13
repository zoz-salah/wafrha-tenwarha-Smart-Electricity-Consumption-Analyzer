import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="wafrha tenwarha", page_icon="⚡", layout="wide")
sns.set_style("whitegrid")

FEATURES = ["lag_1", "lag_2", "lag_3", "lag_7", "avg_7", "dayofweek", "month"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LEVELS = ["Low", "Medium", "High"]


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Reading file...")
def read_csv_bytes(content):
    first_line = content[:2000].decode("utf-8", errors="ignore").splitlines()[0]
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    return pd.read_csv(io.BytesIO(content), sep=sep, na_values=["?"], low_memory=False)


@st.cache_data(show_spinner="Preparing hourly data...")
def to_hourly(df, date_col, time_col, power_col, dayfirst):
    text = df[date_col].astype(str)
    if time_col != "(none)":
        text = text + " " + df[time_col].astype(str)
    when = pd.to_datetime(text, dayfirst=dayfirst, errors="coerce")
    power = pd.to_numeric(df[power_col], errors="coerce")

    data = pd.DataFrame({"kw": power.values}, index=when.values).dropna()
    data = data[data.index.notna()].sort_index()
    if len(data) < 24:
        return None

    step = data.index.to_series().diff().median()
    if step > pd.Timedelta(hours=1):
        return None

    hourly = data["kw"].resample("h").mean()
    return hourly.interpolate(limit=3)


@st.cache_data
def build_daily(hourly):
    valid_hours = hourly.resample("D").count()
    daily = hourly.resample("D").mean() * 24
    daily = daily[valid_hours >= 20]
    return daily.asfreq("D").interpolate().rename("kwh")


@st.cache_data
def build_profile(hourly):
    data = hourly.dropna().to_frame("kw")
    data["date"] = data.index.normalize()
    data["hour"] = data.index.hour
    return data.pivot_table(index="date", columns="hour", values="kw").dropna()


# ------------------------------------------------------------------ models
def make_features(kwh):
    data = pd.DataFrame({"kwh": kwh})
    for lag in [1, 2, 3, 7]:
        data[f"lag_{lag}"] = data["kwh"].shift(lag)
    data["avg_7"] = data["kwh"].shift(1).rolling(7).mean()
    data["dayofweek"] = data.index.dayofweek
    data["month"] = data.index.month
    return data.dropna()


def make_row(history, date):
    return pd.DataFrame({
        "lag_1": [history.iloc[-1]],
        "lag_2": [history.iloc[-2]],
        "lag_3": [history.iloc[-3]],
        "lag_7": [history.iloc[-7]],
        "avg_7": [history.iloc[-7:].mean()],
        "dayofweek": [date.dayofweek],
        "month": [date.month],
    })


def forecast_next_days(model, kwh, days):
    history = kwh.copy()
    dates, values = [], []
    for _ in range(days):
        next_date = history.index[-1] + pd.Timedelta(days=1)
        value = model.predict(make_row(history, next_date)[FEATURES])[0]
        history.loc[next_date] = value
        dates.append(next_date)
        values.append(value)
    return pd.Series(values, index=dates, name="Forecast kWh")


def score(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


@st.cache_data(show_spinner="Training models...")
def train_all(kwh):
    data = make_features(kwh)
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    }

    results = {"Yesterday baseline": score(test["kwh"], test["lag_1"])}
    predictions = {}
    for name, model in models.items():
        model.fit(train[FEATURES], train["kwh"])
        predictions[name] = model.predict(test[FEATURES])
        results[name] = score(test["kwh"], predictions[name])
    results = pd.DataFrame(results).T.round(3)

    best_name = results.drop("Yesterday baseline")["RMSE"].idxmin()
    best_model = models[best_name]
    best_model.fit(data[FEATURES], data["kwh"])

    low, high = train["kwh"].quantile([0.33, 0.66])
    bins = [-np.inf, low, high, np.inf]
    y_train = pd.cut(train["kwh"], bins=bins, labels=LEVELS)
    y_test = pd.cut(test["kwh"], bins=bins, labels=LEVELS)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(train[FEATURES], y_train)
    y_pred = clf.predict(test[FEATURES])

    return {
        "results": results,
        "test": test,
        "predictions": predictions,
        "best_name": best_name,
        "best_model": best_model,
        "clf": clf,
        "low": low,
        "high": high,
        "accuracy": accuracy_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred, labels=LEVELS),
        "importances": pd.Series(clf.feature_importances_, index=FEATURES),
    }


# ------------------------------------------------------------------ sidebar
st.title("⚡ wafrha tenwarha")
st.caption("Smart Electricity Consumption Analyzer")

with st.sidebar:
    st.header("Data")
    source = st.radio("Data source", ["UCI household dataset", "Upload my own CSV"])

    hourly = None

    if source == "UCI household dataset":
        try:
            hourly = pd.read_csv("hourly_power.csv", index_col=0, parse_dates=True)["kw"]
        except FileNotFoundError:
            st.error("hourly_power.csv not found. Run the notebook first to create it.")
            st.stop()
    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv", "txt"])
        if uploaded is None:
            st.info("Upload a CSV with a date column (and optionally a time column) and a power column in kW. Data must be hourly or finer.")
            st.stop()

        raw = read_csv_bytes(uploaded.getvalue())
        columns = list(raw.columns)

        date_default = next((i for i, c in enumerate(columns) if "date" in c.lower()), 0)
        time_default = columns.index("Time") + 1 if "Time" in columns else 0
        power_default = next(
            (i for i, c in enumerate(columns) if "active_power" in c.lower() or "power" in c.lower() or "kw" in c.lower()),
            len(columns) - 1,
        )

        date_col = st.selectbox("Date (or datetime) column", columns, index=date_default)
        time_col = st.selectbox("Time column (optional)", ["(none)"] + columns, index=time_default)
        power_col = st.selectbox("Power column (kW)", columns, index=power_default)
        dayfirst = st.checkbox("Dates are day-first (31/12/2024)", value=True)

        hourly = to_hourly(raw, date_col, time_col, power_col, dayfirst)
        if hourly is None:
            st.error("Could not read the data. Make sure the columns are correct and the data is hourly or finer.")
            st.stop()

    kwh = build_daily(hourly)
    if len(kwh) < 120:
        st.error("Not enough data. At least about 4 months of daily data are needed.")
        st.stop()

    st.success(f"{len(kwh)} days: {kwh.index[0].date()} to {kwh.index[-1].date()}")

    st.header("Settings")
    forecast_days = st.slider("Days to forecast", 1, 14, 7)
    sigma = st.slider("Anomaly sensitivity (std. deviations)", 1.0, 3.0, 2.0, 0.5)
    n_clusters = st.slider("Number of clusters", 2, 6, 3)

info = train_all(kwh)

tab_overview, tab_peak, tab_forecast, tab_anomaly, tab_level, tab_cluster = st.tabs(
    ["Overview", "Peak Usage", "Forecast", "Anomalies", "Levels", "Clusters"]
)

# ------------------------------------------------------------------ overview
with tab_overview:
    st.subheader("Has usage increased or decreased?")

    cols = st.columns(4)
    cols[0].metric("Days in data", len(kwh))
    cols[1].metric("Average per day", f"{kwh.mean():.1f} kWh")

    for col, days in zip(cols[2:], [7, 30]):
        recent = kwh.iloc[-days:].mean()
        before = kwh.iloc[-2 * days:-days].mean()
        change = (recent - before) / before * 100
        col.metric(f"Last {days} days (kWh/day)", f"{recent:.1f}", f"{change:+.1f}% vs previous {days} days", delta_color="inverse")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(kwh, color="lightgray", label="Daily kWh")
    ax.plot(kwh.rolling(7).mean(), label="7-day average")
    ax.plot(kwh.rolling(30).mean(), label="30-day average", linewidth=2)
    ax.set_ylabel("kWh")
    ax.legend()
    show(fig)

# ------------------------------------------------------------------ peak
with tab_peak:
    st.subheader("When is the most power used?")

    hourly_df = hourly.dropna().to_frame("kw")
    hourly_df["hour"] = hourly_df.index.hour
    hourly_df["dayofweek"] = hourly_df.index.dayofweek

    hour_avg = hourly_df.groupby("hour")["kw"].mean()
    st.write("Top 3 peak hours:", ", ".join(f"{h}:00" for h in hour_avg.nlargest(3).index))

    left, right = st.columns(2)

    fig, ax = plt.subplots(figsize=(7, 4))
    hour_avg.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Average Power by Hour")
    ax.set_ylabel("kW")
    left.pyplot(fig)
    plt.close(fig)

    dow_avg = kwh.groupby(kwh.index.dayofweek).mean()
    dow_avg.index = [DAY_NAMES[i] for i in dow_avg.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    dow_avg.plot(kind="bar", ax=ax, color="darkorange")
    ax.set_title("Average Daily kWh by Day of Week")
    ax.set_ylabel("kWh")
    right.pyplot(fig)
    plt.close(fig)

    heat = hourly_df.pivot_table(index="dayofweek", columns="hour", values="kw")
    heat.index = [DAY_NAMES[i] for i in heat.index]
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(heat, cmap="YlOrRd", ax=ax)
    ax.set_title("Average Power (kW): Day of Week vs Hour")
    show(fig)

# ------------------------------------------------------------------ forecast
with tab_forecast:
    st.subheader("Model comparison (tested on the most recent 20% of days)")
    st.dataframe(info["results"])
    st.write(f"Best model: **{info['best_name']}**")

    test = info["test"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test.index[-90:], test["kwh"].iloc[-90:], color="black", label="Actual")
    for name, pred in info["predictions"].items():
        ax.plot(test.index[-90:], pred[-90:], label=name)
    ax.set_ylabel("kWh")
    ax.legend()
    show(fig)

    st.subheader(f"Next {forecast_days} day(s)")
    future = forecast_next_days(info["best_model"], kwh, forecast_days)
    st.metric(f"Forecast for {future.index[0].date()}", f"{future.iloc[0]:.1f} kWh")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(kwh.iloc[-30:], label="Last 30 days")
    ax.plot(future, marker="o", color="red", label="Forecast")
    ax.set_ylabel("kWh")
    ax.legend()
    show(fig)

    st.dataframe(future.round(1).to_frame())

# ------------------------------------------------------------------ anomalies
with tab_anomaly:
    st.subheader("Days that are unusually high vs the previous 30 days")

    previous = kwh.shift(1)
    mean_30 = previous.rolling(30).mean()
    std_30 = previous.rolling(30).std()
    limit = mean_30 + sigma * std_30
    is_anomaly = kwh > limit

    anomalies = pd.DataFrame({"kWh": kwh, "30-day average": mean_30})[is_anomaly]
    anomalies["% above average"] = (anomalies["kWh"] / anomalies["30-day average"] - 1) * 100

    st.write(f"**{len(anomalies)}** anomalous days ({len(anomalies) / len(kwh) * 100:.1f}% of all days)")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(kwh, color="lightgray", label="Daily kWh")
    ax.plot(limit, color="orange", linewidth=1, label="Anomaly limit")
    ax.scatter(anomalies.index, anomalies["kWh"], color="red", s=15, label="Anomaly")
    ax.set_ylabel("kWh")
    ax.legend()
    show(fig)

    st.dataframe(anomalies.sort_values("% above average", ascending=False).round(1))

# ------------------------------------------------------------------ levels
with tab_level:
    st.subheader("Low / Medium / High days")
    st.write(f"Low: below {info['low']:.1f} kWh | Medium: up to {info['high']:.1f} kWh | High: above {info['high']:.1f} kWh")

    tomorrow = kwh.index[-1] + pd.Timedelta(days=1)
    tomorrow_level = info["clf"].predict(make_row(kwh, tomorrow)[FEATURES])[0]

    c1, c2 = st.columns(2)
    c1.metric("Classifier accuracy (test days)", f"{info['accuracy'] * 100:.1f}%")
    c2.metric(f"Predicted level for {tomorrow.date()}", tomorrow_level)

    left, right = st.columns(2)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(info["cm"], annot=True, fmt="d", cmap="Blues", xticklabels=LEVELS, yticklabels=LEVELS, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    left.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    info["importances"].sort_values().plot(kind="barh", ax=ax, color="teal")
    ax.set_title("Feature Importance")
    right.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------------ clusters
with tab_cluster:
    st.subheader("Groups of similar days (K-Means)")

    profile = build_profile(hourly)
    if len(profile) < 30:
        st.info("Not enough complete days (with all 24 hours) for clustering.")
    else:
        hour_cols = list(profile.columns)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        profile["cluster"] = kmeans.fit_predict(profile[hour_cols])

        centers = profile.groupby("cluster")[hour_cols].mean()
        fig, ax = plt.subplots(figsize=(12, 4))
        for cluster_id, row in centers.iterrows():
            ax.plot(row.index, row.values, linewidth=2, label=f"Cluster {cluster_id}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("kW")
        ax.set_title("Average Daily Pattern of Each Cluster")
        ax.legend()
        show(fig)

        profile["daily_kwh"] = profile[hour_cols].mean(axis=1) * 24
        profile["is_weekend"] = profile.index.dayofweek >= 5
        summary = profile.groupby("cluster").agg(
            days=("daily_kwh", "size"),
            avg_kwh=("daily_kwh", "mean"),
            weekend_share=("is_weekend", "mean"),
        )
        summary["weekend_share"] = summary["weekend_share"] * 100
        summary.columns = ["Days", "Avg kWh per day", "Weekend share (%)"]
        st.dataframe(summary.round(1))

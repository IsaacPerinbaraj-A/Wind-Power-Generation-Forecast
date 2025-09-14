import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Load model and scaler
# -----------------------
@st.cache_resource
def load_model():
    model = joblib.load("wind_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Wind Power Prediction", layout="wide")

# -----------------------
# Introduction Section
# -----------------------
st.title("💨 Wind Turbine Power Prediction App")
st.markdown("""
Welcome! This app allows you to:
- Predict **Active Power (kW)** of a wind turbine given wind speed and time.
- Analyze turbine datasets.
- Visualize trends using **Matplotlib/Seaborn** plots.
""")
st.info("Scroll down to make predictions or upload data for analysis.")

# -----------------------
# Prediction Section
# -----------------------
st.header("🔮 Make a Prediction")
with st.form("prediction_form"):
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, step=0.1)
    theo_power = st.number_input("Theoretical Power Curve (KWh)", min_value=0.0, step=0.1)
    hour = st.slider("Hour of Day", 0, 23, 12)
    day = st.slider("Day of Month", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)

    submit = st.form_submit_button("Predict")

if submit:
    input_data = np.array([[wind_speed, theo_power, hour, day, month]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Active Power: **{prediction:.2f} kW**")

# -----------------------
# Data Analysis Section
# -----------------------
st.header("📊 Data Analysis")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Try to parse Date/Time
    if "Date/Time" in df.columns:
        df["Date/Time"] = pd.to_datetime(df["Date/Time"], dayfirst=True, errors="coerce")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Scatter Plot: Wind Speed vs Active Power
    if "Wind Speed (m/s)" in df.columns and "Active Power (kW)" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x="Wind Speed (m/s)", y="Active Power (kW)", ax=ax)
        ax.set_title("Wind Speed vs Active Power")
        ax.grid(True)
        st.pyplot(fig)

    # Time Series: Active Power over Time
    if "Date/Time" in df.columns and "Active Power (kW)" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df, x="Date/Time", y="Active Power (kW)", ax=ax)
        ax.set_title("Active Power Over Time")
        ax.set_xlabel("Date/Time")
        ax.set_ylabel("Active Power (kW)")
        ax.grid(True)
        st.pyplot(fig)

    # Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

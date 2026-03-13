import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from scipy.io import loadmat

st.set_page_config(page_title="Bearing Fault Dashboard", layout="wide")

# -------- Custom CSS (Dark Dashboard UI) --------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.metric-card {
    background-color:#1e293b;
    padding:20px;
    border-radius:12px;
    text-align:center;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# -------- Sidebar --------
st.sidebar.title("⚙️ Bearing Monitor")
st.sidebar.write("Upload vibration signal")

# Load model
model = joblib.load("bearing_model.pkl")

file = st.sidebar.file_uploader("Upload file", type=["csv","mat"])

st.title("AI Bearing Condition Monitoring Dashboard")

if file is not None:

    # ---------- Load Signal ----------
    if file.name.endswith(".csv"):
        data = pd.read_csv(file)
        signal = data.values.flatten()

    else:
        mat = loadmat(file)
        signal = None
        for key in mat.keys():
            if "DE_time" in key:
                signal = mat[key].flatten()

    # ---------- Features ----------
    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal)
    sk = skew(signal)
    peak = np.max(signal)
    std = np.std(signal)

    X = [[rms, kurt, sk, peak, std]]

    prediction = model.predict(X)[0]
    confidence = np.max(model.predict_proba(X))*100

    # -------- KPI CARDS --------
    c1,c2,c3,c4 = st.columns(4)

    c1.metric("RMS", round(rms,3))
    c2.metric("Kurtosis", round(kurt,3))
    c3.metric("Peak", round(peak,3))
    c4.metric("Confidence", f"{confidence:.1f}%")

    st.divider()

    # -------- Waveform --------
    col1,col2 = st.columns(2)

    df = pd.DataFrame({"signal":signal})

    with col1:
        st.subheader("Vibration Waveform")
        fig = px.line(df,y="signal")
        st.plotly_chart(fig,use_container_width=True)

    # -------- FFT --------
    with col2:

        st.subheader("Frequency Spectrum")

        fft_values = np.abs(fft(signal))
        freq = np.arange(len(fft_values))

        df_fft = pd.DataFrame({
            "freq":freq[:500],
            "amp":fft_values[:500]
        })

        fig2 = px.line(df_fft,x="freq",y="amp")

        st.plotly_chart(fig2,use_container_width=True)

    st.divider()

    # -------- Prediction Panel --------
    col3,col4 = st.columns(2)

    with col3:
        st.subheader("Fault Prediction")
        st.success(prediction)

    with col4:
        st.subheader("Health Indicator")

        if confidence > 90:
            st.progress(100)
        elif confidence > 70:
            st.progress(70)
        else:
            st.progress(40)

else:
    st.info("Upload vibration data to start analysis")

import os

image_map = {
    "Normal": "images/normal.png",
    "Inner Race": "images/inner_race.png",
    "Outer Race": "images/outer_race.png",
    "Ball Fault": "images/ball_fault.png"
}

if prediction in image_map:
    st.subheader("Bearing Fault Illustration")

    image_path = image_map[prediction]

    if os.path.exists(image_path):
        st.image(image_path, width=350)
    else:
        st.error(f"Image not found: {image_path}")


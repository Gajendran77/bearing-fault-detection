import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from scipy.io import loadmat
import os

# -------- PAGE CONFIG --------
st.set_page_config(page_title="AI Bearing Monitoring", layout="wide")

# -------- CUSTOM STYLE --------
st.markdown("""
<style>
.main {
    background-color:#0f172a;
}
h1,h2,h3 {
    color:white;
}
</style>
""", unsafe_allow_html=True)

# -------- SIDEBAR --------
st.sidebar.title("⚙️ Bearing Monitoring System")
file = st.sidebar.file_uploader("Upload vibration file", type=["csv","mat"])

# -------- LOAD MODEL --------
model = joblib.load("bearing_model.pkl")

st.title("AI Bearing Condition Monitoring Dashboard")

if file is not None:

    # ---------- LOAD SIGNAL ----------
    if file.name.endswith(".csv"):
        data = pd.read_csv(file)
        signal = data.values.flatten()

    else:
        mat = loadmat(file)
        signal = None

        for key in mat.keys():
            if "DE_time" in key:
                signal = mat[key].flatten()

    # ---------- FEATURES ----------
    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal)
    sk = skew(signal)
    peak = np.max(signal)
    std = np.std(signal)

    X = [[rms,kurt,sk,peak,std]]

    prediction = model.predict(X)[0]
    confidence = np.max(model.predict_proba(X))*100

    # ---------- KPI METRICS ----------
    c1,c2,c3,c4 = st.columns(4)

    c1.metric("RMS", round(rms,3))
    c2.metric("Kurtosis", round(kurt,3))
    c3.metric("Peak", round(peak,3))
    c4.metric("Std Dev", round(std,3))

    st.divider()

    # ---------- CHARTS ----------
    col1,col2 = st.columns(2)

    df = pd.DataFrame({"Signal":signal})

    with col1:
        st.subheader("Vibration Waveform")
        fig = px.line(df,y="Signal")
        st.plotly_chart(fig,use_container_width=True)

    with col2:

        st.subheader("Frequency Spectrum")

        fft_values = np.abs(fft(signal))
        freq = np.arange(len(fft_values))

        df_fft = pd.DataFrame({
            "Frequency":freq[:500],
            "Amplitude":fft_values[:500]
        })

        fig2 = px.line(df_fft,x="Frequency",y="Amplitude")
        st.plotly_chart(fig2,use_container_width=True)

    st.divider()

    # ---------- RESULT SECTION ----------
    col3,col4 = st.columns(2)

    with col3:

        st.subheader("Fault Prediction")

        st.success(prediction)

        explanation = {
            "Normal":"Bearing operates under normal condition.",
            "Inner Race":"Inner race defect causes periodic impacts.",
            "Outer Race":"Outer race fault produces repetitive vibration peaks.",
            "Ball Fault":"Rolling element damage creates irregular vibration."
        }

        st.info(explanation.get(prediction,""))

    with col4:

        st.subheader("Prediction Confidence")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text':"Model Confidence"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"#22c55e"},
                'steps':[
                    {'range':[0,50],'color':'#ef4444'},
                    {'range':[50,75],'color':'#facc15'},
                    {'range':[75,100],'color':'#22c55e'}
                ]
            }
        ))

        st.plotly_chart(fig_gauge,use_container_width=True)

    st.divider()

    # ---------- FEATURE TABLE ----------
    st.subheader("Extracted Vibration Features")

    feature_df = pd.DataFrame({
        "Feature":["RMS","Kurtosis","Skewness","Peak","Std Dev"],
        "Value":[rms,kurt,sk,peak,std]
    })

    st.table(feature_df)

    st.divider()

    # ---------- BEARING IMAGE ----------
    image_map = {
        "Normal":"images/normal.png",
        "Inner Race":"images/inner_race.png",
        "Outer Race":"images/outer_race.png",
        "Ball Fault":"images/ball_fault.png"
    }

    if prediction in image_map:

        st.subheader("Bearing Fault Illustration")

        img = image_map[prediction]

        if os.path.exists(img):
            st.image(img,use_container_width=True)
        else:
            st.warning("Image not found")

else:
    st.info("Upload vibration data to start analysis")

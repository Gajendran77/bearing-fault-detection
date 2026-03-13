import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from scipy.io import loadmat
from reportlab.pdfgen import canvas
import os
import io

st.set_page_config(page_title="AI Bearing Monitoring", layout="wide")

st.title("⚙️ AI Bearing Condition Monitoring Dashboard")

# Sidebar
st.sidebar.header("Upload Vibration Data")
file = st.sidebar.file_uploader("Upload .csv or .mat", type=["csv","mat"])

demo = st.sidebar.checkbox("Run Demo Signal")

model = joblib.load("bearing_model.pkl")

# -------- DEMO SIGNAL --------
if demo:
    signal = np.sin(np.linspace(0,50,2000)) + np.random.normal(0,0.2,2000)

elif file is not None:

    if file.name.endswith(".csv"):
        data = pd.read_csv(file)
        signal = data.values.flatten()

    else:
        mat = loadmat(file)
        signal = None

        for key in mat.keys():
            if "DE_time" in key:
                signal = mat[key].flatten()

else:
    st.info("Upload vibration data or enable demo mode.")
    st.stop()

# -------- FEATURE EXTRACTION --------
rms = np.sqrt(np.mean(signal**2))
kurt = kurtosis(signal)
sk = skew(signal)
peak = np.max(signal)
std = np.std(signal)

X = [[rms,kurt,sk,peak,std]]

prediction = model.predict(X)[0]
confidence = np.max(model.predict_proba(X))*100

# -------- MACHINE HEALTH --------
if confidence > 90:
    health = "🟢 Healthy"
elif confidence > 70:
    health = "🟡 Warning"
else:
    health = "🔴 Fault"

# -------- KPI ROW --------
c1,c2,c3,c4,c5 = st.columns(5)

c1.metric("RMS", round(rms,3))
c2.metric("Kurtosis", round(kurt,3))
c3.metric("Peak", round(peak,3))
c4.metric("Std Dev", round(std,3))
c5.metric("Machine Status", health)

st.divider()

# -------- WAVEFORM --------
col1,col2 = st.columns(2)

df = pd.DataFrame({"Signal":signal})

with col1:
    st.subheader("Vibration Waveform")
    fig = px.line(df,y="Signal")
    st.plotly_chart(fig,use_container_width=True)

# -------- FFT --------
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

# -------- PREDICTION --------
col3,col4 = st.columns(2)

with col3:

    st.subheader("Fault Prediction")

    st.success(prediction)

    explanation = {
        "Normal":"Bearing operates under normal condition.",
        "Inner Race":"Inner race defect produces periodic impacts.",
        "Outer Race":"Outer race damage creates repetitive vibration peaks.",
        "Ball Fault":"Rolling element damage causes irregular vibration impulses."
    }

    st.info(explanation.get(prediction,""))

# -------- CONFIDENCE GAUGE --------
with col4:

    st.subheader("Prediction Confidence")

    gauge = go.Figure(go.Indicator(
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

    st.plotly_chart(gauge,use_container_width=True)

st.divider()

# -------- FEATURE TABLE --------
st.subheader("Extracted Vibration Features")

feature_df = pd.DataFrame({
    "Feature":["RMS","Kurtosis","Skewness","Peak","Std Dev"],
    "Value":[rms,kurt,sk,peak,std]
})

st.table(feature_df)

st.divider()

# -------- BEARING IMAGE --------
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

st.divider()

# -------- PDF REPORT --------
st.subheader("Download Diagnostic Report")

buffer = io.BytesIO()

c = canvas.Canvas(buffer)

c.drawString(100,800,"Bearing Fault Diagnosis Report")

c.drawString(100,760,f"Prediction: {prediction}")
c.drawString(100,740,f"Confidence: {confidence:.2f}%")
c.drawString(100,720,f"RMS: {rms}")
c.drawString(100,700,f"Kurtosis: {kurt}")
c.drawString(100,680,f"Peak: {peak}")

c.save()

buffer.seek(0)

st.download_button(
    label="Download PDF Report",
    data=buffer,
    file_name="bearing_diagnosis_report.pdf",
    mime="application/pdf"
)

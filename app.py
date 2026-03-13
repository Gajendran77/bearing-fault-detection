import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from scipy.io import loadmat

# Load trained model
model = joblib.load("bearing_model.pkl")

st.title("AI Bearing Fault Detection System")

file = st.file_uploader("Upload vibration data (.csv or .mat)", type=["csv","mat"])

if file is not None:

    # ---------- Load File ----------
    if file.name.endswith(".csv"):
        data = pd.read_csv(file)
        signal = data.values.flatten()

    elif file.name.endswith(".mat"):
        mat = loadmat(file)

        # automatically find vibration signal variable
        signal = None
        for key in mat.keys():
            if "DE_time" in key:
                signal = mat[key].flatten()

        if signal is None:
            st.error("No vibration signal found in MAT file.")
            st.stop()

    # ---------- Waveform ----------
    st.subheader("Vibration Waveform")

    fig1, ax1 = plt.subplots()
    ax1.plot(signal)
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time Domain Signal")

    st.pyplot(fig1)

    # ---------- FFT Spectrum ----------
    st.subheader("FFT Spectrum")

    fft_values = np.abs(fft(signal))
    freq = np.arange(len(fft_values))

    fig2, ax2 = plt.subplots()
    ax2.plot(freq[:500], fft_values[:500])
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Frequency Spectrum")

    st.pyplot(fig2)

    # ---------- Feature Extraction ----------
    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal)
    sk = skew(signal)
    peak = np.max(signal)
    std = np.std(signal)

    X = [[rms, kurt, sk, peak, std]]

    # ---------- Prediction ----------
    prediction = model.predict(X)
    confidence = np.max(model.predict_proba(X)) * 100

    st.subheader("Prediction Result")
    st.write("Detected Fault:", prediction[0])

    st.subheader("Prediction Confidence")
    st.write(f"{confidence:.2f}%")

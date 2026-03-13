import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.stats import kurtosis, skew

model = joblib.load("bearing_model.pkl")

st.title("Bearing Fault Detection System")

file = st.file_uploader("Upload vibration data (.csv)")

if file is not None:

    data = pd.read_csv(file)
    signal = data.values.flatten()

    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal)
    sk = skew(signal)
    peak = np.max(signal)
    std = np.std(signal)

    X = [[rms, kurt, sk, peak, std]]

    result = model.predict(X)

    st.write("Detected Fault:", result[0])
import streamlit as st

import pandas as pd
import joblib

# Title
st.header("Risk prediction of early neurological deterioration within 72 hours after thrombolytic therapy in ischemic stroke")

# Input bar 1
ott = st.number_input("Enter Time from onset to treatment (h)")

# Input bar 2
NIHSS_score_before_thrombolysis = st.number_input("Enter NIHSS score before thrombolysis")
#NIHSS_score_after_thrombolysis1 = st.number_input("Enter NIHSS score after thrombolysis")
hemoglobin = st.number_input("Enter hemoglobin (g/L)")
Thrombin_time = st.number_input("Enter Thrombin_time (sec)")
lymphocyte= st.number_input("Enter lymphocyte_count (×10⁹/L)")
# Dropdown input
prothrombin_time = st.number_input("Enter prothrombin_time (sec)")

# If button is pressed
if st.button("Submit"):
    # Unpickle classifier
    clf = joblib.load("D:/Documents/Thrombolysis/second/clf1.pkl")

    # Store inputs into dataframe
    X = pd.DataFrame([[ott, NIHSS_score_before_thrombolysis,hemoglobin,
                       Thrombin_time,lymphocyte,prothrombin_time]],
                     columns=["ott", "NIHSS_score_before_thrombolysis","hemoglobin",
                       "Thrombin_time","lymphocyte","prothrombin_time"])


    # Get prediction
    prediction = clf.predict(X)[0]

    # Output prediction
    st.text(f"This patient has a higher probability of {prediction} within 72 hours")
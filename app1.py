import streamlit as st
import shap
from streamlit_shap import st_shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Title
st.header("Risk prediction of early neurological deterioration within 72 hours after thrombolytic therapy in ischemic stroke")

# Input bar 1
ott = st.number_input("Enter Time from onset to treatment (h)")

# Input bar 2
White_blood_cell = st.number_input("Enter White blood cell count(×10⁹/L)")
LMR= st.number_input("Enter Lymphocyte count(×10⁹/L) to monocyte count(×10⁹/L) ratio")
#NIHSS_score_after_thrombolysis1 = st.number_input("Enter NIHSS score after thrombolysis")
hemoglobin = st.number_input("Enter Hemoglobin (g/L)")
Thrombin_time = st.number_input("Enter Thrombin_time (sec)")
prothrombin_time = st.number_input("Enter Prothrombin_time (sec)")
# Dropdown input


# If button is pressed
if st.button("Submit"):
    # Unpickle classifier
    clf = joblib.load("clfFinal.pkl")

    # Store inputs into dataframe
    X = pd.DataFrame([[ott,White_blood_cell,LMR,hemoglobin,
                       Thrombin_time,prothrombin_time]],
                     columns=["ott", "White_blood_cell","LMR","hemoglobin",
                       "Thrombin_time","prothrombin_time"])


    # Get prediction
    prediction = clf.predict(X)[0]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    #f = plt.figure()
    #shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
    #f.savefig("shap_force_plot.png", bbox_inches='tight', dpi=600)
    #Output prediction
    #P = mpimg.imread("shap_force_plot.png")
    #st.image(P, caption="shap_force_plot", channels="RGB")
    
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]), height=200, width=800)
    st.text(f"This patient has a higher probability of {prediction} within 72 hours")

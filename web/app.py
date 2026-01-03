import streamlit as st
import numpy as np
import pandas as pd
from data_processing import load_and_process_data, train_model
from visualization import render_simulator, render_analysis, render_physics_section

st.set_page_config(
    page_title="Exoplanet Temp Predictor",
    layout="wide"
)

st.title("Exoplanet Habitability Predictor")
st.markdown("""
Predict surface temperature of exoplanets using Linear Regression on Kepler Telescope data.
""")

st.sidebar.header("Data Configuration")
uploaded_file = st.sidebar.file_uploader("Upload 'exoplanet_data.csv'", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Simulator Parameters")
input_period = st.sidebar.slider("Orbital Period (Days)", 1.0, 1000.0, 365.25)
input_star_temp = st.sidebar.slider("Star Temperature (K)", 3000, 10000, 5778)
input_star_radius = st.sidebar.slider("Star Radius (Solar Radii)", 0.1, 10.0, 1.0)

if uploaded_file is None:
    st.info("Please upload your 'exoplanet_data.csv' file in the sidebar to begin.")
    st.stop()

# Load and process data
try:
    data, X_train, X_test, y_train, y_test, model = load_and_process_data(uploaded_file)
    
    # Get predictions and metrics
    preds_log = model.predict(X_test)
    preds = np.exp(preds_log)
    actual = np.exp(y_test)
    
    final_pred, calc_dist = render_simulator(
        model, input_period, input_star_temp, input_star_radius
    )
    
    st.divider()
    render_analysis(data, actual, preds, final_pred, X_train)
    st.markdown("---")
    render_physics_section()
    
except KeyError as e:
    st.error(f"Missing columns: {e}")
except Exception as e:
    st.error(f"Error: {e}")
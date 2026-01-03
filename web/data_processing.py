import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_data(file_input):
    """Load CSV data from uploaded file."""
    return pd.read_csv(file_input)

def load_and_process_data(uploaded_file):
    """
    Load, clean, and prepare exoplanet data for modeling.
    
    Returns:
        tuple: (data, X_train, X_test, y_train, y_test, model)
    """
    df_raw = load_data(uploaded_file)
    
    # Filter confirmed and candidate exoplanets
    df = df_raw[df_raw['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE'])].copy()
    
    # Select and clean data
    data = df[['koi_period', 'koi_steff', 'koi_srad', 'koi_teq']].dropna()
    
    # Calculating the distance from orbital period
    data['Distance'] = (data['koi_period'] / 365.25) ** (2/3)
    
    # Renaming columns for clarity
    data = data.rename(columns={
        'koi_steff': 'Star_Temp',
        'koi_srad': 'Star_Radius',
        'koi_teq': 'Planet_Temp'
    })
    
    # Apply log transformation for linear regression
    data['log_dist'] = np.log(data['Distance'])
    data['log_star_temp'] = np.log(data['Star_Temp'])
    data['log_star_radius'] = np.log(data['Star_Radius'])
    data['log_planet_temp'] = np.log(data['Planet_Temp'])
    
    # Prepare features and target
    X = data[['log_dist', 'log_star_temp', 'log_star_radius']]
    y = data['log_planet_temp']
    
    # Train-test split and model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = train_model(X_train, y_train)
    
    return data, X_train, X_test, y_train, y_test, model

def train_model(X_train, y_train):
    """Train Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def calculate_prediction(model, period, star_temp, star_radius):
    """
    Calculate predicted planet temperature based on input parameters.
    
    Returns:
        tuple: (predicted_temp, calculated_distance)
    """
    calc_dist = (period / 365.25) ** (2/3)
    
    user_input = pd.DataFrame([[
        np.log(calc_dist),
        np.log(star_temp),
        np.log(star_radius)
    ]], columns=['log_dist', 'log_star_temp', 'log_star_radius'])
    
    log_pred = model.predict(user_input)
    final_pred = np.exp(log_pred)[0]
    
    return final_pred, calc_dist

def get_habitability_status(temperature):
    """
    Determine if a planet is in the habitable zone based on temperature.
    
    Returns:
        tuple: (status_message, status_color)
    """
    if 180 < temperature < 330:
        return "Habitable Zone Candidate", "green"
    elif temperature >= 330:
        return "Too Hot", "red"
    else:
        return "Too Cold", "blue"
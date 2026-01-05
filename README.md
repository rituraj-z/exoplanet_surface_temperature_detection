# Exoplanet Surface Temperature Prediction

A comprehensive data science project that predicts exoplanet surface temperatures using machine learning on NASA Kepler Observatory data, featuring both analytical notebooks and an interactive web application.

## Project Overview

This project bridges astronomy and machine learning by building a predictive model for exoplanet equilibrium temperatures. The fascinating aspect is that through log transformations, the model learns coefficients that approximately matches actual physics equations.

**Key Achievement**: R² Score of **0.9958**.

## Project Structure

```
exoplanet_surface_temperature_detection/
│
├── web/                                    # Streamlit Web Application
│   ├── app.py                             # Main Streamlit entry point
│   ├── data_processing.py                 # Data pipeline & ML model
│   └── visualization.py                   # UI components & plots
│
├── exoplanet_data.csv                     # Kepler dataset (upload here)
├── exoplanet_surface_temperature_detection.ipynb  # Jupyter analysis notebook
└── README.md                              # This file
```

## Background: The Kepler Transit Method

The Kepler Space Telescope uses the **Transit Method** to detect exoplanets:

1. **Continuous Monitoring**: Kepler observes the brightness of thousands of stars
2. **Transit Detection**: When a planet passes in front of its star, it blocks a small portion of light (~0.01-1% dimmer)
3. **Periodic Dips**: These repeating brightness dips confirm a planet's existence
4. **Data Extraction**: From these transits, scientists calculate orbital periods and stellar properties

This transit data forms the foundation of our predictive model.

## Dataset

**Source**: [Kaggle - Kepler Exoplanet Search Results](https://www.kaggle.com/nasa/kepler-exoplanet-search-results)

**Description**: ~10,000 exoplanet candidates observed by NASA's Kepler Space Observatory (2009-2017)

**Key Features**:
| Column | Description | Unit |
|--------|-------------|------|
| `koi_disposition` | Classification status | CONFIRMED/CANDIDATE/FALSE POSITIVE |
| `koi_period` | Orbital period | Days |
| `koi_steff` | Host star temperature | Kelvin |
| `koi_srad` | Host star radius | Solar radii |
| `koi_teq` | Planet equilibrium temperature (target) | Kelvin |

## Problem Statement

Direct measurement of exoplanet temperatures is nearly impossible—the host star's brightness overwhelms any thermal signal from the planet. However, we can **predict temperature** using orbital characteristics and stellar properties through physics-based feature engineering.

## Methodology

### 1. Data Cleaning
Filter out FALSE POSITIVE entries (eclipsing binaries, instrumental artifacts) that don't follow planetary physics:

```python
df = df[df['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE'])]
```

### 2. Feature Engineering: Kepler's Third Law
Calculate orbital distance from period:

```python
Distance = (Period / 365.25) ** (2/3)  # Result in AU
```

### 3. Log Transformation
Transform the physics equation from non-linear to linear form:

**Original Physics (Stefan-Boltzmann Law)**:
```
T_planet = T_star × √(R_star / 2D)
```

**Log-Transformed (Linear)**:
```
log(T_planet) = log(T_star) + 0.5×log(R_star) - 0.5×log(D) - constant
```

### 4. Model Training
Train Linear Regression on log-transformed features:

```python
X = data[['log_dist', 'log_star_temp', 'log_star_radius']]
y = data['log_planet_temp']
model = LinearRegression()
model.fit(X_train, y_train)
```

## Results

### Model Performance
- **R² Score**: 0.9958 (explains 99.58% of temperature variance)
- **Mean Absolute Error**: ~10.16 Kelvin

### Coefficient Validation
| Feature | Learned Coefficient | Theoretical Value |
|---------|-------------------|-------------------|
| log(Distance) | -0.499 | -0.5 |
| log(Star Temperature) | 0.828 | 1.0 |
| log(Star Radius) | 0.442 | 0.5 |

**Interpretation**: The model successfully recovered theoretical physics constants from raw data.

## Installation & Usage

### Prerequisites
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Option 1: Interactive Web App (Streamlit)

1. Navigate to the web folder:
```bash
cd web
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Upload `exoplanet_data.csv` via the sidebar

4. Adjust simulator parameters:
   - Orbital Period (1-1000 days)
   - Star Temperature (3000-10000 K)
   - Star Radius (0.1-10 solar radii)

5. View real-time predictions and habitability classification

### Option 2: Jupyter Notebook Analysis

```bash
jupyter notebook exoplanet_surface_temperature_detection.ipynb
```

The notebook includes:
- Detailed exploratory data analysis
- Step-by-step model development
- Coefficient interpretation
- Visualization of results

## Web Application Features

### Interactive Simulator
- Real-time temperature prediction
- Habitability zone classification (180-330 K)
- Distance calculation in AU

### Visualizations
1. **Correlation Heatmap**: Relationships between orbital/stellar properties
2. **Actual vs Predicted Plot**: Model accuracy visualization with color-coded temperatures
3. **Current Simulation Marker**: Highlighted prediction on scatter plot

### Educational Component
- Expandable physics/math section
- Stefan-Boltzmann law explanation
- Kepler's Third Law derivation
- Log transformation rationale

## Technical Stack

**Data Science**:
- `pandas` - Data manipulation
- `numpy` - Numerical operations and log transformations
- `scikit-learn` - Machine learning (Linear Regression, metrics)

**Visualization**:
- `matplotlib` - Static plotting
- `seaborn` - Statistical visualizations
- `streamlit` - Interactive web dashboard

## Key Insights

 **Data Quality Matters**: Removing false positives improved accuracy significantly  
 **Domain Knowledge is Crucial**: Understanding physics enabled correct feature engineering  
 **Simple Models Can Be Powerful**: Linear Regression achieved near-perfect results with proper formulation  
 **Interpretable ML**: Learned coefficients have real physical meaning (white-box model)

## Limitations

- Ignores atmospheric greenhouse effects
- Approximates all host stars as having solar mass
- Limited to Kepler's detection capabilities and field of view
- Equilibrium temperature assumes no atmosphere or albedo effects





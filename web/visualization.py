import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from data_processing import calculate_prediction, get_habitability_status

def render_simulator(model, input_period, input_star_temp, input_star_radius):
    """Render the prediction simulator section."""
    st.markdown("#### Prediction Simulator")
    
    # Calculate prediction
    final_pred, calc_dist = calculate_prediction(
        model, input_period, input_star_temp, input_star_radius
    )
    
    # Determine habitable status
    status_msg, status_color = get_habitability_status(final_pred)
    
    # Display metrics
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    
    with col_sim1:
        st.metric("Equilibrium Temp", f"{final_pred:.0f} K")
    
    with col_sim2:
        st.markdown(f"**Status**")
        st.markdown(f":{status_color}[{status_msg}]")
        
    with col_sim3:
        st.metric("Distance", f"{calc_dist:.2f} AU")
    
    return final_pred, calc_dist

def render_analysis(data, actual, preds, final_pred, X_train):
    """Render the analysis and model performance section."""
    st.markdown("#### Analysis & Model Performance")
    
    # Calculate metrics
    r2 = r2_score(actual, preds)
    mae = mean_absolute_error(actual, preds)
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.caption("Correlation Heatmap")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            data[['Distance', 'Star_Temp', 'Star_Radius', 'Planet_Temp']].corr(), 
            annot=True, 
            cmap='coolwarm', 
            ax=ax1, 
            cbar=False
        )
        st.pyplot(fig1)
    
    with col_g2:
        st.caption("Actual vs Predicted")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        
        # Plot test data
        scatter = ax2.scatter(
            actual, preds, 
            c=actual, 
            cmap='plasma', 
            alpha=0.7, 
            s=20, 
            label='Exoplanets'
        )
        
        ax2.plot(
            [actual.min(), actual.max()], 
            [actual.min(), actual.max()], 
            'r--', 
            lw=1, 
            label='Perfect Fit'
        )
        
        # Current simulation point
        ax2.scatter(
            [final_pred], [final_pred], 
            color='#00ff00', 
            edgecolors='black', 
            marker='*', 
            s=250, 
            zorder=10, 
            label='Current Sim'
        )
        
        ax2.annotate(
            f"{final_pred:.0f} K", 
            (final_pred, final_pred), 
            xytext=(10, -10), 
            textcoords='offset points',
            color='green', 
            fontweight='bold', 
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8)
        )
        
        ax2.set_xlabel("Actual (K)")
        ax2.set_ylabel("Predicted (K)")
        ax2.legend()
        st.pyplot(fig2)
    
    # Display metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.markdown(f"**R² Score:** {r2:.4f}")
    col_m2.markdown(f"**MAE:** {mae:.2f} K")
    col_m3.markdown(f"**Samples:** {len(X_train)}")

def render_physics_section():
    """Render the physics and math explanation section."""
    st.header("The Physics & Math")
    
    with st.expander("Click to see the formulas and logic"):
        st.markdown("""
        ### 1. The Physics: Radiative Equilibrium
        The temperature of a planet is determined by the energy it receives from its star versus the energy it radiates away. 
        Ignoring greenhouse effects, this is governed by the **Stefan-Boltzmann Law**:
        """)
        
        st.latex(r"T_{eq} = T_{*} \sqrt{\frac{R_{*}}{2D}}")
        
        st.markdown("""
        Where:
        * $T_{eq}$ = Planet Equilibrium Temperature
        * $T_{*}$ = Star Temperature
        * $R_{*}$ = Star Radius
        * $D$ = Distance from Star
        """)

        st.markdown("""
        ### 2. Feature Engineering: Kepler's Third Law
        The dataset gives us the **Orbital Period ($P$)**, not the distance. We derive distance using Kepler's Third Law:
        """)
        
        st.latex(r"D \approx P^{2/3}")
        
        st.markdown("Note: Period is converted to years to get Distance in Astronomical Units (AU).")

        st.markdown("""
        ### 3. Log-Log Transformation
        The physics equation is a **Power Law** (curves), but Linear Regression fits **Straight Lines**.
        To fix this, we take the natural logarithm ($\ln$) of both sides:
        """)
        
        st.latex(r"\ln(T_{eq}) = \ln(T_{*}) + 0.5 \ln(R_{*}) - 0.5 \ln(D) + C")
        
        st.markdown(r"""
        Now it looks like a linear equation: $y = \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + C$
        
        The model "learns" these coefficients from the data:
        * $\beta_{star\_temp} \approx 1.0$
        * $\beta_{star\_radius} \approx 0.5$
        * $\beta_{distance} \approx -0.5$
        """)
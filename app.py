import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Alloy Sustainability Calculator",
    page_icon="🌍",
    layout="wide"
)

# --- CONSTANTS & DATA ---
# Atomic masses (g/mol) for the 18 elements
ATOMIC_MASSES = {
    'Al': 26.9815, 'Co': 58.9331, 'Cr': 51.9961, 'Cu': 63.546, 'Fe': 55.845,
    'Hf': 178.49, 'Mn': 54.938, 'Mo': 95.95, 'Nb': 92.906, 'Ni': 58.6934,
    'Re': 186.207, 'Ru': 101.07, 'Si': 28.085, 'Ta': 180.947, 'Ti': 47.867,
    'V': 50.9415, 'W': 183.84, 'Zr': 91.224
}

# Categorization of indicators for better visualization
INDICATOR_CATEGORIES = {
    'Mass price (USD/kg)': 'Economic',
    'Supply risk': 'Economic',
    'Normalized vulnerability to supply restriction': 'Economic',
    'Embodied energy (MJ/kg)': 'Environmental',
    'Rock to metal ratio (kg/kg)': 'Environmental',
    'Water usage (l/kg)': 'Environmental',
    'Human health damage': 'Societal',
    'Human rights pressure': 'Societal',
    'Labor rights pressure': 'Societal'
}

@st.cache_data
def load_data(filepath):
    """Loads and cleans the element indicator data."""
    try:
        df = pd.read_csv(filepath)
        # Rename column to match expected logic if necessary
        df = df.rename(columns={'Raw material price (USD/kg)': 'Mass price (USD/kg)'})
        df = df.set_index('elements')
        return df
    except FileNotFoundError:
        return None

# --- HELPER FUNCTIONS ---
def convert_at_to_wt(composition_at):
    """Converts atomic percentage dictionary to mass fraction series."""
    mass_dict = {}
    total_mass = 0.0
    
    for el, at_pct in composition_at.items():
        mass = at_pct * ATOMIC_MASSES.get(el, 0)
        mass_dict[el] = mass
        total_mass += mass
        
    if total_mass == 0:
        return pd.Series()
    
    # Return mass fractions (0-1 range)
    return pd.Series({k: v / total_mass for k, v in mass_dict.items()})

def calculate_impacts(mass_fractions, data_df):
    """Calculates the 9 indicators based on mass fractions."""
    results = {}
    
    # Align mass fractions with the full 18 elements list (filling missing with 0)
    full_wt_vector = pd.Series(0.0, index=data_df.index)
    full_wt_vector.update(mass_fractions)
    
    # 1. Standard Weighted Average Calculations
    weighted_indicators = [col for col in data_df.columns if col != 'Supply risk']
    for ind in weighted_indicators:
        results[ind] = full_wt_vector.dot(data_df[ind])
        
    # 2. Specific 'Supply Risk' Calculation (from Notebook logic)
    # Formula: 1 - Product(1 - fraction_i * risk_i)
    if 'Supply risk' in data_df.columns:
        risk_vector = data_df['Supply risk']
        risk_contrib = 1 - (full_wt_vector * risk_vector)
        # Check if the product logic is intended or weighted sum. 
        # Using notebook logic: 1 - prod(1 - x*R)
        results['Supply risk'] = 1 - risk_contrib.prod()
        
    return pd.DataFrame([results]).T.reset_index()

# --- MAIN APP LAYOUT ---

st.title("🌍 Alloy Sustainability Calculator")
st.markdown("""
This tool calculates the **economic, environmental, and societal impacts** of your alloy design.  
Enter the composition in **Atomic Percent (%at)** below.
""")

# Load Data from 'data' folder
DATA_FILE = os.path.join("data", "gen_18element_imputed_v202412.csv")
data = load_data(DATA_FILE)

if data is None:
    st.error(f"Error: The data file was not found at `{DATA_FILE}`. Please ensure the 'data' folder exists and contains the CSV file.")
    st.stop()

# Sidebar: Input
st.sidebar.header("1. Define Composition")
st.sidebar.markdown("Select elements and specify their Atomic %.")

available_elements = sorted(list(data.index))
selected_elements = st.sidebar.multiselect("Select Elements", available_elements, default=['Fe', 'Ni', 'Cr'])

composition_at = {}
current_total_at = 0.0

if selected_elements:
    st.sidebar.markdown("---")
    for el in selected_elements:
        # Default value logic to be helpful
        default_val = 100.0 / len(selected_elements)
        val = st.sidebar.number_input(f"{el} (at%)", min_value=0.0, max_value=100.0, value=default_val, step=0.1)
        composition_at[el] = val
        current_total_at += val

    st.sidebar.markdown("---")
    if abs(current_total_at - 100.0) > 0.01:
        st.sidebar.warning(f"Total: {current_total_at:.1f}% (Normalized to 100% for calculation)")
    else:
        st.sidebar.success(f"Total: {current_total_at:.1f}%")
else:
    st.info("Please select at least one element in the sidebar.")
    st.stop()

# Main Logic
mass_fractions = convert_at_to_wt(composition_at)
impacts_df = calculate_impacts(mass_fractions, data)
impacts_df.columns = ['Indicator', 'Value']
impacts_df['Category'] = impacts_df['Indicator'].map(INDICATOR_CATEGORIES)

# Sort for consistent display
impacts_df = impacts_df.sort_values(by=['Category', 'Indicator'])

# --- DISPLAY RESULTS ---

st.header("2. Sustainability Profile")

# Metric Cards
cols = st.columns(3)
categories = ['Economic', 'Environmental', 'Societal']
colors = ['blue', 'green', 'orange']

for i, cat in enumerate(categories):
    with cols[i]:
        st.subheader(f"{cat} Impact")
        cat_data = impacts_df[impacts_df['Category'] == cat]
        for _, row in cat_data.iterrows():
            # Format value nicely
            val = row['Value']
            fmt = "{:.2f}" if val < 100 else "{:.0f}"
            st.metric(label=row['Indicator'].split('(')[0].strip(), value=fmt.format(val))

# Visualization
st.markdown("### Indicator Overview")

fig = px.bar(
    impacts_df, 
    x='Value', 
    y='Indicator', 
    color='Category',
    orientation='h',
    text_auto='.2s',
    title="Sustainability Indicators (Log Scale)",
    color_discrete_map={'Economic': '#1f77b4', 'Environmental': '#2ca02c', 'Societal': '#ff7f0e'}
)

# Use Log scale because Price ($) and Supply Risk (0-1) differ by orders of magnitude
fig.update_layout(
    xaxis_type="log",
    xaxis_title="Value (Log Scale)",
    yaxis_title="",
    height=500,
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# Data Table Expander
with st.expander("View Detailed Data & Mass Composition"):
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Calculated Indicators**")
        st.dataframe(impacts_df.style.format({"Value": "{:.4f}"}))
    with c2:
        st.write("**Mass Fractions (Calculated from at%)**")
        mass_df = mass_fractions.to_frame(name="Mass Fraction")
        mass_df['Mass %'] = mass_df['Mass Fraction'] * 100
        st.dataframe(mass_df.style.format({"Mass Fraction": "{:.4f}", "Mass %": "{:.2f}%"}))
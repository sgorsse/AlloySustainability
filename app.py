import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Alloy Sustainability Calculator",
    page_icon="🌍",
    layout="wide"
)

# --- CONSTANTS ---
# Atomic masses (g/mol)
ATOMIC_MASSES = {
    'Al': 26.9815, 'Co': 58.9331, 'Cr': 51.9961, 'Cu': 63.546, 'Fe': 55.845,
    'Hf': 178.49, 'Mn': 54.938, 'Mo': 95.95, 'Nb': 92.906, 'Ni': 58.6934,
    'Re': 186.207, 'Ru': 101.07, 'Si': 28.085, 'Ta': 180.947, 'Ti': 47.867,
    'V': 50.9415, 'W': 183.84, 'Zr': 91.224
}

# Categorization and Units
INDICATOR_INFO = {
    'Mass price (USD/kg)': {'cat': 'Economic', 'unit': 'USD/kg'},
    'Supply risk': {'cat': 'Economic', 'unit': 'Index (0-1)'},
    'Normalized vulnerability to supply restriction': {'cat': 'Economic', 'unit': 'Index'},
    'Embodied energy (MJ/kg)': {'cat': 'Environmental', 'unit': 'MJ/kg'},
    'Rock to metal ratio (kg/kg)': {'cat': 'Environmental', 'unit': 'kg/kg'},
    'Water usage (l/kg)': {'cat': 'Environmental', 'unit': 'l/kg'},
    'Human health damage': {'cat': 'Societal', 'unit': 'Points'},
    'Human rights pressure': {'cat': 'Societal', 'unit': 'Points'},
    'Labor rights pressure': {'cat': 'Societal', 'unit': 'Points'}
}

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Loads element data and benchmark datasets."""
    data_path = "data"
    
    # 1. Load Elements Data
    try:
        df_elements = pd.read_csv(os.path.join(data_path, "gen_18element_imputed_v202412.csv"))
        df_elements = df_elements.rename(columns={'Raw material price (USD/kg)': 'Mass price (USD/kg)'})
        df_elements = df_elements.set_index('elements')
    except FileNotFoundError:
        st.error("File 'gen_18element_imputed_v202412.csv' not found in data/ folder.")
        return None, None

    # 2. Load Benchmarks
    benchmarks = []
    try:
        # Load Ni/HEAs benchmarks
        df_ni = pd.read_csv(os.path.join(data_path, "gen_HTHEAs_vs_Ni_df.csv"), sep=';')
        benchmarks.append(df_ni)
        
        # Load Fe/HEAs benchmarks
        df_fe = pd.read_csv(os.path.join(data_path, "gen_RTHEAs_vs_Fe_df.csv"), sep=';')
        benchmarks.append(df_fe)
        
        df_bench = pd.concat(benchmarks, ignore_index=True)
        
        # Map disparate column names if necessary (Files seem consistent on indicators based on inspection)
        # Calculate mean values per Class for plotting
        indicators = list(INDICATOR_INFO.keys())
        # Filter only existing columns
        valid_indicators = [col for col in indicators if col in df_bench.columns]
        
        df_bench_grouped = df_bench.groupby('Class')[valid_indicators].mean().reset_index()
        
    except FileNotFoundError:
        st.warning("Benchmark files not found. Comparison will be disabled.")
        df_bench_grouped = None

    return df_elements, df_bench_grouped

# --- LOGIC ---
def parse_formula(formula):
    """Parses chemical formula string (e.g., 'Co20Cr20') into a dictionary."""
    # Regex to find Element and Number. Handles integers and floats.
    # If no number matches, assumes 1.0 (handled by logic below if we wanted, but strict regex here)
    pattern = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    
    composition = {}
    for el, qty in pattern:
        if el not in ATOMIC_MASSES:
            return None, f"Element '{el}' is not supported (only 18 specific elements)."
        
        amount = float(qty) if qty else 1.0
        composition[el] = composition.get(el, 0) + amount
        
    if not composition:
        return None, "Invalid format. Use standard notation like 'Co20Cr20Fe40Ni20'."
        
    return composition, None

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
    
    return pd.Series({k: v / total_mass for k, v in mass_dict.items()})

def calculate_impacts(mass_fractions, data_df):
    """Calculates the indicators based on mass fractions."""
    results = {}
    full_wt_vector = pd.Series(0.0, index=data_df.index)
    full_wt_vector.update(mass_fractions)
    
    for ind in INDICATOR_INFO.keys():
        if ind == 'Supply risk':
            # Specific formula: 1 - Product(1 - fraction * risk)
            risk_vector = data_df['Supply risk']
            risk_contrib = 1 - (full_wt_vector * risk_vector)
            results[ind] = 1 - risk_contrib.prod()
        else:
            # Weighted average
            if ind in data_df.columns:
                results[ind] = full_wt_vector.dot(data_df[ind])
            else:
                results[ind] = 0.0
                
    return results

# --- APP LAYOUT ---
st.title("🌍 Alloy Sustainability Calculator")
st.markdown("Compare the **Economic, Environmental, and Societal impacts** of your alloy against industry standards.")

# Load Data
df_elements, df_benchmarks = load_data()
if df_elements is None:
    st.stop()

# 1. INPUT SECTION
with st.container():
    st.subheader("1. Alloy Composition")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        formula_input = st.text_input(
            "Enter Alloy Formula (Atomic %)", 
            value="Co20Cr20Fe40Ni20",
            help="Example: Co20Cr20Fe40Ni20. Elements must be among: " + ", ".join(sorted(ATOMIC_MASSES.keys()))
        )
    
    # Parse and Calculate
    comp_at, error_msg = parse_formula(formula_input)
    
    if error_msg:
        st.error(error_msg)
        st.stop()
        
    # Validate Total
    total_at = sum(comp_at.values())
    if abs(total_at - 100.0) > 0.1 and total_at != 0:
        st.warning(f"Note: Total atomic % is {total_at:.1f}%. It will be normalized to 100% for calculation.")

    mass_fractions = convert_at_to_wt(comp_at)
    user_impacts = calculate_impacts(mass_fractions, df_elements)

# 2. VISUALIZATION SECTION
st.divider()
st.subheader("2. Sustainability Profile & Benchmarking")

# Prepare Data for Plotting
# User Data
plot_data = []
for ind, val in user_impacts.items():
    meta = INDICATOR_INFO.get(ind, {})
    plot_data.append({
        'Indicator': ind,
        'Value': val,
        'Type': 'Your Alloy',
        'Category': meta.get('cat', 'Other'),
        'Unit': meta.get('unit', '')
    })

# Benchmark Data
if df_benchmarks is not None:
    for _, row in df_benchmarks.iterrows():
        cls_name = row['Class']
        for ind in INDICATOR_INFO.keys():
            if ind in row:
                meta = INDICATOR_INFO.get(ind, {})
                plot_data.append({
                    'Indicator': ind,
                    'Value': row[ind],
                    'Type': cls_name, # e.g., "Steels", "Ni superalloys"
                    'Category': meta.get('cat', 'Other'),
                    'Unit': meta.get('unit', '')
                })

df_plot = pd.DataFrame(plot_data)

# Sort for tidy plotting
df_plot = df_plot.sort_values(by=['Category', 'Indicator'])

# Create Plot
fig = px.bar(
    df_plot,
    x="Value",
    y="Indicator",
    color="Type",
    barmode="group",
    orientation='h',
    text_auto='.2s',
    height=800,
    color_discrete_map={
        "Your Alloy": "#E63946",  # Red for visibility
        "Steels": "#A8DADC",
        "Ni superalloys": "#457B9D",
        "FCC HEAs": "#1D3557",
        "BCC-(R)HEAs & HESAs": "#F1FAEE"
    }
)

fig.update_layout(
    xaxis_type="log",
    xaxis_title="Impact Value (Log Scale)",
    yaxis_title="",
    legend_title="Alloy Class",
    font=dict(size=14),
    margin=dict(l=250) # More space for long labels
)

# Update y-axis labels to include units if desired, or keep them clean
# Adding units to hover text is automatic
st.plotly_chart(fig, use_container_width=True)


# 3. DETAILED METRICS
st.divider()
st.subheader("3. Detailed Metrics")

cols = st.columns(3)
categories = ['Economic', 'Environmental', 'Societal']

for i, cat in enumerate(categories):
    with cols[i]:
        st.markdown(f"#### {cat}")
        # Filter user data for this category
        cat_impacts = {k:v for k,v in user_impacts.items() if INDICATOR_INFO[k]['cat'] == cat}
        
        for name, val in cat_impacts.items():
            unit = INDICATOR_INFO[name]['unit']
            # Clean name for display
            short_name = name.split('(')[0].strip()
            
            # Formatting
            if val < 1:
                fmt_val = f"{val:.3f}"
            elif val < 100:
                fmt_val = f"{val:.1f}"
            else:
                fmt_val = f"{val:.0f}"
                
            st.metric(label=short_name, value=f"{fmt_val} {unit}")

# Footer Data Table
with st.expander("Show Raw Data"):
    st.write("Values for **Your Alloy**:")
    st.dataframe(pd.DataFrame([user_impacts]))

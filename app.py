import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Alloy Sustainability Calculator",
    page_icon="🌍",
    layout="centered"  # Centered layout often looks better for vertical stacked plots
)

# --- CONSTANTS ---
ATOMIC_MASSES = {
    'Al': 26.9815, 'Co': 58.9331, 'Cr': 51.9961, 'Cu': 63.546, 'Fe': 55.845,
    'Hf': 178.49, 'Mn': 54.938, 'Mo': 95.95, 'Nb': 92.906, 'Ni': 58.6934,
    'Re': 186.207, 'Ru': 101.07, 'Si': 28.085, 'Ta': 180.947, 'Ti': 47.867,
    'V': 50.9415, 'W': 183.84, 'Zr': 91.224
}

SUPPORTED_ELEMENTS_STR = ", ".join(sorted(ATOMIC_MASSES.keys()))

# Configuration: Category, Unit, and if Unit should be hidden (dimensionless)
INDICATOR_INFO = {
    'Mass price (USD/kg)': {'cat': 'Economic', 'unit': 'USD/kg'},
    'Supply risk': {'cat': 'Economic', 'unit': ''}, # No unit displayed
    'Normalized vulnerability to supply restriction': {'cat': 'Economic', 'unit': ''}, # No unit displayed
    'Embodied energy (MJ/kg)': {'cat': 'Environmental', 'unit': 'MJ/kg'},
    'Rock to metal ratio (kg/kg)': {'cat': 'Environmental', 'unit': 'kg/kg'},
    'Water usage (l/kg)': {'cat': 'Environmental', 'unit': 'l/kg'},
    'Human health damage': {'cat': 'Societal', 'unit': ''}, # No unit displayed
    'Human rights pressure': {'cat': 'Societal', 'unit': ''}, # No unit displayed
    'Labor rights pressure': {'cat': 'Societal', 'unit': ''}
}

# Colors for the plot
CLASS_COLORS = {
    'BCC-(R)HEAs & HESAs': '#377eb8', # Blue
    'FCC HEAs': '#4daf4a',            # Green
    'Ni superalloys': '#ff7f00',      # Orange
    'Steels': '#984ea3',              # Purple
    'Evaluated alloy': '#e41a1c'      # Red
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
        
        # Calculate Median per Class for plotting (Robust center)
        indicators = list(INDICATOR_INFO.keys())
        valid_indicators = [col for col in indicators if col in df_bench.columns]
        df_bench_grouped = df_bench.groupby('Class')[valid_indicators].median().reset_index()
        
    except FileNotFoundError:
        st.warning("Benchmark files not found. Comparison will be disabled.")
        df_bench_grouped = None

    return df_elements, df_bench_grouped

# --- LOGIC ---
def parse_formula(formula):
    """Parses chemical formula string."""
    pattern = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    composition = {}
    for el, qty in pattern:
        if el not in ATOMIC_MASSES:
            return None, f"Element '{el}' not supported."
        amount = float(qty) if qty else 1.0
        composition[el] = composition.get(el, 0) + amount
        
    if not composition:
        return None, "Invalid format."
    return composition, None

def convert_at_to_wt(composition_at):
    """Converts atomic percentage dictionary to mass fraction series."""
    mass_dict = {}
    total_mass = 0.0
    for el, at_pct in composition_at.items():
        mass = at_pct * ATOMIC_MASSES.get(el, 0)
        mass_dict[el] = mass
        total_mass += mass
    if total_mass == 0: return pd.Series()
    return pd.Series({k: v / total_mass for k, v in mass_dict.items()})

def calculate_impacts(mass_fractions, data_df):
    """Calculates the indicators based on mass fractions."""
    results = {}
    full_wt_vector = pd.Series(0.0, index=data_df.index)
    full_wt_vector.update(mass_fractions)
    
    for ind in INDICATOR_INFO.keys():
        if ind == 'Supply risk':
            risk_vector = data_df['Supply risk']
            risk_contrib = 1 - (full_wt_vector * risk_vector)
            results[ind] = 1 - risk_contrib.prod()
        else:
            if ind in data_df.columns:
                results[ind] = full_wt_vector.dot(data_df[ind])
            else:
                results[ind] = 0.0
    return results

# --- PLOTTING FUNCTION (MATPLOTLIB) ---
def plot_comparison(user_results, df_benchmarks):
    """
    Creates a static Matplotlib figure comparing User Alloy vs Benchmarks.
    Style: Vertical lines for benchmarks, Red Dot for user.
    """
    indicators = list(INDICATOR_INFO.keys())
    
    # Setup Figure
    # Height scales with number of indicators
    fig, axes = plt.subplots(len(indicators), 1, figsize=(10, 12), sharey=False)
    plt.subplots_adjust(hspace=0.6) # Add space between plots
    
    # Legend Data Collection
    legend_handles = {} # Use dict to avoid duplicates

    for idx, indicator in enumerate(indicators):
        ax = axes[idx]
        
        # Aesthetic setup
        ax.set_facecolor('#fafafa')
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_yticks([]) # Hide Y axis
        
        # Remove top/right/left spines for cleanliness
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#dddddd')

        # 1. Plot Benchmarks (Vertical Lines)
        if df_benchmarks is not None:
            for _, row in df_benchmarks.iterrows():
                cls_name = row['Class']
                if indicator in row:
                    val = row[indicator]
                    color = CLASS_COLORS.get(cls_name, 'gray')
                    
                    line = ax.axvline(x=val, color=color, linewidth=4, alpha=0.7, zorder=2)
                    
                    # Store for legend
                    if cls_name not in legend_handles:
                        legend_handles[cls_name] = line

        # 2. Plot User Alloy (Red Dot)
        user_val = user_results.get(indicator, 0)
        scatter = ax.scatter(user_val, 0, color=CLASS_COLORS['Evaluated alloy'], 
                             s=150, edgecolors='white', linewidth=1.5, zorder=5)
        
        if 'Evaluated alloy' not in legend_handles:
             legend_handles['Evaluated alloy'] = scatter

        # 3. Titles and Limits
        unit = INDICATOR_INFO[indicator]['unit']
        unit_str = f" ({unit})" if unit else ""
        
        # Clean Title
        clean_title = indicator.split('(')[0].strip()
        ax.set_title(f"{clean_title}{unit_str}", loc='left', fontsize=11, fontweight='bold', color='#333333')
        
        # Dynamic Limits with padding
        # Collect all points to determine limits
        all_vals = [user_val]
        if df_benchmarks is not None and indicator in df_benchmarks.columns:
            all_vals.extend(df_benchmarks[indicator].dropna().tolist())
        
        if all_vals:
            min_v, max_v = min(all_vals), max(all_vals)
            margin = (max_v - min_v) * 0.1 if max_v != min_v else max_v * 0.1
            ax.set_xlim(min_v - margin, max_v + margin)
            
            # Formatting X-axis tick labels
            if max_v > 1000:
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

    # Global Legend
    # Sort legend to put User First, then Alphabetical Benchmarks
    sorted_labels = ['Evaluated alloy'] + sorted([k for k in legend_handles.keys() if k != 'Evaluated alloy'])
    handles_list = [legend_handles[l] for l in sorted_labels if l in legend_handles]
    
    fig.legend(
        handles=handles_list,
        labels=sorted_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.05), # Position at bottom
        ncol=len(sorted_labels),
        frameon=False,
        fontsize=10
    )
    
    return fig

# --- APP MAIN ---

st.title("🌍 Alloy Sustainability Calculator")

st.markdown(f"""
Compare the **Economic, Environmental, and Societal impacts** of your alloy against industry standards.  
**Compatible elements:** {SUPPORTED_ELEMENTS_STR}
""")

# Load Data
df_elements, df_benchmarks = load_data()
if df_elements is None:
    st.stop()

# --- INPUT SECTION ---
st.markdown("### 1. Define Alloy")
formula_input = st.text_input(
    "Enter Alloy Formula (Atomic %)", 
    value="Co20Cr20Fe40Ni20",
    help="Example: Co20Cr20Fe40Ni20"
)

# Process Input
comp_at, error_msg = parse_formula(formula_input)
if error_msg:
    st.error(error_msg)
    st.stop()

# Calculation
total_at = sum(comp_at.values())
if abs(total_at - 100.0) > 0.1 and total_at != 0:
    st.caption(f"Note: Input total is {total_at:.1f}%. Normalizing to 100%.")

mass_fractions = convert_at_to_wt(comp_at)
user_impacts = calculate_impacts(mass_fractions, df_elements)

# --- RESULTS SECTION ---
st.divider()
st.markdown("### 2. Sustainability Profile")

# Create and Display Plot
if df_benchmarks is not None:
    fig = plot_comparison(user_impacts, df_benchmarks)
    st.pyplot(fig, use_container_width=True)
else:
    st.warning("Benchmarks unavailable, displaying raw numbers only.")

# --- METRICS SECTION (Compact) ---
with st.expander("View Detailed Values"):
    cols = st.columns(3)
    categories = ['Economic', 'Environmental', 'Societal']

    for i, cat in enumerate(categories):
        with cols[i]:
            st.markdown(f"**{cat}**")
            cat_impacts = {k:v for k,v in user_impacts.items() if INDICATOR_INFO[k]['cat'] == cat}
            
            for name, val in cat_impacts.items():
                unit = INDICATOR_INFO[name]['unit']
                # Formatting logic
                if val < 1: fmt = "{:.4f}"
                elif val < 100: fmt = "{:.2f}"
                else: fmt = "{:.0f}"
                
                display_val = fmt.format(val)
                display_unit = f" {unit}" if unit else ""
                
                st.write(f"{name.split('(')[0]}: **{display_val}{display_unit}**")

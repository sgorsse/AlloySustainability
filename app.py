import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import re
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Alloy Sustainability Calculator",
    page_icon="🌍",
    layout="centered"
)

# --- CONSTANTS ---
ATOMIC_MASSES = {
    'Al': 26.9815, 'Co': 58.9331, 'Cr': 51.9961, 'Cu': 63.546, 'Fe': 55.845,
    'Hf': 178.49, 'Mn': 54.938, 'Mo': 95.95, 'Nb': 92.906, 'Ni': 58.6934,
    'Re': 186.207, 'Ru': 101.07, 'Si': 28.085, 'Ta': 180.947, 'Ti': 47.867,
    'V': 50.9415, 'W': 183.84, 'Zr': 91.224
}

SUPPORTED_ELEMENTS_STR = ", ".join(sorted(ATOMIC_MASSES.keys()))

INDICATOR_INFO = {
    'Mass price (USD/kg)': {'cat': 'Economic', 'unit': 'USD/kg'},
    'Supply risk': {'cat': 'Economic', 'unit': ''},
    'Normalized vulnerability to supply restriction': {'cat': 'Economic', 'unit': ''},
    'Embodied energy (MJ/kg)': {'cat': 'Environmental', 'unit': 'MJ/kg'},
    'Rock to metal ratio (kg/kg)': {'cat': 'Environmental', 'unit': 'kg/kg'},
    'Water usage (l/kg)': {'cat': 'Environmental', 'unit': 'l/kg'},
    'Human health damage': {'cat': 'Societal', 'unit': ''},
    'Human rights pressure': {'cat': 'Societal', 'unit': ''},
    'Labor rights pressure': {'cat': 'Societal', 'unit': ''}
}

# Professional Color Palette
CLASS_COLORS = {
    'BCC-(R)HEAs & HESAs': '#4575b4', # Professional Blue
    'FCC HEAs': '#74add1',            # Light Blue
    'Ni superalloys': '#fdae61',      # Soft Orange
    'Steels': '#abd9e9',              # Very Light Blue/Grey
    'Evaluated alloy': '#d73027'      # Strong Red
}

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Loads element data and FULL benchmark datasets (no pre-aggregation)."""
    data_path = "data"
    
    try:
        df_elements = pd.read_csv(os.path.join(data_path, "gen_18element_imputed_v202412.csv"))
        df_elements = df_elements.rename(columns={'Raw material price (USD/kg)': 'Mass price (USD/kg)'})
        df_elements = df_elements.set_index('elements')
    except FileNotFoundError:
        st.error("File 'gen_18element_imputed_v202412.csv' not found.")
        return None, None

    benchmarks = []
    try:
        df_ni = pd.read_csv(os.path.join(data_path, "gen_HTHEAs_vs_Ni_df.csv"), sep=';')
        benchmarks.append(df_ni)
        df_fe = pd.read_csv(os.path.join(data_path, "gen_RTHEAs_vs_Fe_df.csv"), sep=';')
        benchmarks.append(df_fe)
        df_bench = pd.concat(benchmarks, ignore_index=True)
    except FileNotFoundError:
        st.warning("Benchmark files not found. Comparison disabled.")
        df_bench = None

    return df_elements, df_bench

# --- LOGIC ---
def parse_formula(formula):
    pattern = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    composition = {}
    for el, qty in pattern:
        if el not in ATOMIC_MASSES:
            return None, f"Element '{el}' not supported."
        amount = float(qty) if qty else 1.0
        composition[el] = composition.get(el, 0) + amount
    if not composition: return None, "Invalid format."
    return composition, None

def convert_at_to_wt(composition_at):
    mass_dict = {}
    total_mass = 0.0
    for el, at_pct in composition_at.items():
        mass = at_pct * ATOMIC_MASSES.get(el, 0)
        mass_dict[el] = mass
        total_mass += mass
    if total_mass == 0: return pd.Series()
    return pd.Series({k: v / total_mass for k, v in mass_dict.items()})

def calculate_impacts(mass_fractions, data_df):
    results = {}
    full_wt_vector = pd.Series(0.0, index=data_df.index)
    full_wt_vector.update(mass_fractions)
    
    for ind in INDICATOR_INFO.keys():
        if ind == 'Supply risk':
            risk_vector = data_df['Supply risk']
            # Logic: 1 - Prod(1 - risk_i * fraction_i)
            # Note: This is a specific aggregation model provided in user context
            risk_contrib = 1 - (full_wt_vector * risk_vector)
            results[ind] = 1 - risk_contrib.prod()
        else:
            if ind in data_df.columns:
                results[ind] = full_wt_vector.dot(data_df[ind])
            else:
                results[ind] = 0.0
    return results

# --- PLOTTING (HIGH AESTHETIC) ---
def plot_comparison(user_results, df_benchmarks):
    indicators = list(INDICATOR_INFO.keys())
    
    # 1. Setup Figure
    # Taller figure to give breathing room
    fig, axes = plt.subplots(len(indicators), 1, figsize=(9, 14), sharey=False)
    
    # Adjust spacing
    plt.subplots_adjust(hspace=0.7) 
    
    legend_artists = {} # To store handles for custom legend

    for idx, indicator in enumerate(indicators):
        ax = axes[idx]
        
        # 2. Data Preparation for Scaling
        all_values = [user_results.get(indicator, 0)]
        if df_benchmarks is not None and indicator in df_benchmarks.columns:
            # Filter valid data
            valid_series = df_benchmarks[indicator].dropna()
            valid_series = valid_series[valid_series > 0] # Filter 0 for Log scale safety
            if not valid_series.empty:
                all_values.extend(valid_series.tolist())
        
        # Determine Scale (Log if max/min > 50)
        use_log = False
        if len(all_values) > 1:
            vmin, vmax = min(all_values), max(all_values)
            if vmin > 0 and (vmax / vmin > 50):
                use_log = True
        
        # 3. Plot Benchmarks (IQR Band + Median Line)
        if df_benchmarks is not None and indicator in df_benchmarks.columns:
            for cls_name in CLASS_COLORS:
                if cls_name == 'Evaluated alloy': continue
                
                subset = df_benchmarks[df_benchmarks['Class'] == cls_name][indicator].dropna()
                if subset.empty: continue
                
                # Statistics
                median = subset.median()
                q25 = subset.quantile(0.25)
                q75 = subset.quantile(0.75)
                
                color = CLASS_COLORS.get(cls_name, 'grey')
                
                # A. IQR Band (The "Body" of the distribution)
                # We draw a rectangle that spans the height of the plot (ylim 0 to 1)
                # Using axvspan for full vertical coverage within the subplot
                rect = ax.axvspan(q25, q75, ymin=0.1, ymax=0.9, 
                                  color=color, alpha=0.25, lw=0)
                
                # B. Median Line (The "Center")
                line = ax.axvline(median, ymin=0.1, ymax=0.9, 
                                  color=color, linewidth=2.5, alpha=0.9, zorder=2)
                
                if cls_name not in legend_artists:
                    legend_artists[cls_name] = line

        # 4. Plot User Alloy
        user_val = user_results.get(indicator, 0)
        # Add a white halo for contrast
        ax.scatter(user_val, 0.5, s=200, color='white', zorder=4) 
        dot = ax.scatter(user_val, 0.5, s=120, color=CLASS_COLORS['Evaluated alloy'], 
                         marker='o', edgecolors='black', linewidth=1.5, zorder=5)
        
        if 'Evaluated alloy' not in legend_artists:
            legend_artists['Evaluated alloy'] = dot

        # 5. Styling & Axis
        ax.set_ylim(0, 1)
        ax.set_yticks([]) # No Y axis ticks
        
        # Spines
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_color('#333333')
        ax.spines['bottom'].set_linewidth(1)

        # Title formatting
        meta = INDICATOR_INFO[indicator]
        unit_text = f" ({meta['unit']})" if meta['unit'] else ""
        title_text = f"{indicator.split('(')[0].strip()}"
        
        # Left-aligned title inside the plot area or just above
        ax.text(0, 1.05, title_text + unit_text, transform=ax.transAxes, 
                fontsize=11, fontweight='bold', color='#2c3e50', ha='left')

        # Grid
        ax.grid(True, axis='x', color='#e0e0e0', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Apply Scale & Margins
        if use_log:
            ax.set_xscale('log')
            # Custom formatter for cleaner log labels
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        
        # Dynamic x-limits with padding
        if all_values:
            vmin, vmax = min(all_values), max(all_values)
            # Avoid 0 in log
            if use_log and vmin <= 0: vmin = min([v for v in all_values if v > 0] or [0.1])
            
            # Add 10% padding
            if use_log:
                log_span = np.log10(vmax) - np.log10(vmin)
                if log_span == 0: log_span = 1
                ax.set_xlim(10**(np.log10(vmin) - 0.1*log_span), 10**(np.log10(vmax) + 0.1*log_span))
            else:
                span = vmax - vmin
                if span == 0: span = vmax * 0.1 if vmax != 0 else 1.0
                ax.set_xlim(vmin - 0.1*span, vmax + 0.1*span)

    # 6. Global Legend
    # Order: User First, then predefined order
    preferred_order = ['Evaluated alloy', 'Steels', 'Ni superalloys', 'FCC HEAs', 'BCC-(R)HEAs & HESAs']
    
    handles = []
    labels = []
    
    for label in preferred_order:
        if label in legend_artists:
            handles.append(legend_artists[label])
            labels.append(label)
    
    # Legend at the bottom
    fig.legend(handles, labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.06),
               ncol=3, 
               frameon=False, 
               fontsize=10,
               handletextpad=0.5, columnspacing=1.5)

    return fig

# --- APP EXECUTION ---

st.title("🌍 Alloy Sustainability Calculator")
st.markdown(f"**Compatible elements:** {SUPPORTED_ELEMENTS_STR}")
st.markdown("Compare the **Economic, Environmental, and Societal impacts** of your alloy against industry standards.")

df_elements, df_benchmarks = load_data()

if df_elements is None:
    st.stop()

# Input
st.markdown("### 1. Alloy Composition")
c1, c2 = st.columns([3, 1])
with c1:
    formula = st.text_input("Enter Formula (e.g., Co20Cr20Fe40Ni20)", value="Co20Cr20Fe40Ni20")

# Compute
comp, err = parse_formula(formula)
if err:
    st.error(err)
    st.stop()
    
mass_fractions = convert_at_to_wt(comp)
user_results = calculate_impacts(mass_fractions, df_elements)

# Check total
total = sum(comp.values())
if abs(total - 100) > 0.1:
    st.caption(f"Normalized from {total:.1f}% to 100%.")

# Plot
st.markdown("### 2. Sustainability Profile")
if df_benchmarks is not None:
    fig = plot_comparison(user_results, df_benchmarks)
    st.pyplot(fig, use_container_width=True)
else:
    st.warning("Benchmarks not available. Showing raw values.")

# Table
with st.expander("Show Detailed Numbers"):
    st.write("Calculated values for your alloy:")
    # Format nicely
    disp_dict = {}
    for k, v in user_results.items():
        unit = INDICATOR_INFO[k]['unit']
        k_clean = k.split('(')[0].strip()
        disp_dict[k_clean] = f"{v:.4g} {unit}".strip()
    st.dataframe(pd.DataFrame([disp_dict]))

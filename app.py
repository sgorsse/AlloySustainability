import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Alloy Sustainability Calculator",
    page_icon="🌍",
    layout="wide"  # Layout 'Wide' pour profiter de l'espace grille
)

# --- STYLING MATPLOTLIB ---
# Configuration globale pour un rendu "Publication Quality"
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# --- CONSTANTS ---
ATOMIC_MASSES = {
    'Al': 26.9815, 'Co': 58.9331, 'Cr': 51.9961, 'Cu': 63.546, 'Fe': 55.845,
    'Hf': 178.49, 'Mn': 54.938, 'Mo': 95.95, 'Nb': 92.906, 'Ni': 58.6934,
    'Re': 186.207, 'Ru': 101.07, 'Si': 28.085, 'Ta': 180.947, 'Ti': 47.867,
    'V': 50.9415, 'W': 183.84, 'Zr': 91.224
}

SUPPORTED_ELEMENTS_STR = ", ".join(sorted(ATOMIC_MASSES.keys()))

# Structure des indicateurs pour la grille 3x3
INDICATOR_GRID = [
    # Row 1: Economic
    ['Mass price (USD/kg)', 'Supply risk', 'Normalized vulnerability to supply restriction'],
    # Row 2: Environmental
    ['Embodied energy (MJ/kg)', 'Rock to metal ratio (kg/kg)', 'Water usage (l/kg)'],
    # Row 3: Societal
    ['Human health damage', 'Human rights pressure', 'Labor rights pressure']
]

# Meta-data pour l'affichage
INDICATOR_META = {
    'Mass price (USD/kg)': {'unit': 'USD/kg', 'color': '#1f77b4'}, # Blue
    'Supply risk': {'unit': '', 'color': '#1f77b4'},
    'Normalized vulnerability to supply restriction': {'unit': '', 'color': '#1f77b4'},
    'Embodied energy (MJ/kg)': {'unit': 'MJ/kg', 'color': '#2ca02c'}, # Green
    'Rock to metal ratio (kg/kg)': {'unit': 'kg/kg', 'color': '#2ca02c'},
    'Water usage (l/kg)': {'unit': 'l/kg', 'color': '#2ca02c'},
    'Human health damage': {'unit': '', 'color': '#ff7f0e'}, # Orange
    'Human rights pressure': {'unit': '', 'color': '#ff7f0e'},
    'Labor rights pressure': {'unit': '', 'color': '#ff7f0e'}
}

# Couleurs des familles (Palette douce et professionnelle)
CLASS_COLORS = {
    'Steels': '#bdc3c7',              # Light Grey
    'Ni superalloys': '#f39c12',      # Orange
    'FCC HEAs': '#2ecc71',            # Green
    'BCC-(R)HEAs & HESAs': '#3498db'  # Blue
}
USER_COLOR = '#e74c3c' # Red

# --- DATA LOADING ---
@st.cache_data
def load_data():
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
        if el not in ATOMIC_MASSES: return None, f"Element '{el}' not supported."
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
    
    for ind in INDICATOR_META.keys():
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

# --- VISUALIZATION ENGINE ---
def create_dashboard(user_results, df_benchmarks):
    """
    Creates a 3x3 Grid Dashboard with Violin Plots.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    
    # Flatten axes for easy iteration if needed, but we use nested loops here
    
    benchmark_classes = sorted([c for c in CLASS_COLORS.keys() if c in df_benchmarks['Class'].unique()])
    
    for row_idx in range(3):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            indicator = INDICATOR_GRID[row_idx][col_idx]
            meta = INDICATOR_META[indicator]
            
            # 1. Prepare Data
            data_per_class = []
            valid_classes = []
            
            all_vals = [user_results.get(indicator, 0)] # Start with user val for scale
            
            if df_benchmarks is not None and indicator in df_benchmarks.columns:
                for cls in benchmark_classes:
                    subset = df_benchmarks[df_benchmarks['Class'] == cls][indicator].dropna()
                    if not subset.empty:
                        data_per_class.append(subset.values)
                        valid_classes.append(cls)
                        all_vals.extend(subset.values)
            
            # 2. Determine Scale (Log vs Linear)
            # Filter positive values for log check
            pos_vals = [v for v in all_vals if v > 0]
            if not pos_vals: pos_vals = [0.1]
            vmin, vmax = min(pos_vals), max(pos_vals)
            
            use_log = False
            if vmax / vmin > 50:
                use_log = True
                
            # 3. Plot Violins (Benchmarks)
            # Note: Violinplot handles log data poorly natively in matplotlib (KDE issues).
            # We will plot on linear scale but transform data if needed, or simply use boxplot if log is extreme.
            # However, simpler approach for stability: Use Boxplot for technical rigour on log scales, 
            # or Violin on linear data. Given the range (e.g. Price), Log is mandatory.
            # Let's use simple Horizontal Boxplots which are very robust and clean.
            
            # Position classes on Y axis: 0, 1, 2...
            positions = np.arange(len(valid_classes))
            
            if valid_classes:
                # Create the boxplot
                # vert=False -> Horizontal
                bp = ax.boxplot(data_per_class, positions=positions, vert=False, 
                                patch_artist=True, widths=0.6,
                                showfliers=False, # Hide outliers for cleaner look
                                boxprops=dict(linewidth=0), # No border on box fill
                                medianprops=dict(color='white', linewidth=2))
                
                # Color the boxes
                for patch, cls in zip(bp['boxes'], valid_classes):
                    patch.set_facecolor(CLASS_COLORS[cls])
                    patch.set_alpha(0.7)
            
            # 4. Plot User Alloy
            user_val = user_results.get(indicator, 0)
            
            # Add a vertical line for User Value
            # ax.axvline(...) draws across the whole plot
            ax.axvline(user_val, color=USER_COLOR, linewidth=2.5, linestyle='-', zorder=10)
            
            # Add a small text annotation for the user value on top of the line
            # Determine position: slightly above the top box
            top_y = len(valid_classes) - 0.5
            
            # 5. Styling
            ax.set_yticks(positions)
            ax.set_yticklabels(valid_classes)
            
            # Remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#dddddd')
            
            # Grid
            ax.grid(True, axis='x', color='#eeeeee', linestyle='--')
            
            # Title
            # Clean title
            clean_title = indicator.split('(')[0].strip()
            unit_str = f" ({meta['unit']})" if meta['unit'] else ""
            
            # Title color matches category
            ax.set_title(f"{clean_title}{unit_str}", 
                         color=meta['color'], loc='left', pad=10)
            
            # Scale
            if use_log:
                ax.set_xscale('log')
                # Add some padding to x-limits
                if vmin > 0:
                    ax.set_xlim(vmin * 0.5, vmax * 2)
            else:
                 # Add 5% padding
                 margin = (vmax - vmin) * 0.1
                 ax.set_xlim(min(all_vals) - margin, max(all_vals) + margin)

            # Re-enable user line if it was clipped by limits?
            # Usually axvline works regardless, but let's ensure limits cover it
            cur_xlim = ax.get_xlim()
            if user_val < cur_xlim[0] or user_val > cur_xlim[1]:
                # Expand to include user
                new_min = min(cur_xlim[0], user_val * 0.8 if user_val>0 else user_val-0.1)
                new_max = max(cur_xlim[1], user_val * 1.2 if user_val>0 else user_val+0.1)
                ax.set_xlim(new_min, new_max)

    # Global Legend (Manual)
    handles = []
    # Add Benchmarks
    for cls, color in CLASS_COLORS.items():
        if df_benchmarks is not None and cls in df_benchmarks['Class'].unique():
            patch = mpatches.Patch(color=color, label=cls, alpha=0.7)
            handles.append(patch)
    
    # Add User
    user_line = plt.Line2D([0], [0], color=USER_COLOR, linewidth=2.5, label='Your Alloy')
    handles.insert(0, user_line) # Put user first
    
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), 
               bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=12)
    
    # Add global titles for rows? (Economic, Env, Soc)
    # Using text annotations on the figure
    fig.text(0.01, 0.78, "Economic", rotation=90, va='center', ha='left', fontsize=14, fontweight='bold', color='#1f77b4')
    fig.text(0.01, 0.50, "Environment", rotation=90, va='center', ha='left', fontsize=14, fontweight='bold', color='#2ca02c')
    fig.text(0.01, 0.22, "Societal", rotation=90, va='center', ha='left', fontsize=14, fontweight='bold', color='#ff7f0e')

    return fig

# --- APP EXECUTION ---

st.title("🌍 Alloy Sustainability Calculator")
st.markdown(f"**Compatible elements:** {SUPPORTED_ELEMENTS_STR}")
st.markdown("Enter your alloy composition to assess its **Economic, Environmental, and Societal footprint** compared to industry standards.")

df_elements, df_benchmarks = load_data()

if df_elements is None:
    st.stop()

# --- INPUT ---
st.markdown("### 1. Composition Input")
c1, c2 = st.columns([2, 3])
with c1:
    formula = st.text_input("Formula", value="Co20Cr20Fe40Ni20", help="Ex: Al10Co20Cr20Fe40Ni10")
    
    comp, err = parse_formula(formula)
    if err:
        st.error(err)
        st.stop()
        
    mass_fractions = convert_at_to_wt(comp)
    user_results = calculate_impacts(mass_fractions, df_elements)
    
    # Show tiny metrics
    total = sum(comp.values())
    if abs(total - 100) > 0.1:
        st.info(f"Input sum: {total:.1f}% (Normalized to 100%)")

# --- DASHBOARD ---
st.divider()
st.markdown("### 2. Sustainability Dashboard")

if df_benchmarks is not None:
    fig = create_dashboard(user_results, df_benchmarks)
    st.pyplot(fig, use_container_width=True)
else:
    st.warning("Benchmark data missing. Cannot generate dashboard.")

# --- DATA TABLE ---
with st.expander("📊 View Detailed Data Table"):
    # Create a nice dataframe
    rows = []
    for k, v in user_results.items():
        meta = INDICATOR_META[k]
        rows.append({
            "Indicator": k,
            "Your Alloy Value": v,
            "Unit": meta['unit']
        })
    st.dataframe(pd.DataFrame(rows).set_index("Indicator").style.format({"Your Alloy Value": "{:.4g}"}))

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
    layout="wide"
)

# --- DARK MODE STYLING (MATPLOTLIB) ---
# On configure Matplotlib pour qu'il s'adapte au fond sombre
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'text.color': '#fafafa',        # Texte quasi blanc
    'axes.labelcolor': '#fafafa',   # Labels blancs
    'xtick.color': '#fafafa',       # Ticks X blancs
    'ytick.color': '#fafafa',       # Ticks Y blancs
    'axes.edgecolor': '#444444',    # Bordures gris foncé discret
    'axes.facecolor': 'none',       # Fond des axes transparent
    'figure.facecolor': 'none',     # Fond de la figure transparent
    'grid.color': '#444444',        # Grille discrète
    'grid.alpha': 0.5,
    'axes.linewidth': 0.8,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
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

INDICATOR_GRID = [
    ['Mass price (USD/kg)', 'Supply risk', 'Normalized vulnerability to supply restriction'],
    ['Embodied energy (MJ/kg)', 'Rock to metal ratio (kg/kg)', 'Water usage (l/kg)'],
    ['Human health damage', 'Human rights pressure', 'Labor rights pressure']
]

# Couleurs adaptées au Dark Mode (plus lumineuses/pastel)
INDICATOR_META = {
    # Economic: Cyan/Light Blue
    'Mass price (USD/kg)': {'unit': 'USD/kg', 'color': '#4fc3f7'}, 
    'Supply risk': {'unit': '', 'color': '#4fc3f7'},
    'Normalized vulnerability to supply restriction': {'unit': '', 'color': '#4fc3f7'},
    # Environmental: Light Green
    'Embodied energy (MJ/kg)': {'unit': 'MJ/kg', 'color': '#81c784'}, 
    'Rock to metal ratio (kg/kg)': {'unit': 'kg/kg', 'color': '#81c784'},
    'Water usage (l/kg)': {'unit': 'l/kg', 'color': '#81c784'},
    # Societal: Light Orange/Peach
    'Human health damage': {'unit': '', 'color': '#ffb74d'}, 
    'Human rights pressure': {'unit': '', 'color': '#ffb74d'},
    'Labor rights pressure': {'unit': '', 'color': '#ffb74d'}
}

# Couleurs des familles (Palette distincte sur fond sombre)
CLASS_COLORS = {
    'Steels': '#95a5a6',              # Cool Grey
    'Ni superalloys': '#f39c12',      # Bright Orange
    'FCC HEAs': '#2ecc71',            # Bright Green
    'BCC-(R)HEAs & HESAs': '#3498db'  # Bright Blue
}
USER_COLOR = '#ff4b4b' # Streamlit Red (Bright)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    data_path = "data"
    try:
        df_elements = pd.read_csv(os.path.join(data_path, "gen_18element_imputed_v202412.csv"))
        df_elements = df_elements.rename(columns={'Raw material price (USD/kg)': 'Mass price (USD/kg)'})
        df_elements = df_elements.set_index('elements')
    except FileNotFoundError:
        return None, None

    benchmarks = []
    try:
        df_ni = pd.read_csv(os.path.join(data_path, "gen_HTHEAs_vs_Ni_df.csv"), sep=';')
        benchmarks.append(df_ni)
        df_fe = pd.read_csv(os.path.join(data_path, "gen_RTHEAs_vs_Fe_df.csv"), sep=';')
        benchmarks.append(df_fe)
        df_bench = pd.concat(benchmarks, ignore_index=True)
    except FileNotFoundError:
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
    Creates a 3x3 Grid Dashboard optimized for Dark Background.
    """
    # Force transparent figure background
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.patch.set_alpha(0.0)
    
    plt.subplots_adjust(wspace=0.35, hspace=0.6)
    
    benchmark_classes = sorted([c for c in CLASS_COLORS.keys() if c in df_benchmarks['Class'].unique()])
    
    for row_idx in range(3):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            # Force transparent axes background
            ax.patch.set_alpha(0.0)
            
            indicator = INDICATOR_GRID[row_idx][col_idx]
            meta = INDICATOR_META[indicator]
            
            # 1. Prepare Data
            data_per_class = []
            valid_classes = []
            all_vals = [user_results.get(indicator, 0)]
            
            if df_benchmarks is not None and indicator in df_benchmarks.columns:
                for cls in benchmark_classes:
                    subset = df_benchmarks[df_benchmarks['Class'] == cls][indicator].dropna()
                    if not subset.empty:
                        data_per_class.append(subset.values)
                        valid_classes.append(cls)
                        all_vals.extend(subset.values)
            
            # 2. Determine Scale
            pos_vals = [v for v in all_vals if v > 0]
            if not pos_vals: pos_vals = [0.1]
            vmin, vmax = min(pos_vals), max(pos_vals)
            use_log = (vmax / vmin > 50)
                
            # 3. Plot Horizontal Boxplots
            positions = np.arange(len(valid_classes))
            
            if valid_classes:
                bp = ax.boxplot(data_per_class, positions=positions, vert=False, 
                                patch_artist=True, widths=0.6,
                                showfliers=False,
                                # Props for Dark Mode: White edges for boxes
                                boxprops=dict(linewidth=1, color='#fafafa'),
                                whiskerprops=dict(color='#fafafa', linewidth=1),
                                capprops=dict(color='#fafafa', linewidth=1),
                                medianprops=dict(color='white', linewidth=2))
                
                # Fill colors
                for patch, cls in zip(bp['boxes'], valid_classes):
                    patch.set_facecolor(CLASS_COLORS[cls])
                    patch.set_alpha(0.8) # Slightly more opaque for contrast
            
            # 4. Plot User Alloy
            user_val = user_results.get(indicator, 0)
            
            # Vertical line for User
            ax.axvline(user_val, color=USER_COLOR, linewidth=3, linestyle='-', zorder=10)
            
            # 5. Styling
            ax.set_yticks(positions)
            ax.set_yticklabels(valid_classes, color='#eeeeee') # Light grey labels
            
            # Spines: Keep bottom, hide others
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#666666')
            
            # Grid
            ax.grid(True, axis='x', color='#444444', linestyle='--')
            
            # Title
            clean_title = indicator.split('(')[0].strip()
            unit_str = f" ({meta['unit']})" if meta['unit'] else ""
            
            ax.set_title(f"{clean_title}{unit_str}", 
                         color=meta['color'], loc='left', pad=10, fontweight='bold')
            
            # Scale & Limits
            if use_log:
                ax.set_xscale('log')
                if vmin > 0:
                    ax.set_xlim(vmin * 0.5, vmax * 2)
            else:
                 margin = (vmax - vmin) * 0.1
                 ax.set_xlim(min(all_vals) - margin, max(all_vals) + margin)

    # 6. Global Legend
    handles = []
    for cls, color in CLASS_COLORS.items():
        if df_benchmarks is not None and cls in df_benchmarks['Class'].unique():
            patch = mpatches.Patch(color=color, label=cls, alpha=0.8)
            handles.append(patch)
    
    # User Line Legend
    user_line = plt.Line2D([0], [0], color=USER_COLOR, linewidth=3, label='Your Alloy')
    handles.insert(0, user_line)
    
    # Legend Text Color forced to white
    leg = fig.legend(handles=handles, loc='lower center', ncol=len(handles), 
               bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=12)
    for text in leg.get_texts():
        text.set_color('#fafafa')
    
    # Category Labels (Sidebar text)
    fig.text(0.01, 0.78, "Economic", rotation=90, va='center', ha='left', fontsize=14, fontweight='bold', color=INDICATOR_META['Mass price (USD/kg)']['color'])
    fig.text(0.01, 0.50, "Environment", rotation=90, va='center', ha='left', fontsize=14, fontweight='bold', color=INDICATOR_META['Embodied energy (MJ/kg)']['color'])
    fig.text(0.01, 0.22, "Societal", rotation=90, va='center', ha='left', fontsize=14, fontweight='bold', color=INDICATOR_META['Human health damage']['color'])

    return fig

# --- APP EXECUTION ---

st.title("🌍 Alloy Sustainability Calculator")
st.markdown(f"**Compatible elements:** {SUPPORTED_ELEMENTS_STR}")
st.markdown("Enter your alloy composition to assess its **Economic, Environmental, and Societal footprint** compared to industry standards.")

df_elements, df_benchmarks = load_data()

if df_elements is None:
    st.error("Data files not found. Please ensure `data/` folder contains the CSV files.")
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
    # Background transparent is handled in fig creation, but streamlit needs to respect it
    st.pyplot(fig, use_container_width=True)
else:
    st.warning("Benchmark data missing. Cannot generate dashboard.")

# --- DATA TABLE ---
with st.expander("📊 View Detailed Data Table"):
    rows = []
    for k, v in user_results.items():
        meta = INDICATOR_META[k]
        rows.append({
            "Indicator": k,
            "Your Alloy Value": v,
            "Unit": meta['unit']
        })
    st.dataframe(pd.DataFrame(rows).set_index("Indicator").style.format({"Your Alloy Value": "{:.4g}"}))

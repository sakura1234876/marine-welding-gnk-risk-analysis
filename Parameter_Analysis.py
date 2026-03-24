#-------------------------------------------------
# Overlap Count for Different Weight Combinations
#-------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. GLOBAL STYLE CONFIGURATION ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False 

# --- 2. DATA LOADING & CONFIGURATION ---
# Placeholder for the path of your processed data
INPUT_FILE = "max_prob_paths_codes_transition_probability.xlsx"
df = pd.read_excel(INPUT_FILE, sheet_name="three-factor")

# Parameters for sensitivity testing
alphas = np.linspace(0, 1, 11)
top_n = 15
top_sets = {}
baseline_alpha = 0.5

# --- 3. SENSITIVITY CALCULATION ---

# Dynamically calculate the Top 15 paths for each weight combination
for a in alphas:
    a = round(a, 1)
    b = round(1 - a, 1)
    
    # Combined Score = α * Coupling_T + β * Transition_Probability
    # Note: Column names 'Coupling_T', 'Transition_Prob', and 'Factor_Code' 
    # should match your Excel headers.
    current_scores = a * df['Coupling_T'] + b * df['Transition_Prob']
    
    # Identify the unique codes for the Top N paths
    top_indices = current_scores.nlargest(top_n).index
    current_top_paths = set(df.iloc[top_indices]['Factor_Code'])
    top_sets[a] = current_top_paths

# Calculate overlap count compared to the baseline (α=0.5, β=0.5)
baseline_set = top_sets[baseline_alpha]
stability_results = []

for a, current_set in top_sets.items():
    intersection = baseline_set.intersection(current_set)
    stability_results.append({'alpha': a, 'overlap': len(intersection)})

stability_df = pd.DataFrame(stability_results)

# --- 4. VISUALIZATION ---

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# Primary Plot: Overlap Count
ax1.plot(stability_df['alpha'], stability_df['overlap'], 
         marker='D', markersize=8, color='#1b4f72', 
         linestyle='-', linewidth=2, label='Path Overlap Count')

# Baseline vertical line at α = 0.5
ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, 
            label=r'Baseline ($\alpha$=$\beta$=0.5)')

# Annotate points with numerical values
for x, y in zip(stability_df['alpha'], stability_df['overlap']):
    ax1.text(x, y + 0.3, f'{int(y)}', ha='center', va='bottom', 
             fontsize=10, fontweight='bold')

# Configure Primary X-axis: Alpha (Mutual Information Weight)
ax1.set_xlabel(r'Weight of Mutual Information ($\alpha$)', fontsize=12)
ax1.set_ylabel(f'Number of Overlapping Paths (Top {top_n})', fontsize=12)
ax1.set_ylim(0, top_n + 3)
ax1.grid(True, linestyle=':', alpha=0.5)

# Configure Secondary X-axis (Top): Beta (Transition Probability Weight)
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(alphas)
ax2.set_xticklabels([round(1-a, 1) for a in alphas])
ax2.set_xlabel(r'Weight of Transition Probability ($\beta$)', fontsize=12, labelpad=10)

# Legend and Layout
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig('Top15_Overlap_Stability_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()

#-------------------------------------------------
# Parameter Analysis for Different Weight Combinations
#-------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- 1. PLOT STYLE CONFIGURATION ---
# Using standard academic fonts (removing Chinese font requirements)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['axes.unicode_minus'] = False 

# --- 2. DATA LOADING & PREPARATION ---
# Placeholder for the path of your processed probability path data
INPUT_FILE = "max_prob_paths_codes_with_probabilities.xlsx"
df = pd.read_excel(INPUT_FILE, sheet_name="three-factor")

# Step 1: Identify the Top 15 paths based on the 0.5/0.5 baseline for comparison
# Score = 0.5 * Transition_Prob + 0.5 * Coupling_T
df['base_score'] = 0.5 * df['Transition_Prob'] + 0.5 * df['Coupling_T']
top_15_df = df.nlargest(15, 'base_score').copy()

# Weight steps for alpha (0.0 to 1.0)
alphas = np.linspace(0, 1, 11)
x_labels = top_15_df['Factor_Code'].tolist()
x_indices = range(len(x_labels))
t_values = top_15_df['Coupling_T'].tolist()

# --- 3. GLOBAL SCALE CALCULATION ---
# Calculate the global maximum across all possible weight combinations 
# to ensure a unified Y-axis for fair comparison across subplots.
all_possible_scores = []
for a in alphas:
    b = 1 - a
    s = a * top_15_df['Coupling_T'] + b * top_15_df['Transition_Prob']
    all_possible_scores.extend(s.tolist())
all_possible_scores.extend(t_values)
y_limit_max = max(all_possible_scores) * 1.1 # Adding 10% headroom

# --- 4. VISUALIZATION (SUBPLOT GRID) ---
# Create a 4x3 grid (Total 12 slots for 11 alpha steps)
fig, axes = plt.subplots(4, 3, figsize=(18, 22), dpi=300)
axes = axes.flatten()

for i, a in enumerate(alphas):
    # Standardize precision for alpha and beta weights
    a = round(a, 1)
    b = round(1 - a, 1)
    ax = axes[i]
    
    # Weighted Score: α (Weight for Coupling T), β (Weight for Transition Probability)
    weighted_scores = a * top_15_df['Coupling_T'] + b * top_15_df['Transition_Prob'] 
    
    # Plotting original T-Value vs. Weighted G-NK Score
    ax.plot(x_indices, t_values, marker='s', markersize=6, linestyle='--', 
            color='#1f77b4', label='NK-T Value')
    ax.plot(x_indices, weighted_scores, marker='o', markersize=6, linestyle='-', 
            color='#d62728', label=f'G-NK Score (α={a}, β={b})')
    
    # Apply Unified Y-Axis range
    ax.set_ylim(0, y_limit_max) 
    
    # Subplot Formatting
    ax.set_title(f'Weight Balance: α={a}, β={b}', fontsize=12, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Interaction Intensity', fontsize=10)
    ax.legend(fontsize='x-small', loc='upper right', frameon=False)
    ax.grid(True, linestyle=':', alpha=0.6)

# Hide any unused subplots in the grid
for j in range(len(alphas), len(axes)):
    axes[j].axis('off')

# Global Plot Adjustment
plt.tight_layout()
OUTPUT_IMAGE = "Sensitivity_Comparison_Unified_Axis.png"
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
print(f"Sensitivity comparison grid saved to: {OUTPUT_IMAGE}")
plt.show()
#-------------------------------------------------
# NK VS G-NK
#-------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
# --- 1. GLOBAL STYLE CONFIGURATION ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False 

# --- 2. REUSABLE PLOTTING FUNCTION ---

def plot_model_comparison(ax, data, title_label):
    """
    Plots a comparison between NK and G-NK models on a specific axis.
    
    Args:
        ax: The matplotlib axis object to draw on.
        data: A DataFrame containing the top risk paths.
        title_label: A string used for labeling the subplot (e.g., 'Two-Factor').
    """
    x_indices = range(len(data))
    path_labels = data['Factor_Code'].tolist()
    nk_values = data['Coupling_T'].values
    gnk_values = data['Weighted_Result'].values

    # Plot NK Model (Baseline structural coupling)
    ax.plot(x_indices, nk_values, color='black', marker='o', 
            linestyle='-', linewidth=2, label='NK Model')
    
    # Plot G-NK Model (Dynamic risk evolution)
    ax.plot(x_indices, gnk_values, color='red', marker='s', 
            linestyle='--', linewidth=2, label='G-NK Model')

    # Formatting axes and labels
    ax.set_ylabel('Coupling Intensity', fontsize=12)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(path_labels, rotation=45, ha='right', fontsize=10)
    ax.set_title(title_label, fontsize=14, fontweight='bold', pad=10)
    ax.legend(frameon=False)
    ax.grid(True, linestyle=':', alpha=0.5)

# --- 3. DATA INGESTION & EXECUTION ---

def run_comparative_analysis(input_path, output_image):
    """
    Main workflow: Loads data, extracts top paths, and generates side-by-side plots.
    """
    try:
        # Load sheets from the processed Excel file
        # Note: Ensure headers 'Factor_Code', 'Coupling_T', and 'Weighted_Result' match your file
        df_two = pd.read_excel(input_path, sheet_name='two-factor').head(15)
        df_three = pd.read_excel(input_path, sheet_name='three-factor').head(15)

        # Create a 1x2 grid (1 row, 2 columns) for side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

        # Execute plotting for both interaction orders
        plot_model_comparison(axes[0], df_two, 'Interaction Order: Two-Factor')
        plot_model_comparison(axes[1], df_three, 'Interaction Order: Three-Factor')

        # Final layout adjustments
        plt.tight_layout()
        
        # Save and display
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"Comparative visualization saved to: {output_image}")
        plt.show()

    except Exception as e:
        print(f"An error occurred during analysis: {e}")

# --- 4. MAIN ENTRY POINT ---

if __name__ == "__main__":
    # FILE PLACEHOLDERS
    INPUT_FILE = "<YOUR_PROCESSED_DATA_FILE>.xlsx"
    OUTPUT_FILE = "Model_Comparison_Results.png"
    
    run_comparative_analysis(INPUT_FILE, OUTPUT_FILE)
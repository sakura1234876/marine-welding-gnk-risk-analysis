#----------------------------
# Main factor coupling calculator
#----------------------------
import pandas as pd
import numpy as np
from itertools import product, combinations
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION & DATA LOADING ---
# Placeholder for the input file containing categorized risk factors
INPUT_FILE = "<YOUR_INPUT_RISK_DATA_FILE>.xlsx"
df = pd.read_excel(INPUT_FILE, sheet_name='Sheet1')

# Define the risk factor mapping from local terms to English codes
# HU: Human, EN: Environment, EQ: Equipment, MA: Material, TE: Technical/Process, MG: Management
CATEGORY_MAP = {
    'Human Factors': 'HU',
    'Environmental Factors': 'EN',
    'Equipment Factors': 'EQ',
    'Material Factors': 'MA',
    'Technical Factors': 'TE',
    'Management Factors': 'MG'
}

# --- 2. DATA PREPROCESSING (BINARY ENCODING) ---
# Initialize a binary matrix where each column represents a risk category
case_matrix = pd.DataFrame(index=df.index, columns=CATEGORY_MAP.values())

for idx, row in df.iterrows():
    # Assuming 'Factor_Categories' contains semicolon-separated strings
    factors_str = row['Factor_Categories'] 
    factor_list = [f.strip() for f in factors_str.split(';')] if pd.notna(factors_str) else []
    
    # Initialize all categories as 0 (absent) for this case
    case_matrix.loc[idx, :] = 0
    
    # Set to 1 if the factor category was identified in the text
    for factor in factor_list:
        if factor in CATEGORY_MAP:
            case_matrix.loc[idx, CATEGORY_MAP[factor]] = 1

print("Data preprocessing complete!")
print(f"Total number of cases: {len(case_matrix)}")
print("First 5 rows of the binary encoded matrix:")
print(case_matrix.head())

# --- 3. JOINT PROBABILITY DISTRIBUTION (STEP 1) ---
# Generate all 2^6 (64) possible binary combinations for the 6 factors
all_combinations = list(product([0, 1], repeat=6))
total_cases = len(case_matrix)
joint_probs_dict = {}

for comb in all_combinations:
    # Filter matrix rows that exactly match the current binary combination
    count = len(case_matrix[
        (case_matrix['HU'] == comb[0]) &
        (case_matrix['EN'] == comb[1]) &
        (case_matrix['EQ'] == comb[2]) &
        (case_matrix['MA'] == comb[3]) &
        (case_matrix['TE'] == comb[4]) &
        (case_matrix['MG'] == comb[5])
    ])
    prob = count / total_cases
    # Use binary string as key (e.g., '101000')
    key = ''.join(map(str, comb))
    joint_probs_dict[key] = prob

print(f"\nStep 1 Complete: Calculated joint probabilities for {len(joint_probs_dict)} combinations.")

# --- 4. COUPLING DEGREE CALCULATION ENGINE ---

def calculate_coupling_degree_t(factor_indices, full_joint_probs):
    """
    Calculates the Information Coupling Degree (T) for a specified subset of risk factors.
    
    The T-value measures the interdependence between factors using multi-information theory.
    It compares the actual joint probability against the product of individual marginals.
    
    Args:
        factor_indices (list): Indices of factors to analyze (0 to 5).
        full_joint_probs (dict): The 64-state joint probability dictionary.
        
    Returns:
        float: The calculated T-Value.
    """
    t_value = 0.0
    k = len(factor_indices)
    
    # Generate all possible states for the selected subset (2^k states)
    sub_states = list(product([0, 1], repeat=k))

    for states in sub_states:
        # A. Calculate Joint Probability P(selected factors) by marginalizing the full set
        p_joint = 0.0
        for full_key, prob in full_joint_probs.items():
            match = True
            for i, factor_pos in enumerate(factor_indices):
                if int(full_key[factor_pos]) != states[i]:
                    match = False
                    break
            if match:
                p_joint += prob

        if p_joint <= 1e-10:
            continue

        # B. Calculate the product of individual marginal probabilities
        marginal_product = 1.0
        for i, factor_pos in enumerate(factor_indices):
            p_marginal = 0.0
            target_state = states[i]
            # Marginalize for a single factor: P(X_i = state)
            for full_key, prob in full_joint_probs.items():
                if int(full_key[factor_pos]) == target_state:
                    p_marginal += prob
            
            if p_marginal <= 1e-10:
                marginal_product = 0.0
                break
            marginal_product *= p_marginal

        if marginal_product <= 1e-10:
            continue

        # C. Apply the Coupling Formula: T = Sum(P_joint * log2(P_joint / Product_of_Marginals))
        ratio = p_joint / marginal_product
        log_val = np.log2(ratio) if ratio > 0 else 0
        t_value += p_joint * log_val

    return t_value

# --- 5. ITERATIVE ANALYSIS OF FACTOR COMBINATIONS ---
factor_names = ['HU', 'EN', 'EQ', 'MA', 'TE', 'MG']
analysis_results = []

# Analyze all possible levels of coupling (2-factor up to 6-factor)
coupling_levels = [
    ('Six-Factor Coupling', 6),
    ('Five-Factor Coupling', 5),
    ('Four-Factor Coupling', 4),
    ('Triple-Factor Coupling', 3),
    ('Double-Factor Coupling', 2),
    ('Single-Factor Baseline', 1) # Theoretically near 0
]

for desc, k in coupling_levels:
    combs = list(combinations(range(6), k))
    for subset_indices in combs:
        t_val = calculate_coupling_degree_t(list(subset_indices), joint_probs_dict)
        codes = [factor_names[i] for i in subset_indices]
        
        analysis_results.append({
            'Category': desc,
            'Combination_Code': ','.join(codes),
            'T_Value': t_val
        })

# --- 6. REPORTING & EXPORT ---
print("=" * 100)
print(f"{'RISK FACTOR COUPLING ANALYSIS RESULTS':^100}")
print("=" * 100)

# Convert to DataFrame for sorting and export
results_df = pd.DataFrame(analysis_results)
results_df = results_df.sort_values(by='T_Value', ascending=False).reset_index(drop=True)

# Print top 20 most coupled combinations
for idx, row in results_df.head(20).iterrows():
    print(f"{idx+1:2d}. {row['Category']:<25} | {row['Combination_Code']:<20} | T = {row['T_Value']:.6f}")

# Save results to a placeholder path
OUTPUT_FILE = "<YOUR_OUTPUT_COUPLING_RESULTS>.xlsx"
results_df.to_excel(OUTPUT_FILE, index=False)
print("-" * 100)
print(f"Analysis complete. Results saved to: {OUTPUT_FILE}")
print("Note: Higher T-Values indicate stronger coupling and higher accident synergy.")
#----------------------------
# Sub_factor coupling calculator
#----------------------------
import pandas as pd
import numpy as np
from itertools import combinations, product
import warnings
import os
warnings.filterwarnings('ignore')
# Unified mapping for secondary sub-factors to alphanumeric codes
# H: Human, MG: Management, MA: Material, TE: Technical, EN: Environment, EQ: Equipment
CATEGORY_CODE_MAP = {
    "Skill Level": "H1", "Responsibility & Attitude": "H2", "Operational Compliance": "H3",
    "Supervision & Inspection": "MG1", "Systems & Procedures": "MG2", "Training & Briefing": "MG3",
    "Material Quality": "MA1", "Consumable Management": "MA2", "Storage Conditions": "MA3",
    "Process Design": "TE1", "Process Execution": "TE2", "Technical Standards": "TE3",
    "Weather Conditions": "EN1", "Construction Environment": "EN2",
    "Equipment Status": "EQ1", "Tool Usage": "EQ2", "Performance Indicators": "EQ3",
}

#  UTILITY FUNCTIONS ---
def clean_and_split_factors(text: str) -> list:
    """
    Standardizes delimiters (handling both full-width and half-width semicolons)
    and returns a cleaned list of factor strings.
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    # Replace common Asian full-width semicolon '；' with standard ';'
    return [f.strip() for f in text.replace('；', ';').split(';') if f.strip()]

def calculate_coupling_degree_t(matrix: pd.DataFrame, factor_subset: list) -> float:
    """
    Calculates the Information Coupling Degree (T) for a subset of risk factors.
    
    Formula: 
    $$T = \sum P_{joint} \cdot \log_2 \left( \frac{P_{joint}}{\prod P_{marginal}} \right)$$
    
    Args:
        matrix: Binary encoded DataFrame (0 or 1).
        factor_subset: List of column names (e.g., ['H1', 'MG2']).
    """
    if not factor_subset:
        return 0.0
    
    k_order = len(factor_subset)
    total_samples = len(matrix)
    if total_samples == 0:
        return 0.0
    
    t_value = 0.0
    # Iterate through all 2^k binary combinations for the selected subset
    for states in product([0, 1], repeat=k_order):
        # Create a logical mask to find matching rows in the dataset
        mask = pd.Series([True] * total_samples, index=matrix.index)
        for i, col in enumerate(factor_subset):
            mask &= (matrix[col] == states[i])
        
        # Calculate the Joint Probability
        p_joint = mask.sum() / total_samples
        if p_joint <= 1e-12:
            continue
            
        # Calculate the product of individual Marginal Probabilities
        marginal_prod = 1.0
        for i, col in enumerate(factor_subset):
            p_marg = (matrix[col] == states[i]).sum() / total_samples
            if p_marg <= 1e-12:
                marginal_prod = 0.0
                break
            marginal_prod *= p_marg
        
        if marginal_prod > 1e-12:
            # Apply Information Gain / Coupling formula
            t_value += p_joint * np.log2(p_joint / marginal_prod)
            
    return t_value

# COUPLING ANALYSIS (N-K MODEL) ---

def run_coupling_analysis(input_path: str, output_path: str):
    """
    Performs binary encoding and calculates 2nd and 3rd order risk coupling.
    """
    print(f"Loading data for coupling analysis from: {input_path}")
    df = pd.read_excel(input_path)    
    # Initialize Binary Encoding Matrix for all sub-factors
    sub_factors = list(CATEGORY_CODE_MAP.values())
    case_matrix = pd.DataFrame(0, index=df.index, columns=sub_factors, dtype=int)
    for idx, row in df.iterrows():
        # Get factors from the 'Secondary_Factors' column
        factor_list = clean_and_split_factors(row.get('Secondary_Factors', ''))
        for f in factor_list:
            # Note: Ensure the input data uses English category names matching CATEGORY_CODE_MAP
            if f in CATEGORY_CODE_MAP:
                case_matrix.at[idx, CATEGORY_CODE_MAP[f]] = 1

    print(f"Matrix built: {len(case_matrix)} rows, {len(sub_factors)} sub-factors.")
    results = []
    # Analyze 2nd-Order (Pairs) and 3rd-Order (Triples) interactions
    for order in [2, 3]:
        print(f"Computing {order}-Factor coupling interactions...")
        for subset in combinations(sub_factors, order):
            t_val = calculate_coupling_degree_t(case_matrix, list(subset))
            if t_val > 1e-6: # Filter insignificant noise
                results.append({
                    'Order': order, 
                    'Factors': ','.join(subset), 
                    'T_Value': t_val
                })

    # Sort results by Coupling Degree (T) descending
    results_df = pd.DataFrame(results).sort_values('T_Value', ascending=False).reset_index(drop=True)
    results_df.to_excel(output_path, index=False)
    print(f"Analysis results saved successfully to: {output_path}")
#  DATA ENCODING & EXCEL RE-WRITING ---
def run_factor_encoding(file_path: str):
    """
    Reads an Excel file, adds an encoded shorthand column to Sheet1, 
    and preserves all other existing sheets.
    """
    print(f"Processing factor encoding for file: {file_path}")
    
    # Read all sheets into a dictionary to ensure no data is lost
    excel_data = pd.read_excel(file_path, sheet_name=None)
    
    if 'Sheet1' not in excel_data:
        print("Error: 'Sheet1' not found in the target file.")
        return

    main_df = excel_data['Sheet1']

    def encode_string(text):
        """Helper to convert full names into shorthand codes."""
        parts = clean_and_split_factors(text)
        return " ".join([CATEGORY_CODE_MAP[f] for f in parts if f in CATEGORY_CODE_MAP])

    # Add the shorthand code column based on 'Secondary_Factors'
    main_df['Factor_Codes'] = main_df['Secondary_Factors'].apply(encode_string)

    # Re-save the file with all sheets intact
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for sheet_name, df_content in excel_data.items():
            df_content.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print("Encoding successful. Column 'Factor_Codes' added to Sheet1.")
# MAIN EXECUTION ---
if __name__ == "__main__":
    # FILE PLACEHOLDERS
    PATH_FOR_ANALYSIS = "<YOUR_INPUT_FILE_FOR_T_VALUE>.xlsx"
    OUTPUT_ANALYSIS_FILE = "Risk_Coupling_Results.xlsx"
    PATH_FOR_ENCODING = "<YOUR_TRANSITION_PROBABILITY_FILE>.xlsx"

    # Execute Task 1: Risk Interaction (Coupling) Analysis
    if os.path.exists(PATH_FOR_ANALYSIS):
        run_coupling_analysis(PATH_FOR_ANALYSIS, OUTPUT_ANALYSIS_FILE)

    # Execute Task 2: Data Preprocessing (Encoding)
    if os.path.exists(PATH_FOR_ENCODING):
        run_factor_encoding(PATH_FOR_ENCODING)
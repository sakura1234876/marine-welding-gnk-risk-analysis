"""
G-NK Robustness Testing Script
-----------------------------------
Features:
1. G-NK Random Edge Deletion Robustness Test
2. Entity Mapping Noise Injection Robustness Test

Applicable Scenarios:
1. Original nodes in the Neo4j graph are not standard secondary factors (e.g., H1, H2, MG1).
2. Requires using an LLM (Ollama) to map original nodes to standard secondary factors.
3. The mutual information file contains Type / Factors / T_Value columns.
4. "Factors" column contains English secondary factor combinations.
5. Mutual information is undirected, so the script finds the most probable directed path based on transition probabilities.
6. Only cross-primary-category coupling paths are retained.
"""

import os
import re
import json
import time
import copy
import random
import itertools
from collections import defaultdict

import requests
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from scipy.stats import spearmanr, kendalltau

# ============================================================
# 0. Configuration Zone (Replace placeholders before running)
# ============================================================

# Neo4j Configuration
NEO4J_URI = "<your_neo4j_uri_here_e.g._bolt://localhost:8687>"
NEO4J_USER = "<your_neo4j_username_here>"
NEO4J_PASSWORD = "<your_neo4j_password_here>"

# Ollama Configuration
OLLAMA_API_URL = "<your_ollama_api_url_here_e.g._http://127.0.0.1:11434/api/generate>"
MODEL_NAME = "<your_model_name_here_e.g._qwen3:30b-a3b-Instruct>"

# Input/Output File Paths
NODE_MAPPING_CACHE_PATH = "<path_to_node_classification_cache>.json"
FINAL_NODE_MAPPING_PATH = "<path_to_node_mapping_final>.json"
ORIGINAL_RELATION_PATH = "<path_to_original_relations>.csv"
MUTUAL_INFORMATION_PATH = "<path_to_mutual_information>.xlsx"

# Output Prefix
OUTPUT_PREFIX_EDGE = "gnk_edge_deletion"
OUTPUT_PREFIX_NOISE = "gnk_entity_mapping_noise_drop"

# ============================================================
# 1. Secondary Factor Mapping
# ============================================================
category_map = {
    "Skill Level": "H1", 
    "Sense of Responsibility": "H2", 
    "Operational Standards": "H3",
    "Supervision & Inspection": "MG1", 
    "System & Process": "MG2", 
    "Training & Briefing": "MG3",
    "Material Quality": "MA1", 
    "Welding Material Management": "MA2", 
    "Storage Conditions": "MA3",
    "Process Design": "TE1", 
    "Process Execution": "TE2", 
    "Technical Standards": "TE3",
    "Weather Conditions": "EN1", 
    "Construction Environment": "EN2",
    "Equipment Status": "EQ1", 
    "Tool Usage": "EQ2", 
    "Performance Indicators": "EQ3",
}

code_to_name = {v: k for k, v in category_map.items()}

# Fixed sorting to generate undirected factor_keys consistently
code_order = {
    "H1": 1, "H2": 2, "H3": 3,
    "MG1": 4, "MG2": 5, "MG3": 6,
    "MA1": 7, "MA2": 8, "MA3": 9,
    "TE1": 10, "TE2": 11, "TE3": 12,
    "EN1": 13, "EN2": 14,
    "EQ1": 15, "EQ2": 16, "EQ3": 17,
}

def get_primary_category(code):
    """ Extracts the primary category from the secondary code (e.g., H1 -> H, MG1 -> MG). """
    match = re.match(r"[A-Z]+", str(code))
    return match.group(0) if match else str(code)

def make_factor_key(codes):
    """ Generates a unique undirected key by sorting codes (e.g., ["TE2", "H3", "MG1"] -> "H3|MG1|TE2"). """
    sorted_codes = sorted(codes, key=lambda x: code_order.get(x, 999))
    return "|".join(sorted_codes)

def is_valid_cross_category_combination(codes):
    """
    Validates if a factor combination crosses different primary categories.
    Rules: 1. No duplicate secondary factors. 2. No duplicate primary categories.
    """
    codes = list(codes)
    if len(set(codes)) != len(codes): return False
    primary_categories = [get_primary_category(c) for c in codes]
    if len(set(primary_categories)) != len(primary_categories): return False
    return True

def build_primary_to_codes(cat_map):
    """ Builds a mapping from primary category to its secondary codes. """
    primary_to_codes = {}
    for _, code in cat_map.items():
        primary = get_primary_category(code)
        if primary not in primary_to_codes:
            primary_to_codes[primary] = []
        primary_to_codes[primary].append(code)
    return primary_to_codes

PRIMARY_TO_CODES = build_primary_to_codes(category_map)

# ============================================================
# 2. Ollama Classification Prompt
# ============================================================
classification_rules_prompt = """
You are a professional ship welding quality inspector.
Please classify the "problem factor" text provided below into one of the specific secondary sub-factors under the 6 major categories.
Each problem factor can only be assigned to ONE most matching secondary sub-factor. Do NOT assign it to the major category names.

The 6 major categories and their corresponding secondary sub-factors are defined as follows:

1. Personnel Factors:
   - Skill Level: Insufficient skills, poor training results, many new employees, inability to read blueprints, working without a license, etc.
   - Sense of Responsibility: Lack of responsibility, weak quality awareness, poor work attitude, blind construction, inertial thinking, etc.
   - Operational Standards: Violations of operations, not constructing according to blueprints, not following process requirements, improper operation, violating welding sequence, etc.

2. Equipment Factors:
   - Equipment Status: Abnormal equipment, functional damage, missing parts, insufficient maintenance, damaged insulation cables, etc.
   - Tool Usage: Improper use of equipment, unqualified tools, insufficient protective measures, welding machine failure, etc.
   - Performance Indicators: Abnormal current/voltage, excessive walking speed, insufficient heating temperature, etc.

3. Material Factors:
   - Material Quality: Unqualified materials, raw materials not meeting requirements, excessive hardness of pressing strips, skipping grades of steel plates, etc.
   - Welding Material Management: Welding material issues, rusty welding wire, damp welding rods, incorrect welding wire type used, etc.
   - Storage Conditions: Improper temporary storage of welding materials, insufficient temperature in insulation buckets, damp/rusty welding materials, etc.

4. Process Factors:
   - Process Design: Design defects, imperfect processes, groove problems, failure to develop targeted processes, etc.
   - Process Execution: Improper parameters, wrong methods, chaotic sequence, insufficient pretreatment, excessive welding speed, etc.
   - Technical Standards: Unclear standards, mismatched specifications, insufficient weld leg, out-of-tolerance groove gaps, etc.

5. Environmental Factors:
   - Weather Conditions: Weather impacts, excessively low temperatures, high humidity, rainy weather, plum rain season, etc.
   - Construction Environment: Narrow spaces, high construction difficulty, poor environmental conditions, high winds, insufficient lighting, etc.

6. Management Factors:
   - Supervision & Inspection: Inadequate inspection, lack of mutual inspection, absence of supervision, failure to discover problems in time, inadequate patrols, etc.
   - System & Process: Imperfect systems, unimplemented requirements, non-standard processes, lax transfer control, improper dispatching, etc.
   - Training & Briefing: Insufficient training, unclear briefings, requirements not communicated, unclear standards, no pre-job training, etc.

Classification Rules:
- ONLY return the exact name of the listed secondary sub-factor.
- NEVER return the major category name.
- Each problem factor is assigned to exactly ONE most matching secondary sub-factor.
- If it cannot clearly correspond to any secondary sub-factor, classify it as "Unclassified".
- The return result MUST be in strict JSON format. Do not include explanations, comments, or extra text.

Please strictly follow the JSON format below:
{
  "Secondary Factor": ["Sub-factor 1"]
}
"""

# ============================================================
# 3. Data Processing & API Functions
# ============================================================
def get_all_relations_from_neo4j():
    """ Fetches all LEADS_TO relations from Neo4j. """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("""
            MATCH (a)-[r:LEADS_TO]->(b)
            RETURN a.name AS source_node,
                   b.name AS target_node,
                   count(r) AS count
        """)
        rows = [{"source_node": rec["source_node"], "target_node": rec["target_node"], "count": int(rec["count"])} for rec in result]
    driver.close()
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No LEADS_TO relations found in Neo4j. Please verify relation type.")
    return df

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def classify_factor_with_ollama(factor_string):
    """ Uses Ollama to classify raw graph nodes into secondary factors. """
    prompt = f"{classification_rules_prompt}\n\nProblem Factor: {factor_string}"
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 300}
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=data, timeout=120)
        response.raise_for_status()
        result = json.loads(response.json().get("response", "").strip())
        subcategories = result.get("Secondary Factor", [])
        if not subcategories: return None
        first = str(subcategories[0]).strip()
        if first == "Unclassified" or first not in category_map: return None
        return first
    except Exception as e:
        print(f"[Classification Failed] Node: {factor_string}; Error: {e}")
        return None

def normalize_mapping_item(value):
    """ Ensures backward compatibility with cached node classifications. """
    if value is None:
        return {"secondary_name": None, "secondary_code": None}
    if isinstance(value, dict):
        sec_name = value.get("secondary_name")
        sec_code = value.get("secondary_code")
        if sec_code in code_to_name:
            return {"secondary_name": code_to_name[sec_code], "secondary_code": sec_code}
        if sec_name in category_map:
            return {"secondary_name": sec_name, "secondary_code": category_map[sec_name]}
        return {"secondary_name": None, "secondary_code": None}
    value = str(value).strip()
    if value in category_map:
        return {"secondary_name": value, "secondary_code": category_map[value]}
    if value in code_to_name:
        return {"secondary_name": code_to_name[value], "secondary_code": value}
    return {"secondary_name": None, "secondary_code": None}

def build_node_mapping(raw_relation_df, sleep_seconds=0.5):
    """ Classifies unique original nodes and caches the mapping. """
    cache = load_json(NODE_MAPPING_CACHE_PATH)
    normalized_cache = {k: normalize_mapping_item(v) for k, v in cache.items()}
    cache = normalized_cache

    all_nodes = sorted(set(raw_relation_df["source_node"].dropna().astype(str)) | 
                       set(raw_relation_df["target_node"].dropna().astype(str)))
    mapping = {}

    for i, node in enumerate(all_nodes, start=1):
        if node in cache:
            mapping[node] = cache[node]
            continue
        print(f"[{i}/{len(all_nodes)}] Classifying node: {node}")
        secondary_name = classify_factor_with_ollama(node)
        item = {"secondary_name": secondary_name, "secondary_code": category_map.get(secondary_name)} if secondary_name else {"secondary_name": None, "secondary_code": None}
        cache[node] = item
        mapping[node] = item
        save_json(cache, NODE_MAPPING_CACHE_PATH)
        time.sleep(sleep_seconds)

    save_json(mapping, FINAL_NODE_MAPPING_PATH)
    return mapping

def map_relations_to_secondary_codes(raw_relation_df, node_mapping):
    """ Maps raw entity relations to standard secondary factor coded relations. """
    mapped_rows, log_rows = [], []
    for _, row in raw_relation_df.iterrows():
        source_node, target_node, cnt = str(row["source_node"]), str(row["target_node"]), int(row["count"])
        source_info = normalize_mapping_item(node_mapping.get(source_node))
        target_info = normalize_mapping_item(node_mapping.get(target_node))
        
        s_code, t_code = source_info.get("secondary_code"), target_info.get("secondary_code")
        status = "Success"
        
        if s_code is None or t_code is None: status = "Skipped_unclassified"
        elif s_code == t_code: status = "Skipped_self_loop"

        log_rows.append({
            "original_source": source_node, "original_target": target_node, "original_count": cnt,
            "mapped_source_code": s_code, "mapped_target_code": t_code,
            "mapped_source_primary": get_primary_category(s_code) if s_code else None,
            "mapped_target_primary": get_primary_category(t_code) if t_code else None,
            "status": status
        })
        
        if status == "Success":
            mapped_rows.append({"source_code": s_code, "target_code": t_code, "count": cnt})

    mapped_df = pd.DataFrame(mapped_rows)
    if not mapped_df.empty:
        mapped_df = mapped_df.groupby(["source_code", "target_code"], as_index=False)["count"].sum()
    return mapped_df, pd.DataFrame(log_rows)

def split_factor_names(factors_str):
    if pd.isna(factors_str): return []
    s = re.sub(r"[，,、;；\s]+", ",", str(factors_str).strip())
    return [p.strip() for p in s.split(",") if p.strip()]

def encode_factor_names(factor_names):
    codes, unknown = [], []
    for name in [str(n).strip() for n in factor_names]:
        if name in category_map: codes.append(category_map[name])
        elif name in code_to_name: codes.append(name)
        else: unknown.append(name)
    return codes, unknown

def load_mutual_information_candidates(mutual_info_path, min_factor_num=2, max_factor_num=3):
    ext = os.path.splitext(mutual_info_path)[1].lower()
    df = pd.read_excel(mutual_info_path) if ext in [".xlsx", ".xls"] else pd.read_csv(mutual_info_path)
    df.columns = [str(c).strip() for c in df.columns]

    missing = {"Type", "Factors", "T_Value"} - set(df.columns)
    if missing: raise ValueError(f"Mutual info file missing columns: {missing}")

    rows, skipped_rows = [], []
    for idx, row in df.iterrows():
        type_value, factors_raw = str(row["Type"]).strip(), str(row["Factors"]).strip()
        try: t_value = float(row["T_Value"])
        except Exception:
            skipped_rows.append({"row_index": idx, "Type": type_value, "Factors": factors_raw, "status": "Skipped_invalid_T_Value"})
            continue
        
        factor_names = split_factor_names(factors_raw)
        factor_codes, unknown = encode_factor_names(factor_names)
        status = "Success"

        if unknown: status = f"Skipped_unknown_factor: {unknown}"
        elif not (min_factor_num <= len(factor_codes) <= max_factor_num): status = f"Skipped_factor_number_{len(factor_codes)}"
        elif not is_valid_cross_category_combination(factor_codes): status = "Skipped_same_primary_category"

        primary_categories = [get_primary_category(c) for c in factor_codes]
        if status != "Success":
            skipped_rows.append({
                "row_index": idx, "Type": type_value, "Factors": factors_raw, "T_Value": t_value,
                "parsed_names": ";".join(factor_names), "parsed_codes": ";".join(factor_codes),
                "primary_categories": ";".join(primary_categories), "status": status
            })
            continue

        rows.append({
            "type": type_value, "factors_raw": factors_raw,
            "factor_names": ";".join(factor_names), "factor_codes": ";".join(factor_codes),
            "primary_categories": ";".join(primary_categories), "factor_key": make_factor_key(factor_codes),
            "T_Value": t_value
        })

    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty: raise ValueError("No valid mutual info candidates. Check filter rules.")
    candidate_df = candidate_df.sort_values("T_Value", ascending=False).drop_duplicates(subset=["factor_key"], keep="first").reset_index(drop=True)

    print(f"Valid MI combinations: {len(candidate_df)}")
    print(f"Filtered MI combinations: {len(skipped_rows)}")
    return candidate_df, pd.DataFrame(skipped_rows)

def compute_transition_probability_for_mi_candidates(mapped_df, mi_candidate_df):
    """ Calculates maximum directed transition probability for each undirected MI combination. """
    freq_dict = defaultdict(lambda: defaultdict(int))
    if mapped_df is not None and not mapped_df.empty:
        for _, row in mapped_df.iterrows():
            freq_dict[row["source_code"]][row["target_code"]] += int(row["count"])
            
    direct_prob = {}
    for source, targets in freq_dict.items():
        total = sum(targets.values())
        direct_prob[source] = {t: c / total for t, c in targets.items()} if total > 0 else {}

    def get_best_path(factor_codes):
        best_path, best_prob = "", 0.0
        for perm in itertools.permutations(factor_codes):
            prob = 1.0
            for i in range(len(perm) - 1):
                edge_prob = direct_prob.get(perm[i], {}).get(perm[i + 1], 0.0)
                if edge_prob <= 0: prob = 0.0; break
                prob *= edge_prob
            if prob > best_prob:
                best_prob, best_path = prob, "→".join(perm)
        return best_path, float(best_prob)

    rows = []
    for _, row in mi_candidate_df.iterrows():
        best_path, best_prob = get_best_path(str(row["factor_codes"]).split(";"))
        r = row.to_dict()
        r.update({"max_path": best_path, "transition_probability": best_prob})
        rows.append(r)
    return pd.DataFrame(rows)

def compute_gnk_ranking_from_mi_and_transition(transition_df, alpha=0.5, beta=0.5):
    """ Computes G-NK combined coupling strength: C = alpha * T_Value + beta * P """
    df = transition_df.copy()
    df["combined_coupling_strength"] = alpha * df["T_Value"].astype(float) + beta * df["transition_probability"].astype(float)
    df = df.sort_values("combined_coupling_strength", ascending=False).reset_index(drop=True)
    df["ranking"] = np.arange(1, len(df) + 1)
    return df

# ============================================================
# 4. Metrics & Utility Functions
# ============================================================
def topk_list(df, col, k=10, drop_empty=True):
    vals = df.head(k)[col].astype(str).tolist()
    return [v for v in vals if v and v.lower() != "nan"] if drop_empty else vals

def jaccard_at_k(list_a, list_b):
    A, B = set(list_a), set(list_b)
    return len(A & B) / len(A | B) if len(A | B) > 0 else 0.0

def rank_correlation_on_baseline_topk(baseline_df, perturbed_df, k=10):
    base_top = baseline_df.head(k)
    pert_score_dict = dict(zip(perturbed_df["factor_key"], perturbed_df["combined_coupling_strength"]))
    x = [float(row["combined_coupling_strength"]) for _, row in base_top.iterrows()]
    y = [float(pert_score_dict.get(row["factor_key"], 0.0)) for _, row in base_top.iterrows()]
    
    sp = spearmanr(x, y).correlation
    kt = kendalltau(x, y).correlation
    return float(sp) if not np.isnan(sp) else 0.0, float(kt) if not np.isnan(kt) else 0.0

# ============================================================
# 5. Robustness Test: Edge Deletion
# ============================================================
def perturb_raw_relations_by_count_thinning(raw_relation_df, deletion_ratio, seed=None):
    """ Simulates edge instance deletion using binomial sampling. """
    rng = np.random.default_rng(seed)
    df = raw_relation_df.copy()
    df["count"] = [rng.binomial(n=cnt, p=1.0 - deletion_ratio) for cnt in df["count"].astype(int)]
    return df[df["count"] > 0].reset_index(drop=True)

def run_gnk_edge_deletion_robustness(raw_relation_df, node_mapping, mi_path, ratios=(0.05, 0.10, 0.15, 0.20), repeats=50, top_k=10, alpha=0.5, beta=0.5):
    print("\n--- Starting Edge Deletion Robustness Test ---")
    mi_cand_df, _ = load_mutual_information_candidates(mi_path)
    base_map_df, _ = map_relations_to_secondary_codes(raw_relation_df, node_mapping)
    base_trans_df = compute_transition_probability_for_mi_candidates(base_map_df, mi_cand_df)
    base_gnk_df = compute_gnk_ranking_from_mi_and_transition(base_trans_df, alpha, beta)

    print("\nBaseline Top-K G-NK:")
    print(base_gnk_df.head(top_k))
    
    detail_rows = []
    for ratio in ratios:
        print(f"\nProcessing Edge Deletion Ratio: {ratio}")
        for r in range(repeats):
            current_seed = 42 + int(ratio * 10000) + r
            pert_raw_df = perturb_raw_relations_by_count_thinning(raw_relation_df, ratio, current_seed)
            pert_map_df, _ = map_relations_to_secondary_codes(pert_raw_df, node_mapping)
            pert_trans_df = compute_transition_probability_for_mi_candidates(pert_map_df, mi_cand_df)
            pert_gnk_df = compute_gnk_ranking_from_mi_and_transition(pert_trans_df, alpha, beta)

            base_factors = topk_list(base_gnk_df, "factor_key", top_k)
            pert_factors = topk_list(pert_gnk_df, "factor_key", top_k)
            base_paths = topk_list(base_gnk_df, "max_path", top_k)
            pert_paths = topk_list(pert_gnk_df, "max_path", top_k)

            sp, kt = rank_correlation_on_baseline_topk(base_gnk_df, pert_gnk_df, top_k)
            base_mean_c = base_gnk_df.head(top_k)["combined_coupling_strength"].mean()
            pert_mean_c = pert_gnk_df.head(top_k)["combined_coupling_strength"].mean()

            detail_rows.append({
                "deletion_ratio": ratio, "repeat": r + 1,
                "raw_relation_count_after": int(pert_raw_df["count"].sum()),
                "mapped_secondary_edges_after": len(pert_map_df),
                "topk_factor_jaccard": jaccard_at_k(base_factors, pert_factors),
                "topk_path_jaccard": jaccard_at_k(base_paths, pert_paths),
                "spearman_on_baseline_topk": sp, "kendall_on_baseline_topk": kt,
                "delta_topk_mean_C": pert_mean_c - base_mean_c
            })

    detail_df = pd.DataFrame(detail_rows)
    summary_df = detail_df.groupby("deletion_ratio").agg(["mean", "std"])
    detail_df.to_csv(f"{OUTPUT_PREFIX_EDGE}_detail.csv", index=False)
    summary_df.to_csv(f"{OUTPUT_PREFIX_EDGE}_summary.csv")
    print("Edge Deletion Tests Completed.")

# ============================================================
# 6. Robustness Test: Entity Mapping Noise
# ============================================================
def perturb_node_mapping(node_mapping, noise_ratio, mode="same_primary_replace", seed=None):
    rng = random.Random(seed)
    noisy_mapping = {k: normalize_mapping_item(v) for k, v in node_mapping.items()}
    classified_nodes = [n for n, item in noisy_mapping.items() if item.get("secondary_code") is not None]
    
    n_perturb = min(int(round(len(classified_nodes) * noise_ratio)), len(classified_nodes))
    if n_perturb <= 0: return noisy_mapping, pd.DataFrame()

    selected_nodes = rng.sample(classified_nodes, n_perturb)
    log_rows = []

    for node in selected_nodes:
        old_code = noisy_mapping[node].get("secondary_code")
        old_primary = get_primary_category(old_code)
        
        new_code, new_name = old_code, noisy_mapping[node].get("secondary_name")
        if mode == "drop":
            new_code, new_name = None, None
        elif mode == "same_primary_replace":
            cands = [c for c in PRIMARY_TO_CODES.get(old_primary, []) if c != old_code]
            if cands: new_code = rng.choice(cands); new_name = code_to_name[new_code]
            else: new_code, new_name = None, None

        noisy_mapping[node] = {"secondary_name": new_name, "secondary_code": new_code}
        log_rows.append({"node": node, "old_code": old_code, "new_code": new_code, "mode": mode})

    return noisy_mapping, pd.DataFrame(log_rows)

def run_gnk_entity_mapping_noise_robustness(raw_relation_df, node_mapping, mi_path, ratios=(0.05, 0.10, 0.15, 0.20), repeats=50, top_k=10, alpha=0.5, beta=0.5, noise_mode="drop"):
    print(f"\n--- Starting Entity Mapping Noise Test (Mode: {noise_mode}) ---")
    mi_cand_df, _ = load_mutual_information_candidates(mi_path)
    base_map_df, _ = map_relations_to_secondary_codes(raw_relation_df, node_mapping)
    base_trans_df = compute_transition_probability_for_mi_candidates(base_map_df, mi_cand_df)
    base_gnk_df = compute_gnk_ranking_from_mi_and_transition(base_trans_df, alpha, beta)

    detail_rows = []
    for ratio in ratios:
        print(f"\nProcessing Noise Ratio: {ratio}")
        for r in range(repeats):
            current_seed = 42 + int(ratio * 10000) + r
            noisy_map, noise_log = perturb_node_mapping(node_mapping, ratio, noise_mode, current_seed)
            pert_map_df, _ = map_relations_to_secondary_codes(raw_relation_df, noisy_map)
            pert_trans_df = compute_transition_probability_for_mi_candidates(pert_map_df, mi_cand_df)
            pert_gnk_df = compute_gnk_ranking_from_mi_and_transition(pert_trans_df, alpha, beta)

            base_factors = topk_list(base_gnk_df, "factor_key", top_k)
            pert_factors = topk_list(pert_gnk_df, "factor_key", top_k)
            base_paths = topk_list(base_gnk_df, "max_path", top_k)
            pert_paths = topk_list(pert_gnk_df, "max_path", top_k)

            sp, kt = rank_correlation_on_baseline_topk(base_gnk_df, pert_gnk_df, top_k)
            base_mean_c = base_gnk_df.head(top_k)["combined_coupling_strength"].mean()
            pert_mean_c = pert_gnk_df.head(top_k)["combined_coupling_strength"].mean()

            detail_rows.append({
                "noise_ratio": ratio, "repeat": r + 1,
                "mapped_secondary_edges_after_noise": len(pert_map_df),
                "noised_node_count": len(noise_log),
                "topk_factor_jaccard": jaccard_at_k(base_factors, pert_factors),
                "topk_path_jaccard": jaccard_at_k(base_paths, pert_paths),
                "spearman_on_baseline_topk": sp, "kendall_on_baseline_topk": kt,
                "delta_topk_mean_C": pert_mean_c - base_mean_c
            })

    detail_df = pd.DataFrame(detail_rows)
    summary_df = detail_df.groupby("noise_ratio").agg(["mean", "std"])
    detail_df.to_csv(f"{OUTPUT_PREFIX_NOISE}_detail.csv", index=False)
    summary_df.to_csv(f"{OUTPUT_PREFIX_NOISE}_summary.csv")

    # Generate Paper Formatting
    summary_df.columns = [f"{m}_{s}" for m, s in summary_df.columns]
    summary_df = summary_df.reset_index()
    
    def format_paper(mean_val, std_val, digits=3):
        return f"{mean_val:.{digits}f} ± NA" if pd.isna(std_val) else f"{mean_val:.{digits}f} ± {std_val:.{digits}f}"

    paper_df = pd.DataFrame({
        "Noise ratio": summary_df["noise_ratio"],
        "Top-10 factor Jaccard": summary_df.apply(lambda r: format_paper(r["topk_factor_jaccard_mean"], r["topk_factor_jaccard_std"]), axis=1),
        "Top-10 path Jaccard": summary_df.apply(lambda r: format_paper(r["topk_path_jaccard_mean"], r["topk_path_jaccard_std"]), axis=1),
        "Spearman": summary_df.apply(lambda r: format_paper(r["spearman_on_baseline_topk_mean"], r["spearman_on_baseline_topk_std"]), axis=1),
        "Kendall": summary_df.apply(lambda r: format_paper(r["kendall_on_baseline_topk_mean"], r["kendall_on_baseline_topk_std"]), axis=1),
        "ΔMean C": summary_df.apply(lambda r: format_paper(r["delta_topk_mean_C_mean"], r["delta_topk_mean_C_std"], 4), axis=1)
    })
    paper_df.to_csv(f"{OUTPUT_PREFIX_NOISE}_paper_summary.csv", index=False)
    print("Entity Mapping Noise Tests Completed.")
    print(paper_df)

# ============================================================
# 7. Main Execution Flow
# ============================================================
if __name__ == "__main__":
    
    # 1. Fetch or Load Original Relations
    if os.path.exists(ORIGINAL_RELATION_PATH):
        print(f"Detected {ORIGINAL_RELATION_PATH}, loading relationships directly.")
        raw_relation_df = pd.read_csv(ORIGINAL_RELATION_PATH)
    else:
        print(f"{ORIGINAL_RELATION_PATH} not found, reading relations from Neo4j.")
        raw_relation_df = get_all_relations_from_neo4j()
        raw_relation_df.to_csv(ORIGINAL_RELATION_PATH, index=False, encoding="utf-8-sig")

    raw_relation_df["source_node"] = raw_relation_df["source_node"].astype(str)
    raw_relation_df["target_node"] = raw_relation_df["target_node"].astype(str)
    raw_relation_df["count"] = raw_relation_df["count"].astype(int)

    # 2. Build or Load Node Mapping Classification Cache
    node_mapping = build_node_mapping(raw_relation_df, sleep_seconds=0.5)

    # 3. Execute Edge Deletion Robustness Test
    run_gnk_edge_deletion_robustness(
        raw_relation_df=raw_relation_df,
        node_mapping=node_mapping,
        mi_path=MUTUAL_INFORMATION_PATH,
        ratios=(0.05, 0.10, 0.15, 0.20),
        repeats=50, top_k=10, alpha=0.5, beta=0.5
    )

    # 4. Execute Entity Mapping Noise Robustness Test (Drop Mode)
    run_gnk_entity_mapping_noise_robustness(
        raw_relation_df=raw_relation_df,
        node_mapping=node_mapping,
        mi_path=MUTUAL_INFORMATION_PATH,
        ratios=(0.05, 0.10, 0.15, 0.20),
        repeats=50, top_k=10, alpha=0.5, beta=0.5,
        noise_mode="drop"
    )

    print("\nAll pipeline tests executed and saved successfully!")
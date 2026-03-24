import pandas as pd
import time
import requests
from collections import defaultdict
from neo4j import GraphDatabase
import json
# Replace placeholders with your actual Neo4j credentials
# bolt://localhost:7687 is the default for local instances
driver = GraphDatabase.driver(
    "bolt://<YOUR_NEO4J_HOST>:<PORT>", 
    auth=("<YOUR_NEO4J_USERNAME>", "<YOUR_NEO4J_PASSWORD>")
)
def get_all_relations():
    """
    Fetches all directional 'LEADS_TO' relationships from the Neo4j Graph.
    
    Returns:
        list: A list of tuples containing (source_node, target_node, occurrence_count).
    """
    with driver.session() as session:
        # Cypher query to find cause-and-effect chains and their frequencies
        result = session.run("""
            MATCH (a)-[r:LEADS_TO]->(b)
            RETURN a.name AS source, b.name AS target, count(r) AS count
        """)
        return [(record["source"], record["target"], record["count"]) for record in result]


#  OLLAMA LLM CLASSIFICATION SETUP
OLLAMA_API_URL = "http://<ollama host>/api/generate" 
MODEL_NAME = "qwen3:30b-a3b-Instruct" 
# Instructions for the LLM to map raw technical descriptions into a standardized taxonomy
classification_rules_prompt = """
You are an expert marine welding quality inspector.
Please categorize the provided "Factor" text based on its core issue into the following **Secondary Sub-factors** under 6 main categories.
Each factor must be mapped to **exactly one** matching Secondary Sub-factor. Do NOT map to the primary category names (e.g., "Human Factor").
Categories and Secondary Sub-factors definitions:

1. Human Factor:
   - Skill Level: Insufficient skill, poor training, new employees, inability to read blueprints, uncertified, etc.
   - Responsibility & Attitude: Lack of responsibility, weak quality awareness, careless attitude, blind construction, etc.
   - Operational Compliance: Illegal operations, not following blueprints/process requirements, improper handling, etc.

2. Equipment Factor:
   - Equipment Status: Abnormal equipment, damaged functions, missing parts, poor maintenance, damaged cables, etc.
   - Tool Usage: Improper use of tools, unqualified tools, insufficient protection, welder failure, etc.
   - Performance Indicators: Abnormal current/voltage, excessive travel speed, insufficient heating temperature, etc.

3. Material Factor:
   - Material Quality: Unqualified materials, raw materials not meeting requirements, hardness exceeding limit, etc.
   - Welding Material Management: Rusty welding wire, damp electrodes, wrong type of welding wire used, etc.
   - Storage Conditions: Improper temporary storage, insufficient oven temperature, damp/rusty consumables, etc.

4. Technology Factor:
   - Process Design: Design defects, incomplete process, groove issues, lack of specific process instructions, etc.
   - Process Execution: Improper parameters, wrong methods, chaotic sequence, insufficient pretreatment, welding too fast, etc.
   - Technical Standards: Unclear standards, specification mismatch, insufficient weld leg, excessive gap, etc.

5. Environment Factor:
   - Weather Conditions: Weather impact, low temperature, high humidity, rain, monsoon season, etc.
   - Construction Environment: Narrow space, high difficulty, poor conditions, strong wind, insufficient light, etc.

6. Management Factor:
   - Supervision & Inspection: Inadequate inspection, missing mutual checks, lack of oversight, failure to detect issues, etc.
   - Systems & Procedures: Imperfect systems, failure to implement requirements, non-standard workflows, poor dispatching, etc.
   - Training & Briefing: Insufficient training, unclear briefing, requirements not communicated, no pre-job training, etc.

[Classification Rules]
- ONLY return names from this list: Skill Level, Responsibility & Attitude, Operational Compliance, Equipment Status, Tool Usage, Performance Indicators, Material Quality, Welding Material Management, Storage Conditions, Process Design, Process Execution, Technical Standards, Weather Conditions, Construction Environment, Supervision & Inspection, Systems & Procedures, Training & Briefing.
- NEVER return primary category names.
- Map to "Unclassified" if no sub-factor matches.
- Return a strict JSON object without any explanation or extra text.

Format:
{
  "secondary_factor": ["Sub-factor1", "Sub-factor2", ...]
}
"""
def classify_factor_with_ollama(factor_string):
    """
    Communicates with local Ollama API to categorize a specific text factor.
    
    Args:
        factor_string (str): The raw text from the Neo4j node.
    Returns:
        str: The matched standardized category name.
    """
    prompt = f"{classification_rules_prompt}\n\nFactor: {factor_string}"
    data = {
        "model": MODEL_NAME, 
        "prompt": prompt,
        "stream": False, 
        "options": {
            "temperature": 0, # Ensures consistent, deterministic categorization
            "num_predict": 500 
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        response.raise_for_status() 
        response_json = response.json()
        response_content = response_json.get("response", "").strip()
        
        # Parse result and filter out unclassified items
        result = json.loads(response_content)
        subcategories = result.get("secondary_factor", [])
        matched_subcategories = [cat for cat in subcategories if cat != "Unclassified"]
        return '; '.join(matched_subcategories) if matched_subcategories else ''
    except Exception as e:
        print(f"Error processing factor: {factor_string}. Error: {e}")
        return ''
#  DATA PROCESSING & STANDARDIZATION
print("--- Step 1: Fetching raw relations from Neo4j ---")
relation_data = get_all_relations()
driver.close() # Close DB connection once data is in memory
print(f"Retrieved {len(relation_data)} records.")
# Export raw data for auditing
df_original = pd.DataFrame(relation_data, columns=['source_node', 'target_node', 'count'])
df_original.to_csv('original_relations.csv', index=False)
print("\n--- Step 2: Classifying raw nodes via LLM ---")
classified_relations = []
classification_log = []
for i, (source, target, count) in enumerate(relation_data):
    print(f"\nProcessing Relation {i+1}: {source} -> {target}")
    
    # Classify both ends of the causal link
    source_class = classify_factor_with_ollama(source)
    time.sleep(1) # Delay to respect local CPU/GPU resources
    target_class = classify_factor_with_ollama(target)
    time.sleep(1) 

    # Extract the primary category from the LLM response
    s_cat = source_class.split(';')[0] if source_class else None
    t_cat = target_class.split(';')[0] if target_class else None
    
    # Log mapping results for quality control
    status = 'Success' if s_cat and t_cat else 'Skipped'
    classification_log.append({
        'original_source': source,
        'original_target': target,
        'original_count': count,
        'mapped_source_class': s_cat,
        'mapped_target_class': t_cat,
        'status': status
    })
    
    if s_cat and t_cat:
        classified_relations.append((s_cat, t_cat, count))
        print(f"  - Mapped: {s_cat} -> {t_cat}")

pd.DataFrame(classification_log).to_csv('classification_log.csv', index=False)
#  PROBABILISTIC MODELING
print("\n--- Step 3: Building frequency dictionary ---")
# Aggregates how many times Category A leads to Category B
freq_dict_classified = defaultdict(lambda: defaultdict(int))
for s_class, t_class, cnt in classified_relations:
    freq_dict_classified[s_class][t_class] += cnt 

with open('freq_dict_classified.json', 'w', encoding='utf-8') as f:
    json.dump({k: dict(v) for k, v in freq_dict_classified.items()}, f, indent=2)

print("\n--- Step 4: Calculating transition probabilities ---")
# Computes the likelihood of moving from one factor to the next
direct_prob_classified = {}
for node, targets in freq_dict_classified.items():
    total = sum(targets.values())
    if total > 0:
        # P(Event B | Event A) = Frequency(A->B) / Total Outbound from A
        direct_prob_classified[node] = {t: cnt / total for t, cnt in targets.items()}

with open('direct_prob_classified.json', 'w', encoding='utf-8') as f:
    json.dump(direct_prob_classified, f, indent=2)
# 5. PATH ANALYSIS (DFS ALGORITHM)
def calculate_path_prob(direct_prob, path):
    """
    Calculates the total probability of a specific chain of events.
    Formula: P(Path) = P(Node1->Node2) * P(Node2->Node3) ...
    """
    prob = 1.0
    for i in range(len(path)-1):
        prob *= direct_prob[path[i]].get(path[i+1], 0.0)
    return prob

def find_all_paths(freq_dict, start, end, max_length=4):
    """
    Finds all potential paths between two factors using Depth-First Search.
    
    Args:
        max_length: Limits the depth to avoid long/circular chains.
    """
    paths = []
    def dfs(current, path, length):
        if current == end and length > 0:
            paths.append(list(path))
            return
        if length >= max_length: return
        for neighbor in freq_dict[current].keys():
            dfs(neighbor, path + [neighbor], length + 1)
    
    dfs(start, [start], 0)
    return paths

print("\n--- Step 5: Identifying maximum probability evolution paths ---")
all_nodes = list(freq_dict_classified.keys())
results = []

# Exhaustive search for the most likely path between all factor pairs
for start in all_nodes:
    for end in all_nodes:
        if start == end: continue
        paths = find_all_paths(freq_dict_classified, start, end)
        if paths:
            # Calculate and rank by total probability
            path_probs = [(p, calculate_path_prob(direct_prob_classified, p)) for p in paths]
            max_path, max_prob = max(path_probs, key=lambda x: x[1])
            results.append({
                "start_node": start,
                "end_node": end,
                "max_path": " -> ".join(max_path),
                "max_prob": max_prob
            })


# ENCODING & FINAL EXPORT
# Standardized industry codes for final reporting
category_map = {
    "Skill Level": "H1", "Responsibility & Attitude": "H2", "Operational Compliance": "H3",
    "Supervision & Inspection": "MG1", "Systems & Procedures": "MG2", "Training & Briefing": "MG3",
    "Material Quality": "MA1", "Welding Material Management": "MA2", "Storage Conditions": "MA3",
    "Process Design": "TE1", "Process Execution": "TE2", "Technical Standards": "TE3",
    "Weather Conditions": "EN1", "Construction Environment": "EN2",
    "Equipment Status": "EQ1", "Tool Usage": "EQ2", "Performance Indicators": "EQ3",
}
if results:
    final_df = pd.DataFrame(results)
    # Apply alphanumeric codes to the calculated paths for brevity
    final_df['path_code'] = final_df['max_path'].apply(
        lambda p: " ".join([category_map.get(step.strip(), "UNK") for step in p.split("->")])
    )    
    # Save final analytical report
    final_df.to_csv('final_max_prob_paths.csv', index=False)
    print("\n--- Final Analysis Complete ---")
    print("Top 5 Risk Evolution Paths:")
    print(final_df.sort_values("max_prob", ascending=False).head())
else:
    print("\nNo valid evolution paths found.")

print("\nAll intermediate and final data files have been saved to the current directory.")
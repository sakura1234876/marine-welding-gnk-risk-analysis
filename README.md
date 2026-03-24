# Marine Welding Quality Risk Evolution & Coupling Analysis System (G-NK Model)

## 📌 Project Overview
In the complex world of shipbuilding, quality failures are rarely isolated events; they are the result of "coupled" risks. This project provides an automated pipeline to transform unstructured marine welding incident reports into a quantifiable **G-NK Risk Model**. This system transforms unstructured welding accident/quality reports into a structured, quantifiable risk model. By leveraging the G-NK Model, it evaluates both the structural coupling (how risks interact) and the evolutionary probability (how risks progress over time).

By combining **Large Language Models (LLMs)**, **Knowledge Graphs**, and **Information Entropy**, the system identifies hidden risk chains and calculates the probability of accident evolution.

---
## 🛠 Tech Stack
Language: Python 3.10+
Database: Neo4j (Graph Database)
LLM Engine: Ollama 
Analysis: Pandas, NumPy, Scikit-learn
Visualization: Matplotlib, Seaborn

## 🔄 System Workflow
### 1. Data Ingestion & Preprocessing
   - Extraction: Extracts raw text from .docx incident reports.
   - Segmentation: Uses Regex to isolate specific fields: Occurrence Time, Stage, Phenomenon, and Root Cause Analysis.
   - Cleaning: Normalizes text and exports to a structured .csv format for downstream processing.
### 2. Semantic Classification (LLM Layer)
   - 5M1E Framework: The LLM categorizes raw "Root Cause" text into a standardized taxonomy:
   - Human (HU): Skill levels, responsibility, habits.
   - Equipment (EQ): Equipment status, tool usage.
   - Material (MA): Consumables, storage quality.
   - Technology (TE): Process design, technical standards.
   - Environment (EN): Weather, construction site conditions.
   - Management (MG): Supervision, systems, training.
   Encoding: Maps 17 sub-factors to alphanumeric codes (e.g., H1, MG2) for mathematical modeling.
### 3. Knowledge Graph Construction
   - Creates nodes for each unique term in the text data.
   - Establishes relationships between nodes based on co-occurrence and semantic similarity.
   - Exports the constructed graph to a Neo4j database for further analysis.
### 4. Mathematical Modeling (G-NK Model)
The system calculates a comprehensive risk score (S) by balancing structural interaction and historical frequency:
S = α · T + β · P
- Coupling Degree (T): Derived from the N-K Model using multi-information entropy. It measures the complexity of risk interactions (2nd-order to 6th-order coupling).
- Transition Probability (P): Calculated via Markov chains based on the historical frequency of risk evolution paths.
- Weights (α, β): Adjustable parameters to shift focus between structural potential and observed frequency.
## 🛠 Prerequisites

### 1. Services
* **Ollama:** Installed and running with the `qwen3:30b-a3b-Instruct` model pulled.
* **Neo4j:** A running instance (local or cloud) with `bolt` protocol enabled.

### 2. Environment
* Python 3.10 or higher.
* Recommended: A virtual environment (`python -m venv venv`).

---

## ⚙️ Installation & Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   Create a `.env` file and add the following variables:
   ```plaintext
   OLLAMA_API_KEY=<your-ollama-api-key>
   NEO4J_BOLT_URL=<your-neo4j-bolt-url>
   NEO4J_USERNAME=<your-neo4j-username>
   NEO4J_PASSWORD=<your-neo4j-password>
   ```

---

## 🚀 How to Run
1.Start Local Services: Ensure Ollama (with Qwen3) and Neo4j are running.
2.Install dependencies: 
    `python -m venv venv`
    `source venv/bin/activate  # On Windows: venv\Scripts\activate`
    `Setup: pip install -r requirements.txt`
3.Data processing:
    Place the text data in the "input" folder.
    `python Data_processing.py`
4.Descriptive statistics:
    Adjust the Descriptive statistics.html file based on the data processing results.
5.Knowledge Graph Construction:
    `python KG_construction.py`
6.Calculation of transition probability:
    `python Transition_probability.py`
7.Mathematical Modeling:
    `python NK.py`
8.Parameter analysis:
    `python Parameter_analysis.py`
## 📊 Results
The system generates a comprehensive risk model, including:
- Descriptive statistics: A detailed analysis of the data.
- Knowledge Graph: A visual representation of the relationships between risks.
- Transition Probability: A quantifiable measure of risk evolution.
- Coupling Intensity Analysis: A quantitative assessment of the synergistic interactions between risk factors, identifying which combinations (e.g., Human-Management) create the highest systemic risk.
- Integrated Risk Prioritization (G-NK): A multi-dimensional scoring mechanism ($S = \alpha T + \beta P$) that balances structural coupling with observed historical frequency to rank the most critical threat paths.
- Structural Stability Validation: Sensitivity analysis confirming the robustness of risk rankings through overlap tests across varying weighting parameters.
- Critical Evolution Logic Chains: Sequential causal paths that visualize how organizational oversights (e.g., training deficiencies) systematically transition into specific technical welding defects.
---
## 📈 Key Outcomes
- Quantitative Risk Coupling: Leveraging the N-K Model to quantify the "synergy" between disparate risks, moving beyond simple frequency counts to analyze complex interaction intensity.

- Optimized Decision Support: The G-NK Score provides a definitive priority list, enabling management to allocate resources to the risk chains with the highest combined structural and historical impact.

- Model Robustness Proof: Stability Analysis proves that the identified "Top 15" risks remain consistent under different expert weighting schemes, ensuring the reliability of the analytical conclusions.





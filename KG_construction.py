#It is necessary to create an "input" folder in the same-level directory and place the data inside it.
from llama_index.core import SimpleDirectoryReader,Document,PropertyGraphIndex
import re
documents = SimpleDirectoryReader("./input").load_data()
import nest_asyncio
nest_asyncio.apply()
from typing import Literal, List, Any
from llama_index.llms.ollama import Ollama
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.embeddings.ollama import OllamaEmbedding
# This function is used to split the documents by 10 "#".
def split_by_10_hashes(documents: List[Document]) -> List[Document]:
    """
    Split the document using the delimiter consisting of 10 '#' symbols. 
    :param documents: List of original documents
    :return: List of segmented documents
    """
    split_docs = []
    separator = r"\s*#{10}\s*"
    for doc in documents:
        raw_text = doc.text.strip()
        segments = re.split(separator, raw_text)
        for i, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue
            split_doc = Document(
                text=segment,
            )
            split_docs.append(split_doc)
    return split_docs
split_docs=split_by_10_hashes(documents)
# Define the entities, relationships and constraints(schema) of the knowledge graph
entities = Literal[
    "QUALITY_DEFECT",      
    "HUMAN_FACTOR",        
    "EQUIPMENT_FACTOR",    
    "MATERIAL_FACTOR",     
    "ENVIRONMENT_FACTOR",  
    "MANAGEMENT_FACTOR",   
    "PROCESS_FACTOR",      
    "CONSTRUCTION_STAGE"   
]
relations = Literal[
    "CAUSED_BY",           
    "OCCURS_IN",            
    "LEADS_TO",             
    "BELONGS_TO"           
]
validation_schema = {
    "QUALITY_DEFECT": ["CAUSED_BY", "OCCURS_IN"],
    "HUMAN_FACTOR": ["LEADS_TO", "BELONGS_TO"],
    "EQUIPMENT_FACTOR": ["LEADS_TO", "BELONGS_TO"],
    "MATERIAL_FACTOR": ["LEADS_TO", "BELONGS_TO"],
    "ENVIRONMENT_FACTOR": ["LEADS_TO", "BELONGS_TO"],
    "MANAGEMENT_FACTOR": ["LEADS_TO", "BELONGS_TO"],
    "PROCESS_FACTOR": ["LEADS_TO", "BELONGS_TO"],
    "CONSTRUCTION_STAGE": [] 
}
# Define the allowed defects
ALLOWED_DEFECTS = ["Spatter", "Edge Burning", "Air Hole", "Overlap", "Crack", "Pit"]
allowed_defects_str = "、".join(ALLOWED_DEFECTS)
# Define the prompt for the LLM
prompt = f"""
You are a professional knowledge graph assistant for welding engineering. Please strictly extract entities and relationships from the case text. 
        Core rule: For the `QUALITY_DEFECT` entity, **only the following 6 types** are permitted to be extracted:
        【{allowed_defects_str}】 
        **Important Instruction:**
        1. **Normalization**: If the words "dense pores" or "surface pores" appear in the text, extract them as the standard term **"pores"**.
        2. **Filtering**: If words such as "inclusion", "unwelded", "weld bulge", "poor formation" etc., which are not on the list, appear in the text, **please ignore them** and do not extract them as defect entities. 
        ### [Schema Definition]
        - Entity Type:
        - QUALITY_DEFECT: Quality Defect (Limited to White List)
        - HUMAN_FACTOR, EQUIPMENT_FACTOR, MATERIAL_FACTOR, ENVIRONMENT_FACTOR, MANAGEMENT_FACTOR, PROCESS_FACTOR: Various Cause Factors
        - CONSTRUCTION_STAGE: Construction Stage 
        - Relationship type:
        - CAUSED_BY: The defect is caused by a factor (Defect -> Factor)
        - OCCURS_IN: The defect occurs in a certain stage (Defect -> Stage)
        - LEADS_TO: The causal transmission between factors (Factor A -> Factor B). For example: "Uncleaned groove (process) LEADS_TO Oil residue (material)"
        - BELONGS_TO: Specific factor classification (Factor -> Factor Category). For example: "Lack of responsibility BELONGS_TO HUMAN_FACTOR" 
        ### [Extraction Rules]
        1. Extract the defects in the "Problem Phenomenon" while strictly adhering to the whitelist.
        2. Extract the specific factors in the "Cause Analysis".
        3. Construct **[Defect] -> CAUSED_BY -> [Factor]**.
        4. Construct **[Defect] -> OCCURS_IN -> [Construction Phase]**.
        5. If there is a clear **causal trigger sequence** among the factors, construct **[Factor A] -> LEADS_TO -> [Factor B]**.
        6. For each specific factor, it is necessary to construct **[Specific Factor] -> BELONGS_TO -> [General Factor Label]** (such as HUMAN_FACTOR).
        7. Do not extract any "coupling" or "related" relationships; only focus on the causal chain. 
        Please provide the text that needs to be translated. -------
        {{text}}
        -------
        Please extract the entities and relationships from the above text according to the schema definition and extraction rules."""
# From the original triplets data generated by large language models (LLMs), a rigorous filtering and standardized cleaning process is carried out.
class StrictCleaningExtractor(SchemaLLMPathExtractor):
    def _extract_from_nodes(self, nodes: List[Any]) -> List[Any]:
        triplets = super()._extract_from_nodes(nodes)
        valid_triplets = []
        for triplet in triplets:
            t_dict = {}
            if hasattr(triplet, "dict"): t_dict = triplet.dict()
            elif isinstance(triplet, dict): t_dict = triplet.copy()
            else: 
                try: t_dict = dict(triplet)
                except: continue
            head = t_dict.get("head")
            tail = t_dict.get("tail")
            head_type = t_dict.get("head_type")
            tail_type = t_dict.get("tail_type")
            if not head or not tail: continue
            if head == tail: continue
            if head_type == "QUALITY_DEFECT":
                matched = None
                for allowed in ALLOWED_DEFECTS:
                    if allowed in head:
                        matched = allowed
                        break
                if matched:
                    t_dict["head"] = matched 
                else:
                    continue 
            if tail_type == "QUALITY_DEFECT":
                matched = None
                for allowed in ALLOWED_DEFECTS:
                    if allowed in tail:
                        matched = allowed
                        break
                if matched:
                    t_dict["tail"] = matched
                else:
                    continue
            valid_triplets.append(t_dict)
        return valid_triplets
# Initialize the custom graph extractor with strict cleaning logic
kg_extractor = StrictCleaningExtractor(
    llm=Ollama(
        model="qwen3:30b-a3b-Instruct", 
        json_mode=True, 
        request_timeout=3600
    ),                                   # High-capacity model with JSON enforcement
    possible_entities=entities,          # Predefined entity vocabulary
    possible_relations=relations,        # Predefined relation vocabulary
    kg_validation_schema=validation_schema, # Structural validation rules
    num_workers=1,                       # Sequential processing to ensure stability
    max_triplets_per_chunk=20,           # Threshold to prevent hallucination/redundancy
    extract_prompt=prompt,               # Custom instruction for the extraction task
    strict=False,                        # Allow flexible extraction before manual filtering
)
# Initialize the Neo4j Property Graph Store 
# Note: In production, replace these with environment variables or a secret manager
graph_store = Neo4jPGStore(
    username="<YOUR_NEO4J_USERNAME>", # Database user (e.g., "neo4j")
    password="<YOUR_NEO4J_PASSWORD>", # Database password
    url="bolt://<YOUR_NEO4J_HOST>:<PORT>", # Connection URL (e.g., "bolt://localhost:7687")
)
# Create the Property Graph Index from processed document chunks
# This step triggers the extraction, embedding, and storage process
index = PropertyGraphIndex.from_documents(
    split_docs,                         # The source text chunks/documents
    kg_extractors=[kg_extractor],       # Using the custom StrictCleaningExtractor defined earlier
    embed_model=OllamaEmbedding(
        model_name="bge-m3:latest"      # Multi-lingual embedding model for node/edge vectorization
    ),
    property_graph_store=graph_store,   # The Neo4j instance to persist the graph data
    show_progress=True,                 # Display a progress bar during the ingestion process
)
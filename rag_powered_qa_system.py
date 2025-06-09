!pip install --upgrade --force-reinstall \
    langchain \
    pymongo \
    langchain-mongodb \
    huggingface_hub==0.22.2 \
    python-dotenv \
    langsmith \
    langchain-community \
    transformers 

!pip show huggingface_hub langchain-community transformers

import os

os.environ["HF_TOKEN"] = ""
os.environ["MONGO_URI"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
# -----------------------------------------------------------------

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ColabGraphRAGProject" # A project name for LangSmith

print("Environment variables set. Ensure you replaced the placeholder values.")

docs = [
    {
        "page_content": "The EcoCharge Pro is a solar-powered phone charger. It features a 10,000mAh battery and two USB-A ports. It's compatible with all smartphones and small electronic devices. Price: $79.99.",
        "metadata": {"source": "product_specs", "product_id": "ECP001"}
    },
    {
        "page_content": "Our new product, the AquaFilter Max, provides clean drinking water using a multi-stage filtration system. It can filter up to 1,500 liters of water and is ideal for camping and emergency preparedness. Price: $49.99.",
        "metadata": {"source": "product_specs", "product_id": "AFM002"}
    },
    {
        "page_content": "The GlowBulb Smart Light offers customizable RGB lighting and integrates with Alexa and Google Home. It has a lifespan of 25,000 hours and a brightness of 800 lumens. Price: $29.99.",
        "metadata": {"source": "product_specs", "product_id": "GBSL003"}
    },
    {
        "page_content": "Customer support for EcoCharge Pro is available via email at support@ecocharge.com. For technical issues, refer to the online manual. Warranty period is 1 year.",
        "metadata": {"source": "support_info", "product_id": "ECP001"}
    }
]

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.graphrag.graph import MongoDBGraphStore
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings 
from langchain_community.llms import HuggingFaceEndpoint 
import os 

# Define your LLM_MODEL
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta" 

# Initialize LLM for general generation and for entity extraction within MongoDBGraphStore
llm_generation = HuggingFaceEndpoint(
    repo_id=LLM_MODEL,
    temperature=0.7, # Adjust as needed
    huggingface_api_token=os.environ["HF_TOKEN"], # Ensure this matches Cell 2
    max_new_tokens=500
)

# Define your database and collection names explicitly (from your Atlas UI)
ATLAS_DB_NAME = "sample_mflix"
ATLAS_COLLECTION_NAME = "comments"
# Initialize MongoDBGraphStore with the new required argument
graph_store = MongoDBGraphStore(
    collection_name=ATLAS_COLLECTION_NAME,
    connection_string=os.environ["MONGO_URI"],
    database_name=ATLAS_DB_NAME,
    entity_extraction_model=llm_generation # <-- NEW REQUIRED ARGUMENT
)

print("MongoDBGraphStore and LLM initialized successfully.")

from langchain_community.llms import HuggingFaceEndpoint 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing import List, Dict, Any
import asyncio
import os 
import re
from pymongo import MongoClient 

ATLAS_DB_NAME = "sample_mflix"           
ATLAS_COLLECTION_NAME = "comments"

# Initialize MongoDBGraphStore using these variables
graph_store = MongoDBGraphStore(
    collection_name=ATLAS_COLLECTION_NAME,
    connection_string=os.environ["MONGO_URI"],
    database_name=ATLAS_DB_NAME,
    entity_extraction_model=llm_generation
)

# 1. Define a prompt for extracting entities from the user's query
entity_extraction_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an assistant that extracts key entities (like product names, features, companies) from user questions to aid in graph traversal. List only the most relevant, distinct entities, one per line, and nothing else."),
    HumanMessage(content="Question: {question}")
])
entity_extractor_chain = entity_extraction_prompt | llm_generation | StrOutputParser()

# 2. Define a function to retrieve relevant graph data from MongoDBGraphStore
async def get_relevant_graph_data(query: str):
    print(f"\nUser query: {query}")
    extracted_entities_str = await entity_extractor_chain.ainvoke({"question": query})

    extracted_entities = []
    for line in extracted_entities_str.split('\n'):
        line = line.strip()
        if line.startswith("Assistant:"):
            entities_in_line = line[len("Assistant:"):].strip()
            
            if ',' in entities_in_line:
                extracted_entities.extend([e.strip() for e in entities_in_line.split(',') if e.strip()])
            elif entities_in_line:
                extracted_entities.append(entities_in_line)
        elif line and "Question:" not in line and "{" not in line and "}" not in line and "Assistant:" not in line:
            extracted_entities.append(line)

    extracted_entities = [e for e in extracted_entities if e] 
    extracted_entities = list(set(extracted_entities)) 

    print(f"Parsed entities from query: {extracted_entities}")


    if not extracted_entities:
        return "No specific entities extracted from the query. Cannot perform graph traversal."

    temp_client = None # Initialize to None for error handling
    try:
        # Instantiate a new MongoClient directly using the MONGO_URI
        temp_client = MongoClient(os.environ["MONGO_URI"])
        temp_db = temp_client["sample_mflix"] # Use database name from graph_store config
        collection = temp_db["comments"] # Use collection name from graph_store config
    except Exception as e:
        if temp_client: # Ensure client is closed if an error occurs during setup
            temp_client.close()
        raise RuntimeError(f"Failed to connect to MongoDB for graph data retrieval: {e}")

    relevant_info = []
    processed_ids = set()

    for entity_name in extracted_entities:
        escaped_entity_name = re.escape(entity_name)

        nodes = collection.find({
            "$or": [
                {"properties.name": {"$regex": escaped_entity_name, "$options": "i"}}, 
                {"_label": {"$regex": escaped_entity_name, "$options": "i"}}
            ]
        })

        for node in nodes:
            node_id = str(node["_id"])
            if node_id not in processed_ids:
                props = node.get("properties", {})
                relevant_info.append(f"Node: {node.get('_label', 'Unknown Type')}: {props.get('name', 'N/A')}, Description: {props.get('description', 'N/A')}")
                processed_ids.add(node_id)

            # Step 2: Find relationships connected to this node
            relationships = collection.find({
                "$or": [
                    {"_from": node_id},
                    {"_to": node_id}
                ]
            })
            for rel in relationships:
                rel_id = str(rel["_id"])
                if rel_id not in processed_ids: # Ensure relationship is not duplicated
                    source_node = collection.find_one({"_id": rel["_from"]})
                    target_node = collection.find_one({"_id": rel["_to"]})

                    if source_node and target_node:
                        source_name = source_node.get("properties", {}).get("name", "N/A")
                        target_name = target_node.get("properties", {}).get("name", "N/A")
                        rel_type = rel.get("_label", "UNKNOWN_RELATIONSHIP")
                        rel_props = rel.get("properties", {})
                        rel_desc = rel_props.get("description", "N/A")

                        relevant_info.append(f"Relationship: {source_name} --({rel_type})--> {target_name}, Desc: {rel_desc}")
                        processed_ids.add(rel_id)

    if not relevant_info:
        return "No relevant information found in the knowledge graph for these entities."

    unique_relevant_info = list(dict.fromkeys(relevant_info)) # Remove duplicates

    # Close the temporary client connection at the end of the function
    if temp_client:
        temp_client.close()

    return "\n".join(unique_relevant_info)

# 3. Define the RAG chain
qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided knowledge graph context. If the answer is not in the context, state that you don't have enough information."),
    HumanMessage(content="Knowledge Graph Context:\n{context}\n\nQuestion: {question}")
])

rag_chain = (
    {"context": get_relevant_graph_data, "question": lambda x: x["question"]}
    | qa_prompt
    | llm_generation
    | StrOutputParser()
)

async def run_qa_tests():
    questions = [
        "What are the features of EcoCharge Pro?",
        "How much does the AquaFilter Max cost and what is its purpose?",
        "What devices is the GlowBulb Smart Light compatible with?",
        "How can I get support for EcoCharge Pro?",
        "What is the warranty for EcoCharge Pro?",
        "Tell me about a product called SmartHome Hub."
    ]

    for q in questions:
        response = await rag_chain.ainvoke({"question": q})
        print(f"\n--- Question: {q} ---")
        print(f"--- Answer: {response} ---\n")

# Run the QA tests
await run_qa_tests()

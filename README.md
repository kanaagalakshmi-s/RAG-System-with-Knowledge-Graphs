# RAG-System-with-Knowledge-Graphs
Demystifying LLMs: My Hands-On Journey Building a RAG System with Knowledge Graphs!
Graph RAG with LangChain and MongoDB
This code cell sets up and runs a Graph RAG (Retrieval Augmented Generation) system using LangChain and MongoDB. It demonstrates how to extract entities from a user query, retrieve relevant information from a MongoDB knowledge graph based on those entities, and then use a Large Language Model (LLM) to answer the user's question based on the retrieved graph data.

Imports and Initialization
The cell begins by importing necessary libraries and modules:

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
from langchain_mongodb.graphrag.graph import MongoDBGraphStore
Use code with caution
HuggingFaceEndpoint: Used to connect to a Hugging Face LLM hosted on their inference API.
ChatPromptTemplate, SystemMessage, HumanMessage: Used to create structured prompts for the LLM.
BaseModel, Field: Used for data validation (although not strictly used in the final RAG chain in this cell, they are common in LangChain).
JsonOutputParser, StrOutputParser: Used to parse the output from the LLM. StrOutputParser is used here to get a simple string response.
List, Dict, Any: Standard Python typing hints.
asyncio: Used to run asynchronous code, particularly the LLM invocations.
os: Used to access environment variables for API keys and connection strings.
re: Used for regular expressions, specifically to escape special characters in entity names for MongoDB queries.
MongoClient: The standard Python driver for interacting with MongoDB.
MongoDBGraphStore: A LangChain component specifically designed to interact with a knowledge graph stored in MongoDB.
Next, it re-initializes the MongoDBGraphStore and the LLM for generating text:

ATLAS_DB_NAME = "sample_mflix"
ATLAS_COLLECTION_NAME = "comments"

graph_store = MongoDBGraphStore(
    collection_name=ATLAS_COLLECTION_NAME,
    connection_string=os.environ["MONGO_URI"],
    database_name=ATLAS_DB_NAME,
    entity_extraction_model=llm_generation
)

# Initialize LLM for query understanding and response generation
llm_generation = HuggingFaceEndpoint(
    repo_id=LLM_MODEL,
    temperature=0.7,
    huggingface_api_token=os.environ["HF_TOKEN"],
    max_new_tokens=500
)
Use code with caution
This part sets the MongoDB database and collection names and then creates instances of MongoDBGraphStore and HuggingFaceEndpoint. The llm_generation instance is used both for entity extraction within the graph_store and for the final answer generation.

Entity Extraction
This section defines how the system identifies key concepts or "entities" from the user's input question:

# 1. Define a prompt for extracting entities from the user's query
entity_extraction_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an assistant that extracts key entities (like product names, features, companies) from user questions to aid in graph traversal. List only the most relevant, distinct entities, one per line, and nothing else."),
    HumanMessage(content="Question: {question}")
])
entity_extractor_chain = entity_extraction_prompt | llm_generation | StrOutputParser()
Use code with caution
entity_extraction_prompt: This is a ChatPromptTemplate that instructs the LLM (via the SystemMessage) to act as an entity extractor and then provides the user's question as a HumanMessage.
entity_extractor_chain: This is a simple LangChain "chain" created by piping the entity_extraction_prompt, the llm_generation model, and a StrOutputParser. When this chain is invoked with a user question, it will format the prompt, send it to the LLM, and the LLM's response (which is expected to be a list of entities) will be parsed into a string.
Graph Data Retrieval
The core logic for fetching relevant information from the MongoDB graph is encapsulated in the get_relevant_graph_data asynchronous function:

# 2. Define a function to retrieve relevant graph data from MongoDBGraphStore
async def get_relevant_graph_data(query: str):
    print(f"\nUser query: {query}")
    extracted_entities_str = await entity_extractor_chain.ainvoke({"question": query})

    # --- REFINED ENTITY EXTRACTION LOGIC ---
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
    # --- END OF REFINED ENTITY EXTRACTION LOGIC ---


    if not extracted_entities:
        return "No specific entities extracted from the query. Cannot perform graph traversal."

    temp_client = None
    try:
        temp_client = MongoClient(os.environ["MONGO_URI"])
        temp_db = temp_client["sample_mflix"]
        collection = temp_db["comments"]
    except Exception as e:
        if temp_client:
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

            relationships = collection.find({
                "$or": [
                    {"_from": node_id},
                    {"_to": node_id}
                ]
            })
            for rel in relationships:
                rel_id = str(rel["_id"])
                if rel_id not in processed_ids:
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

    unique_relevant_info = list(dict.fromkeys(relevant_info))

    if temp_client:
        temp_client.close()

    return "\n".join(unique_relevant_info)
Use code with caution
The function takes the user query as input.
It first uses the entity_extractor_chain to get a list of entities from the query.
Refined Entity Extraction Logic: This block parses the LLM's output to extract only the actual entities, handling potential conversational elements or unexpected formatting from the LLM.
If no entities are found, it returns a message indicating that graph traversal cannot be performed.
It then establishes a direct connection to the MongoDB collection designated for the graph.
It iterates through the extracted entities:
For each entity, it searches the collection for nodes where the properties.name or the node's label (_label) matches the entity name (case-insensitive regex search).
For each matching node found, it adds the node's information to a list (relevant_info) and tracks its ID in processed_ids to avoid duplicates.
It then searches for relationships (_from or _to fields) connected to the current node's ID.
For each connected relationship, it retrieves the source and target nodes, formats the relationship information (source node name, relationship type, target node name, relationship description), adds it to relevant_info, and tracks the relationship ID in processed_ids.
Finally, it removes any duplicate information from relevant_info and returns the unique information as a single string, closing the MongoDB connection before returning.
RAG Chain Definition
This part defines the overall Retrieval Augmented Generation (RAG) process:

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
Use code with caution
qa_prompt: This is a ChatPromptTemplate designed for the final answer generation. It includes a SystemMessage instructing the LLM to answer based only on the provided context and a HumanMessage that includes the context (the retrieved graph data) and the original question.
rag_chain: This is the main LangChain chain that orchestrates the RAG process:
{"context": get_relevant_graph_data, "question": lambda x: x["question"]}: This is a dictionary that defines the inputs for the next step (qa_prompt). It calls the get_relevant_graph_data function to get the context based on the original question.
| qa_prompt: The output of the previous step (the dictionary containing context and question) is passed to the qa_prompt to create the final prompt for the LLM.
| llm_generation: The generated prompt is sent to the llm_generation model to produce an answer.
| StrOutputParser(): The LLM's output is parsed into a simple string.
Running the QA Tests
The final part of the cell defines and runs a set of test questions:

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
Use code with caution
run_qa_tests(): This asynchronous function iterates through a predefined list of questions.
For each question, it invokes the rag_chain using await rag_chain.ainvoke({"question": q}). The ainvoke method is used because the get_relevant_graph_data function is asynchronous.
The response from the rag_chain (the LLM's answer) is then printed along with the original question.
await run_qa_tests(): This line executes the run_qa_tests asynchronous function, starting the process of asking questions and printing the generated answers.
In summary, this cell orchestrates a process where user questions are analyzed to find key entities. These entities are then used to query a MongoDB knowledge graph to retrieve relevant facts (nodes and relationships). Finally, the retrieved facts are provided as context to an LLM, which generates an answer to the original question based on that context.

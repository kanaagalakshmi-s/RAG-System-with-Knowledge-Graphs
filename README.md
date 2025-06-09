# RAG-Powered QA System with MongoDB Knowledge Graph & LangChain

This project implements a **Retrieval Augmented Generation (RAG)** based Question-Answering (QA) system. It leverages a structured **Knowledge Graph (KG) stored in MongoDB Atlas** to provide precise, factual grounding for a Large Language Model (LLM), aiming to generate accurate and context-rich answers while significantly mitigating common LLM "hallucinations."

This repository represents a hands-on learning journey into building robust AI applications, tackling real-world integration challenges, and understanding the practicalities of modern LLM pipelines.

## ✨ Features

* **Intelligent Knowledge Graph Management:** Stores and retrieves structured product information (entities like products, features, companies, and their relationships) in MongoDB Atlas using LangChain's `MongoDBGraphStore`.
* **LLM-Powered Entity Extraction:** Utilizes a Hugging Face LLM to accurately extract key entities from user's natural language queries.
* **Contextual Graph Retrieval:** Efficiently queries the MongoDB knowledge graph based on extracted entities to fetch highly relevant contextual information.
* **Retrieval Augmented Generation (RAG):** Combines the retrieved factual context with the original user question, feeding it to an LLM for grounded and accurate answer generation.
* **Flexible LLM Integration:** Designed to work with various Hugging Face LLMs via the Inference API, allowing for model experimentation.

## 🚀 Technologies Used

* **Python 3.x:** The core programming language.
* **LangChain:** The framework orchestrating the LLM, RAG, and graph database integrations.
    * `langchain-core`
    * `langchain-community`
    * `langchain-mongodb` (specifically `langchain_mongodb.graphrag.graph.MongoDBGraphStore`)
* **MongoDB Atlas:** Cloud database service for hosting the Knowledge Graph.
* **Hugging Face Inference API:** For accessing powerful LLMs (e.g., Zephyr-7b-beta, Flan-T5).
* **`pymongo`:** Python driver for MongoDB.
* **`python-dotenv`:** For secure management of environment variables.
* **Jupyter Notebook / Google Colab:** The development environment for interactive execution.

## 🛠️ Setup and Installation

This project is designed to be run interactively, ideally within a Google Colab or local Jupyter Notebook environment.

### Prerequisites

* A **MongoDB Atlas account** with a deployed cluster.
    * Ensure you have a MongoDB Connection String (SRV format is recommended).
    * Identify your desired Database Name and Collection Name within your Atlas cluster.
* A **Hugging Face API Token** (available from your Hugging Face profile settings).
* Python 3.9+ installed.

### Steps to Get Started

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
    *(Replace `your-username/your-repo-name` with your actual GitHub repository details).*

2.  **Set Up Environment Variables:**
    * Create a file named `.env` in the root directory of your cloned repository.
    * Add the following variables to your `.env` file, replacing the placeholders with your actual credentials:
        ```dotenv
        MONGO_URI="mongodb+srv://<username>:<password>@<cluster-url>/<database-name>?retryWrites=true&w=majority"
        HF_API_TOKEN="hf_<your_huggingface_api_token>"
        ```
        * **Important:** The `<database-name>` in `MONGO_URI` is crucial but should **not** be the `ATLAS_DB_NAME` you'll use in Cell 4 for the graph store. It's usually the default database the connection string points to, or left blank. Your `ATLAS_DB_NAME` and `ATLAS_COLLECTION_NAME` will be specified explicitly in the notebook (Cell 4).

3.  **Install Dependencies:**
    * Create a `requirements.txt` file in your repository with the following content:
        ```
        langchain-core
        langchain-community
        langchain-mongodb
        pymongo
        python-dotenv
        jupyter # Optional, if you plan to run locally
        ```
    * Create and activate a Python virtual environment (highly recommended):
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate # For macOS/Linux
        # .venv\Scripts\activate # For Windows CMD
        # .venv\Scripts\Activate.ps1 # For Windows PowerShell
        ```
    * Install the dependencies:
        ```bash
        pip install -r requirements.txt
        ```

## 🚀 Usage (Jupyter/Google Colab Notebook Workflow)

The core logic and execution flow are contained within the Jupyter/Google Colab notebook (e.g., `rag_qa_system.ipynb`).

1.  **Open the Notebook:**
    * Upload `rag_qa_system.ipynb` to Google Colab, or open it with Jupyter if running locally.

2.  **Execute Cells Sequentially:**
    * **Cell 1:** Installs required libraries (e.g., `langchain-mongodb`).
    * **Cell 2:** Loads environment variables (`MONGO_URI`, `HF_API_TOKEN`) from your `.env` file.
    * **Cell 3:** Defines sample raw documents (`docs`) that will be ingested into the knowledge graph.
    * **Cell 4:**
        * Initializes your chosen LLM (e.g., `HuggingFaceH4/zephyr-7b-beta`, `google/flan-t5-base`) via `HuggingFaceEndpoint`.
        * Defines `ATLAS_DB_NAME` and `ATLAS_COLLECTION_NAME` which **must match the database and collection names you intend to use in your MongoDB Atlas cluster for the knowledge graph.**
        * Initializes `MongoDBGraphStore`, crucially passing the LLM as `entity_extraction_model` (a new requirement in recent `langchain-mongodb` versions).
    * **Cell 5:** Defines the functions for ingesting documents and extracting graph data.
    * **Cell 6:** Executes the ingestion process, populating your MongoDB Knowledge Graph.
    * **Cell 7:**
        * Defines `get_relevant_graph_data` for efficient graph querying (including `re.escape` for robust regex handling).
        * Sets up the RAG `qa_prompt` and constructs the `rag_chain`.
        * Includes `run_qa_tests` to demonstrate the system with predefined questions.
    * **Cell 8:** Executes `run_qa_tests()` to show the QA system in action.

## 📈 Future Enhancements

* **Interactive UI:** Develop a simple front-end using Streamlit or Gradio for a more user-friendly experience.
* **Advanced Graph Queries:** Implement more complex traversal patterns for richer contextual retrieval.
* **Error Handling & Logging:** Enhance robustness with more comprehensive error handling and logging.
* **Performance Benchmarking:** Measure and optimize the latency and accuracy of the RAG pipeline.
* **Fine-tuning LLM:** Explore fine-tuning smaller LLMs for domain-specific entity extraction or answer generation.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, feature requests, or bug reports, please open an issue or submit a pull request.

---

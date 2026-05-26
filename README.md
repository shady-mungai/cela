# Cela - RAG-Based Company Assistant

Cela is a minimalist, document-augmented chat assistant built using a **RAG (Retrieval-Augmented Generation)** architecture. The system ingests local PDF documents, processes and indexes their contents into a local vector database, and uses a self-refining LLM chain via OpenRouter to answer user queries strictly based on the provided context.

---

## System Architecture & Workflow

The chatbot processes information and answers queries through the following pipeline:

1. **Document Ingestion:** A local PDF file (`sample.pdf`) is loaded and split into small, manageable text fragments.
2. **Vector Embedding:** These text fragments are transformed into dense mathematical vectors representing their semantic meaning.
3. **Local Storage:** The vectors are indexed locally using a highly efficient similarity search database.
4. **Retrieval & Refinement Chain:** When a user asks a question, the system retrieves the top 4 most relevant text fragments. It then sends the first fragment to the LLM for an initial answer, and sequentially passes subsequent fragments to **refine** and improve that answer.

---

## Core Dependencies Explained

The application utilizes a specific selection of libraries to handle interface rendering, document processing, and LLM orchestration:

### 1. User Interface
* **`streamlit`**: Manages the frontend web interface, reactive chat components, and session state processing.

### 2. LLM Orchestration & RAG Pipelines (`langchain` Ecosystem)
* **`langchain-openai`**: Handles connections to OpenAI-compatible endpoints. It points to OpenRouter to leverage the `llama-3.3-70b-instruct` model.
* **`langchain-huggingface`**: Generates high-quality vector representations of text. It runs the lightweight `all-MiniLM-L6-v2` model locally on the host machine.
* **`langchain-community`**: Provides modular integrations for loading third-party assets (such as local PDFs) and managing standard data structures.

### 3. Data Processing & Vector Storage
* **`pypdf`**: Extracts raw text from the input PDF file during the data ingestion phase.
* **`faiss-cpu`**: Facebook AI Similarity Search. A highly optimized local library used to store vector embeddings and perform rapid similarity searches when queries are made.

### 4. Environment & Network Utilities
* **`python-dotenv`**: Loads secret environment variables (like API keys) from a localized `.env` file into the application context safely.
* **`urllib3`**: Handles low-level HTTP network requests to the external OpenRouter API endpoint.

---

## Getting Started

### 1. Environment Setup
Create a `.env` file in the root directory of the project and supply the required API token:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
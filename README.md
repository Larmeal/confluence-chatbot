# Confluence Chatbot with Langchain (Powered by Gemini)

This repository provides an example of how to build a Retrieval-Augmented Generation (RAG) system using LangChain to work with Confluence documents. It leverages a large language model (LLM), such as Gemini, to create a chatbot application that assists Data Engineers with maintenance tasks.

![Chatbot Architecture](./img/langchain_confluence.jpg?raw=true "Chatbot Architecture")
![streamlit](./img/streamlit_app.png?raw=true "streamlit")

## Setup

1.  **Clone the repository (if applicable).**

2.  **Google Cloud Setup:**
    *   Ensure you have a Google Cloud Project.
    *   Enable the "Generative Language API" (Gemini API) for your project.
    *   Create an API Key with permissions for this API.

3.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install dependencies:**
    ```bash
    uv sync
    ```

    If you don't have uv installed, you should install it first by running
    ```bash
    pip install uv
    ```

5.  **Set up environment variables:**
    Create a `.env` file in the root `confluence_rag_chatbot/` directory with your credentials:
    ```env
    GOOGLE_API_KEY="your_google_api_key" # From Google Cloud Console
    CONFLUENCE_URL="https://your-domain.atlassian.net/wiki"
    CONFLUENCE_USER="your_confluence_email@example.com"
    PINECONE_API_KEY="your_pinecone_api_token"
    CONFLUENCE_API_KEY="your_confluence_api_token"
    CONFLUENCE_SPACE="your_space_key"
    PINECONE_INDEX_NAME="your_pinecone_index_database"
    PINECONE_CLOUD="your_pinecone_cloud_provider"
    PINECONE_REGION="your_pinecone_region_of_database"
    PINECONE_NAMESPACE="your_pinecone_namespace"
    # CONFLUENCE_MAX_DOCS="5" # Optional: limit documents for faster testing
    ```
    *   Replace placeholders with your actual Google API key and Confluence details.
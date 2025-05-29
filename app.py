import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
import logging

# Import utility functions
from utils.config import APP_CONFIG
from utils.confluence import load_confluence_documents, split_documents

# Updated import for RAG components
from utils.rag import (
    get_google_llm,
    get_google_embeddings,
    create_vector_store,
    create_conversational_rag_chain,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(page_title="Confluence RAG Chatbot (Gemini)", layout="wide")
st.title("📚 Confluence RAG Chatbot (Gemini Edition)")
st.markdown("Ask questions about your Confluence documents, powered by Google Gemini!")


# --- Helper Functions ---
@st.cache_resource(show_spinner="Loading and processing Confluence documents...")
def initialize_rag_components(_config):
    """
    Loads documents, creates embeddings, vector store, and the RAG chain.
    """
    max_docs_str = _config.get("confluence_max_docs")
    max_docs = int(max_docs_str) if max_docs_str and max_docs_str.isdigit() else 50

    logger.info("Initializing RAG components with Google Gemini...")
    documents = load_confluence_documents(
        url=_config["confluence_url"],
        username=_config["confluence_username"],
        api_token=_config["confluence_api_key"],
        space_key=_config["confluence_space"],
        max_docs=max_docs,
    )

    if not documents:
        st.error(
            "Failed to load documents from Confluence. Please check your .env configuration and Confluence access."
        )
        return None, None

    # chunks = split_documents(documents)
    # if not chunks:
    #     st.error(
    #         "No content to process after splitting documents. Check if the Confluence space has valid content."
    #     )
    #     return None, None

    try:
        embeddings_model = get_google_embeddings(
            google_api_key=_config["gemini_api_key"]
        )
    except Exception as e:
        st.error(
            f"Failed to initialize Google embeddings: {e}. Check your GOOGLE_API_KEY and API permissions."
        )
        logger.error(f"Embedding initialization error: {e}", exc_info=True)
        return None, None

    vector_store = create_vector_store(
        documents=documents,
        embeddings_model=embeddings_model,
        pinecore_api_key=_config["pinecone_api_key"],
    )
    if vector_store is None:
        st.error(
            "Failed to create vector store. This might be due to issues with document "
            "processing or embedding generation (e.g., API errors, rate limits). Check logs for details."
        )
        return None, None

    try:
        llm = get_google_llm(
            google_api_key=_config["gemini_api_key"],
            # model_name="gemini-2.5-pro-preview-03-25",
            temperature=0.6,
        )
    except Exception as e:
        st.error(
            f"Failed to initialize Google LLM: {e}. Check your GOOGLE_API_KEY and API permissions."
        )
        logger.error(f"LLM initialization error: {e}", exc_info=True)
        return None, None

    return llm, vector_store


# --- Session State Initialization (remains mostly the same) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False

# --- Sidebar for Configuration and Loading (remains mostly the same logic) ---
with st.sidebar:
    st.header("Configuration")
    st.markdown("**LLM Provider:** Google Gemini")  # Updated
    st.markdown(f"**Confluence URL:** `{APP_CONFIG['confluence_url']}`")
    st.markdown(f"**Confluence Space:** `{APP_CONFIG['confluence_space']}`")
    max_docs_str = APP_CONFIG.get("confluence_max_docs")
    max_docs_display = max_docs_str if max_docs_str else "All (default 50)"
    st.markdown(f"**Max Docs to Load:** `{max_docs_display}`")

    if not st.session_state.vector_store_loaded:
        if st.button("Load Confluence Docs & Build Gemini RAG"):  # Button text updated
            try:
                # Ensure API key is present before proceeding
                if not APP_CONFIG.get("gemini_api_key"):
                    st.error(
                        "GOOGLE_API_KEY not found in configuration. Please set it in your .env file."
                    )
                    st.stop()

                llm, vector_store = initialize_rag_components(APP_CONFIG)
                if llm and vector_store:
                    st.session_state.rag_chain = create_conversational_rag_chain(
                        llm, vector_store, st.session_state.conversation_memory
                    )
                    if st.session_state.rag_chain:
                        st.session_state.vector_store_loaded = True
                        st.success(
                            "Confluence documents loaded and Gemini RAG chain ready!"
                        )
                        logger.info("Gemini RAG chain initialized and ready.")
                        st.rerun()
                    else:
                        st.error("Failed to create Gemini RAG chain.")
                else:
                    logger.warning(
                        "LLM or Vector Store initialization failed for Gemini."
                    )
                    # Specific error messages for llm/vector_store failure are now in initialize_rag_components
            except ValueError as ve:  # Config errors
                st.error(f"Configuration Error: {ve}")
                logger.error(f"Configuration Error: {ve}")
            except Exception as e:  # Catch-all for other unexpected errors during setup
                st.error(f"An unexpected error occurred during setup: {e}")
                logger.error(
                    f"An unexpected error occurred during RAG initialization: {e}",
                    exc_info=True,
                )
    else:
        st.success("Gemini RAG System Ready!")
        if st.button("Reload Documents (Resets Chat)"):
            st.session_state.chat_history = []
            st.session_state.conversation_memory.clear()
            st.session_state.rag_chain = None
            st.session_state.vector_store_loaded = False
            initialize_rag_components.clear()
            st.info(
                "Document cache cleared. Click 'Load Confluence Documents' to reload."
            )
            st.rerun()

# --- Main Chat Interface (logic remains the same, but interacts with Gemini) ---
if not st.session_state.vector_store_loaded:
    st.info(
        "Please load Confluence documents from the sidebar to begin chatting with Gemini."
    )
else:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="✨"):  # Gemini avatar
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Ask your question about the Confluence documents...")

    if user_query and st.session_state.rag_chain:
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI", avatar="✨"):
            with st.spinner("Gemini is thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(
                        {"question": user_query}
                    )
                    ai_response_content = response.get(
                        "answer",
                        "Sorry, I couldn't find an answer or an error occurred.",
                    )

                    st.markdown(ai_response_content)
                    st.session_state.chat_history.append(
                        AIMessage(content=ai_response_content)
                    )

                    if "source_documents" in response and response["source_documents"]:
                        with st.expander("View Sources"):
                            for doc in response["source_documents"]:
                                source_url = doc.metadata.get("source", "N/A")
                                page_title = doc.metadata.get("title", "Untitled Page")
                                st.markdown(f"**Page:** {page_title}")
                                st.markdown(f"**Source URL:** {source_url}")
                                doc_length = 200
                                content_snippet = (
                                    doc.page_content[:doc_length] + "..."
                                    if len(doc.page_content) > doc_length
                                    else doc.page_content
                                )
                                st.caption(content_snippet)
                                st.markdown("---")
                except Exception as e:
                    # Handle potential API errors from Gemini during inference
                    if (
                        "API key not valid" in str(e)
                        or "User location is not supported" in str(e)
                        or "DeadlineExceeded" in str(e)
                    ):
                        st.error(
                            f"API Error with Gemini: {e}. Please check your API key, usage limits, or network."
                        )
                    else:
                        st.error(f"Error processing your query with Gemini: {e}")
                    logger.error(
                        f"Error during Gemini RAG chain invocation: {e}", exc_info=True
                    )

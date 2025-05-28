import logging

from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)


def encode_string_to_base64(input_string, encoding="utf-8"):
    import base64

    try:
        bytes_data = input_string.encode(encoding)
        base64_bytes = base64.b64encode(bytes_data)
        # ASCII characters is A-Z, a-z, 0-9, +, /, =
        base64_string = base64_bytes.decode("ascii")

        return base64_string
    except UnicodeEncodeError as e:
        print(
            f"Error: Could not encode string with '{encoding}' encoding. Details: {e}"
        )
        return None
    except LookupError:
        print(f"Error: Unknown encoding '{encoding}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Base64 encoding: {e}")
        return None


def get_google_llm(
    google_api_key: str,
    model_name: str = "gemini-2.5-flash-preview-04-17",
    temperature: float = 0.7,
):
    """Initializes and returns the Google Gemini LLM."""
    logger.info(f"Initializing Google LLM with model: {model_name}")
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=temperature,
            convert_system_message_to_human=True,  # Often needed for Gemini compatibility with existing LangChain patterns
        )
    except Exception as e:
        logger.error(f"Error initializing Google LLM: {e}")
        raise


def get_google_embeddings(
    google_api_key: str, model_name: str = "models/gemini-embedding-exp-03-07"
):
    """Initializes and returns Google Gemini embeddings."""
    logger.info(f"Initializing Google Embeddings with model: {model_name}")
    try:
        return GoogleGenerativeAIEmbeddings(
            model=model_name, google_api_key=google_api_key
        )
    except Exception as e:
        logger.error(f"Error initializing Google Embeddings: {e}")
        raise


def create_vector_store(documents: list, embeddings_model, pinecore_api_key):
    """Creates a Pinecore vector store from document chunks."""
    if not documents:
        logger.warning("No document chunks to create vector store from.")
        return None
    try:
        logger.info(f"Creating vector store from {len(documents)} chunks.")
        pc = Pinecone(api_key=pinecore_api_key)
        index_name = "landchain-confluence"
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )

        index = pc.Index(index_name)

        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings_model,
            pinecone_api_key=pinecore_api_key,
            namespace="landchain-confluence-dev",
        )

        ids = []
        for doc in documents:
            ids.append(
                doc.metadata.get(
                    "id",
                    encode_string_to_base64(doc.metadata.get("title")),
                )
            )

        vector_store.add_documents(ids=ids, documents=documents)
        logger.info("Vector store created successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        # It might be helpful to know if the error is during embedding creation
        if "DeadlineExceeded" in str(e) or "API key not valid" in str(
            e
        ):  # Example error checks
            logger.error(
                "This could be due to API rate limits, network issues, or an invalid API key for embeddings."
            )
        return None


def create_conversational_rag_chain(llm, vector_store, memory):
    """Creates a conversational RAG chain."""
    if vector_store is None:
        logger.error("Vector store is None. Cannot create RAG chain.")
        return None
    logger.info("Creating Conversational RAG chain.")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},  # Retrieve top 5 relevant chunks
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
    )
    logger.info("Conversational RAG chain created.")
    return qa_chain

import logging

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


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


def create_vector_store(documents: list, embeddings_model, app_config: dict):
    """Creates a Pinecore vector store from document chunks."""
    if not documents:
        logger.warning("No document chunks to create vector store from.")
        return None
    try:
        logger.info(f"Creating vector store from {len(documents)} chunks.")
        pc = Pinecone(api_key=app_config.get("pinecone_api_key"))
        index_name = app_config.get("pinecone_index_name")
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=app_config.get("pinecone_cloud"),
                    region=app_config.get("pinecone_region"),
                ),
            )

        index = pc.Index(index_name)

        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings_model,
            pinecone_api_key=app_config.get("pinecone_api_key"),
            namespace=app_config.get("pinecone_namespace"),
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

    system_prompt = """
    You are an AI assistant expert in providing information from Confluence documents.
    You have the ability to communicate in multiple languages, so you must respond in the language that the user uses to ask the question.
    Your role is to answer questions based *only* on the provided context from these documents.
    Follow these instructions carefully:
    1.  Analyze the 'Context' section below, which contains relevant excerpts from Confluence.
    2.  Based *solely* on this Context, answer the 'Question'.
    3.  If the Context does not contain the information to answer the Question, state clearly: "I cannot find an answer to that in the provided Confluence documents." Do not try to guess or use external knowledge.
    4.  If the user asks a question that is not related to the Confluence documents (e.g., a general knowledge question, or a greeting like "hello"), politely respond that you are designed to answer questions about the provided Confluence documentation. For example: "I am an assistant for our Confluence documents. Please ask me a question about them."
    5.  Keep your answers concise and directly relevant to the question and context.
    6.  Do not make up information or offer opinions.
    7.  Answer in a professional and helpful tone.
    8.  Finally, if the available data is not sufficient to answer the question, please provide the contact information of the person responsible for that project. If you cannot find the contact, politely decline to answer.

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""

    qa_chain_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=system_prompt,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": qa_chain_prompt},
    )
    logger.info("Conversational RAG chain created.")
    return qa_chain

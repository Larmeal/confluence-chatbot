import logging

from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings


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


def embedding_document(config: dict, documents: Document) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-exp-03-07",
        google_api_key=config.get("gemini_api_key"),
    )

    pc = Pinecone(api_key=config.get("pinecone_api_key"))
    index_name = "landchain-confluence"
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        pinecone_api_key=config.get("pinecone_api_key"),
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
    logging.info("Update Document Successfully")

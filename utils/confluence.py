from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_confluence_documents(
    url: str, username: str, api_token: str, space_key: str, max_docs: int = 50
) -> list[Document] | None:
    """
    Loads documents from a Confluence space.
    """
    try:
        logger.info(f"Loading documents from Confluence space: {space_key} at {url}")
        loader = ConfluenceLoader(
            url=url,
            username=username,
            api_key=api_token,
            space_key=space_key,
            include_restricted_content=False,
            include_attachments=False,
            keep_markdown_format=True,
            limit=max_docs,
        )
        # Load documents from a specific space, optionally limiting the number
        # For Confluence Cloud, include_attachments, include_comments, etc. can be used.
        # For Confluence Server, these might not be supported directly by the loader.
        # You might need to specify page_ids or cql for more granular control on server.
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from Confluence.")
        if not documents:
            logger.warning(
                "No documents loaded. Check space key, permissions, and Confluence API access."
            )
        return documents
    except Exception as e:
        logger.error(f"Error loading Confluence documents: {e}")
        return []


def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Splits loaded documents into smaller chunks.
    """
    if not documents:
        return []
    logger.info(f"Splitting {len(documents)} documents into chunks.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

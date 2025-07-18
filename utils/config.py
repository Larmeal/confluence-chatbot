import os
from dotenv import load_dotenv


def load_config():
    """Loads environment variables from .env file."""
    load_dotenv()
    config = {
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "confluence_url": os.getenv("CONFLUENCE_URL"),
        "confluence_username": os.getenv("CONFLUENCE_USER"),
        "confluence_api_key": os.getenv("CONFLUENCE_API_KEY"),
        "confluence_space": os.getenv("CONFLUENCE_SPACE"),
        "confluence_max_docs": os.getenv("CONFLUENCE_MAX_DOCS", "50"),
        "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME"),
        "pinecone_cloud": os.getenv("PINECONE_CLOUD"),
        "pinecone_region": os.getenv("PINECONE_REGION"),
        "pinecone_namespace": os.getenv("PINECONE_NAMESPACE"),
    }
    # Basic validation
    required_keys = [
        "pinecone_api_key",
        "gemini_api_key",
        "confluence_url",
        "confluence_username",
        "confluence_api_key",
        "confluence_space",
        "pinecone_index_name",
        "pinecone_cloud",
        "pinecone_region",
        "pinecone_namespace",
    ]
    for key in required_keys:
        if not config[key]:
            raise ValueError(f"Missing required environment variable: {key.upper()}")
    return config


# Load configuration once when the module is imported
APP_CONFIG = load_config()

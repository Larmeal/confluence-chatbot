from langchain_community.document_loaders import ConfluenceLoader
from langchain_core.documents import Document


def confluence_loader(config: dict) -> list[Document]:
    loader = ConfluenceLoader(
        url=config.get("confluence_uri"),
        username=config.get("email", "chutdanai.tho@gmail.com"),
        api_key=config.get("confluence_api_key"),
        space_key=config.get("space", None),
        page_ids=config.get("pages", None),
        include_attachments=False,
        keep_markdown_format=True,
    )

    return loader.load()

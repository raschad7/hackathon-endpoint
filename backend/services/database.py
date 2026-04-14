import hashlib
import json
import logging
import os

from docx import Document as DocxDocumentFile
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from backend.config import get_settings
from backend.services.llm import get_embedding_model

logger = logging.getLogger(__name__)


def _get_doc_path() -> str:
    settings = get_settings()
    return os.path.join(settings.data_dir, settings.doc_filename)


def _normalize_source_path(path: str) -> str:
    return os.path.abspath(path)


def _compute_source_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_vectorstore() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_persist_dir,
    )


def collection_is_empty(vectorstore: Chroma) -> bool:
    return vectorstore._collection.count() == 0


def _iter_docx_blocks(doc: DocxDocument):
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _table_to_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _load_docx_documents(path: str) -> list[Document]:
    doc = DocxDocumentFile(path)
    blocks: list[str] = []

    for block in _iter_docx_blocks(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
        else:
            text = _table_to_text(block)

        if text:
            blocks.append(text)

    if not blocks:
        return []

    return [
        Document(
            page_content="\n\n".join(blocks),
            metadata={"source": _normalize_source_path(path), "file_type": "docx"},
        )
    ]


def _load_json_documents(path: str) -> list[Document]:
    """Load a JSON file and convert entries into LangChain Documents."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    if isinstance(data, list):
        for i, entry in enumerate(data):
            if isinstance(entry, dict):
                content_key = next(
                    (k for k in ("content", "text", "description", "body") if k in entry),
                    None,
                )
                if content_key:
                    page_content = entry[content_key]
                    metadata = {
                        k: v
                        for k, v in entry.items()
                        if k != content_key and isinstance(v, (str, int, float, bool))
                    }
                else:
                    page_content = json.dumps(entry, ensure_ascii=False)
                    metadata = {}
                metadata["source_index"] = i
            else:
                page_content = str(entry)
                metadata = {"source_index": i}

            documents.append(Document(page_content=page_content, metadata=metadata))
    else:
        documents.append(
            Document(
                page_content=json.dumps(data, ensure_ascii=False),
                metadata={"source": _normalize_source_path(path), "file_type": "json"},
            )
        )

    return documents


def _load_source_documents(path: str) -> list[Document]:
    extension = os.path.splitext(path)[1].lower()

    if extension == ".docx":
        return _load_docx_documents(path)
    if extension == ".json":
        return _load_json_documents(path)

    raise ValueError(f"Unsupported data file type: {extension}. Supported types are .docx and .json.")


def _build_chunk_ids(source_hash: str, chunk_count: int) -> list[str]:
    return [f"{source_hash}:{index}" for index in range(chunk_count)]


def _annotate_chunks(chunks: list[Document], source_path: str, source_hash: str) -> list[Document]:
    for index, chunk in enumerate(chunks):
        chunk.metadata = {
            **chunk.metadata,
            "source_path": source_path,
            "source_hash": source_hash,
            "chunk_index": index,
        }
    return chunks


def _get_existing_source_chunks(vectorstore: Chroma, source_path: str) -> dict:
    return vectorstore.get(where={"source_path": source_path})


def ingest_document() -> None:
    """Load the configured data file, split it, and store embeddings in ChromaDB.

    Replaces existing chunks for the same source file and skips exact re-ingestion.
    """
    doc_path = _get_doc_path()

    if not os.path.exists(doc_path):
        logger.warning("Data file not found at %s - skipping ingestion.", doc_path)
        return

    source_path = _normalize_source_path(doc_path)
    source_hash = _compute_source_hash(source_path)
    vectorstore = get_vectorstore()

    logger.info("Loading data file: %s", source_path)
    documents = _load_source_documents(source_path)

    if not documents:
        logger.warning("No documents extracted from %s - skipping.", source_path)
        return

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    chunks = _annotate_chunks(chunks, source_path=source_path, source_hash=source_hash)
    chunk_ids = _build_chunk_ids(source_hash=source_hash, chunk_count=len(chunks))

    existing = _get_existing_source_chunks(vectorstore, source_path=source_path)
    existing_ids = existing.get("ids", [])
    existing_hashes = {
        metadata.get("source_hash")
        for metadata in existing.get("metadatas", [])
        if metadata and metadata.get("source_hash")
    }

    if set(existing_ids) == set(chunk_ids) and existing_hashes == {source_hash}:
        logger.info("Source file unchanged - skipping ingestion for %s.", source_path)
        return

    if existing_ids:
        logger.info("Replacing %d existing chunks for %s.", len(existing_ids), source_path)
        vectorstore.delete(ids=existing_ids)

    vectorstore.add_documents(chunks, ids=chunk_ids)
    logger.info("Ingested %d chunks into ChromaDB from %s.", len(chunks), source_path)

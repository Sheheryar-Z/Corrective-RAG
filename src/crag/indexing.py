"""Load PDFs, chunk them, embed them, and persist a FAISS index to disk.

The notebook re-embedded every PDF on every run. Here the index is built once
and reloaded afterwards, which saves both time and embedding cost.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from .config import Settings

log = logging.getLogger(__name__)


def load_pdfs(data_dir: Path) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in '{data_dir}'. Put your source PDFs there (see data/README.md)."
        )
    docs: list[Document] = []
    for path in pdfs:
        log.info("Loading %s", path.name)
        docs.extend(PyPDFLoader(str(path)).load())
    return docs


def chunk_documents(docs: list[Document], settings: Settings) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        # PDF extraction sometimes yields lone surrogates that break the embeddings API.
        c.page_content = c.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return chunks


def load_or_build_retriever(settings: Settings, rebuild: bool = False) -> VectorStoreRetriever:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    index_path = settings.index_dir

    if index_path.exists() and not rebuild:
        log.info("Loading FAISS index from %s", index_path)
        # Safe here: we only ever load an index this project wrote itself.
        store = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    else:
        chunks = chunk_documents(load_pdfs(settings.data_dir), settings)
        log.info("Embedding %d chunks with %s", len(chunks), settings.embedding_model)
        store = FAISS.from_documents(chunks, embeddings)
        index_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(index_path))

    return store.as_retriever(search_type="similarity", search_kwargs={"k": settings.top_k})

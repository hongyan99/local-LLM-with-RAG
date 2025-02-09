from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
import os
import hashlib
from typing import List, Optional
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from chroma_with_progress import ChromaWithProgress
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm  # Added for per-file progress


TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def generate_id(file_path: str) -> str:
    """
    Generates an idempotent ID based on the file path using SHA-256.
    
    Args:
        file_path (str): The file path to generate an ID for.
    
    Returns:
        str: A SHA-256 hash hex digest of the file path.
    """
    return hashlib.sha256(file_path.encode("utf-8")).hexdigest()


def load_documents_into_database(model_name: str, documents_path: str, persist_directory: Optional[str] = "chroma_db") -> ChromaWithProgress:
    """
    Loads documents from the specified directory into the Chroma database
    after splitting the text into chunks, and persists the database to disk.

    Args:
        model_name (str): The name of the embedding model.
        documents_path (str): Path to the documents directory.
        persist_directory (Optional[str]): Directory to persist the Chroma database.

    Returns:
        ChromaWithProgress: An instance of ChromaWithProgress with loaded documents.
    """

    print("Loading documents")
    raw_documents = load_documents(documents_path)
    documents = TEXT_SPLITTER.split_documents(raw_documents)
 
    # Generate IDs for each document using its file path.
    # It is assumed that the file path is stored under the "source" metadata.
    ids = [
        generate_id(f"{doc.metadata.get('source', '')}_{idx}")
        for idx, doc in enumerate(documents)
    ]

    print("Creating embeddings and loading documents into Chroma")
    db = ChromaWithProgress.from_documents(
        documents,
        OllamaEmbeddings(model=model_name),
        ids=ids,
        persist_directory=persist_directory,
    )
    return db


def load_documents(path: str) -> List[Document]:
    """
    Loads documents from the specified directory path.

    This function supports loading of PDF and Markdown documents by utilizing
    different loaders for each file type.

    Args:
        path (str): The path to the directory containing documents to load.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    loaders = {
        ".pdf": DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        ),
    }

    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        loaded_files = loader.load()
        for doc in tqdm(loaded_files, desc=f"Processing {file_type} files", unit="file"):
            docs.append(doc)
    return docs
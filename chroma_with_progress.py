import uuid
from typing import Any, Dict, List, Optional, Type, cast
from langchain_core.documents import Document

from tqdm import tqdm
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.chroma import Chroma


import chromadb
import chromadb.config


 
class ChromaWithProgress(Chroma):
    @classmethod
    def from_texts(
        cls: Type[Chroma],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,  # type: ignore[valid-type]
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a raw documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            texts (List[str]): List of texts to add to the collection.
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings
            collection_metadata (Optional[Dict]): Collection configurations.
                                                  Defaults to None.

        Returns:
            Chroma: Chroma vectorstore.
        """
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            **kwargs,
        )
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if hasattr(
            chroma_collection._client,  # type: ignore[has-type]
            "get_max_batch_size",  # for Chroma 0.5.1 and above
        ) or hasattr(
            chroma_collection._client,  # type: ignore[has-type]
            "max_batch_size",
        ):  # for Chroma 0.4.10 and above
            from chromadb.utils.batch_utils import create_batches

            for batch in tqdm(create_batches(
                api=chroma_collection._client,  # type: ignore[has-type]
                ids=ids,
                metadatas=metadatas,  # type: ignore[arg-type]
                documents=texts,
            ), desc="Processing batches"):
                chroma_collection.add_texts(
                    texts=batch[3] if batch[3] else [],
                    metadatas=batch[2] if batch[2] else None,  # type: ignore[arg-type]
                    ids=batch[0],
                )
        else:
            chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return chroma_collection

    @classmethod
    def from_documents(
        cls: Type["ChromaWithProgress"],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,  # type: ignore[valid-type]
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> "ChromaWithProgress":
        # If you want to simply delegate to the superclass method, do:
        instance = super().from_documents(
            documents,
            embedding=embedding,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            **kwargs,
        )
        # Cast the result to ChromaWithProgress.
        return cast(ChromaWithProgress, instance)
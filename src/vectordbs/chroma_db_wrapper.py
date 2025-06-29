# Class for Chroma DB Client.

import logging

from typing import Any, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from config import Config
from config import DBConfig
from vectordbs.vector_db_interface import VectorDBInterface


class ChromaDBWrapper(VectorDBInterface):
    """A Chroma Vector DB wrapper."""

    def __init__(
        self,
        config: Optional[DBConfig.ChromaConfig] = None,
        embedding: Optional[Embeddings] = None,
        logger: Optional[logging.Logger] = logging.getLogger(__name__),
    ):
        self.config = config or Config.load_chroma_db_config()
        self.embedding = embedding
        self.db_client = None
        self.logger = logger

    def connect(self, config: Optional[DBConfig.ChromaConfig] = None) -> Any:
        self.config = config or self.config
        self.db_client = Chroma(
            collection_name=self.config.collection_name,
            persist_directory=self.config.db_path,
            embedding_function=self.embedding,
            create_collection_if_not_exists=True,
        )
        return self.db_client

    def store_documents(
        self,
        docs: List[Document],
        embedding: Optional[Embeddings] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ) -> bool:
        """Store documents into a ChromaDB vector store.

        :param docs (list[langchain_core.documents.Document]): List of
        document objects to store.
        :param collection_name (Optional[str]): The name of the collection to
            Store embeddings to.
        :param persist_directory (Optional[str]): The directory to Store
            vector DB to.

        :returns bool: True if successful, else False.
        """
        try:
            if not self.db_client:
                self.db_client = self.connect()

            self.db_client.from_documents(
                documents=docs,
                embedding=embedding,
                collection_name=(
                    collection_name or self.config.collection_name
                ),  # no-qa
                persist_directory=persist_directory or self.config.db_path,
            )
            self.logger.info("Document stored.")
            return True
        except Exception as e:
            self.logger.error(e)
            print(e)
            return False

    def get_as_retriever(self, k: int = 4, **kwargs: Any) -> BaseRetriever:
        """Gets Chroma DB Retriever.

        :param k (int): Number of chunks to retrieve.
        :param kwargs (Any): Additional kwargs.

        :returns BaseRetriever: A retriever object/instance.
        """
        if self.db_client is None:
            self.db_client = Chroma(
                persist_directory=self.config.db_path,
                collection_name=(
                    self.config.collection_name
                    if self.config.collection_name
                    else "langchain"
                ),
                embedding_function=self.embedding,
            )
        retriever = self.db_client.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": 0.6},
        )
        self.logger.info("chroma vector retriever loaded")
        return retriever

    def add_document(self, docs: List[Document]) -> List[str]:
        """Updates existing ChromaDB vector store with additional documents.

        :param docs (List[langchain_core.documents.Document]): List of
            document objects to add.

        :returns List[str]: List of IDs of the newly added documents.
        """
        return self.db_client.add_documents(documents=docs)

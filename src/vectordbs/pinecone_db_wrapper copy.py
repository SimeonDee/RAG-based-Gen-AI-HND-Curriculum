# Class for Pinecone DB Client.

import logging

from typing import Any, List, Optional

from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from config import Config
from config import DBConfig
from vectordbs.vector_db_interface import VectorDBInterface


class PineconeDBWrapper(VectorDBInterface):
    """A Pinecone Vector DB wrapper."""

    def __init__(
        self,
        config: Optional[DBConfig.PineconeConfig] = None,
        embedding: Optional[Embeddings] = None,
        logger: Optional[logging.Logger] = logging.getLogger(__name__),
    ):
        self.config = config or Config.load_pinecone_config()
        self.embedding = embedding
        self.db_client = None
        self.logger = logger

    def __create_index(self, index_name: str):
        """Creates index in Pinecone if not already existing.

        :param index_name (str): Name of the index to create.
        """

        if index_name not in self.db_client.list_indexes().names():  # no-qa
            spec = ServerlessSpec(
                cloud=self.config.cloud,
                region=self.config.region,
            )

            self.db_client.create_index(
                name=index_name,
                dimension=self.embedding.dimension,
                metric="cosine",
                spec=spec,
            )
            # See that it is empty
            print("Index before upsert:")
            print(self.db_client.Index(index_name).describe_index_stats())
            print("\n")

    def connect(self, config: Optional[DBConfig.PineconeConfig] = None) -> Any:
        self.config = config or self.config
        self.db_client = Pinecone(api_key=self.config.api_key)

        # create index if not already existing
        self.__create_index(index_name=self.config.index_name)
        return self.db_client

    def store_documents(self, docs: List[Document]) -> bool:
        """Store documents into a Pinecone vector store.

        :param docs (list[langchain_core.documents.Document]): List of
            document objects to store.

        :returns bool: True if successful, else False.
        """
        try:
            if not self.db_client:
                self.db_client = self.connect()

            # Add documents to index for the given namespace
            self.db_client.from_documents(
                documents=docs,
                embedding=self.embedding,
                index_name=self.config.index_name,
                namespace=self.config.namespace,
            )

            self.logger.info("Document stored.")
            return True
        except Exception as e:
            self.logger.error(e)
            print(e)
            return False

    def get_as_retriever(
        self, embedding: Embeddings, k: int = 4, **kwargs: Any
    ) -> BaseRetriever:
        """Gets Pinecone DB Retriever.

        :param embeddings (Optional[Embeddings]): Embedding object to use
        :param k (int): Number of chunks to retrieve.
        :param kwargs (Any): Additional kwargs.

        :returns BaseRetriever: A retriever object/instance.
        """
        if self.db_client is None:
            self.db_client = self.connect()

        retriever = self.db_client.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": 0.01},
        )
        self.logger.info("Pinecone vector retriever loaded")
        return retriever

    def add_document(self, docs: List[Document]) -> List[str]:
        """Updates existing Pinecone vector store with additional documents.

        :param docs (List[langchain_core.documents.Document]): List of
            document objects to add.

        :returns List[str]: List of IDs of the newly added documents.
        """
        if self.db_client is None:
            self.db_client = self.connect()
        return self.db_client.add_documents(documents=docs)

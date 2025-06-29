# Interface to be implemented for all Supported VectorDBs

from abc import ABC, abstractmethod
from typing import Any, List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


class VectorDBInterface(ABC):

    @abstractmethod
    def connect(self, config: Optional[Any] = None) -> Any:
        """Connects to a Vector DB source.

        :param config (Optional[Any]): The database configuration data.

        :returns Any: An instance of the connected Vector DB.
        """
        pass

    def store_documents(
        self,
        docs: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> bool:
        """Stores documents into Vector db using supplied embedding object.

        :param docs (List[angchain_core.documents.Document]):
            Documents to store.
        :param embedding (langchain_core.embeddings.Embeddings): An embedding
            instance to use.
        :param kwargs (Any): Additional kewword args.

        :returns bool: Status of data store. True if successful.
        """
        pass

    def add_document(self, docs: List[Document]) -> List[str]:
        """Updates existing Vector store with additional documents.

        :param docs (List[langchain_core.documents.Document]): List of
            document objects to add.

        :returns List[str]: List of IDs of the newly added documents.
        """
        pass

    def get_as_retriever(
        self,
        embedding: Embeddings,
        k: int = 4,
        **kwargs: Any,
    ) -> BaseRetriever:
        """Returns an instance of the object as a retriever.

        :param embedding (langchain_core.embeddings.Embeddings): An embedding
            instance to use.
        :param k (int): The number of chunks to retrieve.
        :param kwargs (Any): Additional kewword args.

        :returns BaseRetriever: The retriever
        """
        pass

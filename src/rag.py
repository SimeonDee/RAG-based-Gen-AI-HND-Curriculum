import logging
import os

from typing import Any, List, Optional, Union

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import TextSplitter

# RAG Pipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# custom
from vectordbs.vector_db_interface import VectorDBInterface

SYSTEM_PROMPT = """You are an helpful AI assistant.
    Use the given context to answer the question.
    If you don't know the answer, say you don't know.
    Use three sentences maximum and keep the answer concise.
    Context: {context}
"""


class RAG_App_Wrapper:
    def __init__(
        self,
        llm,
        db_client: VectorDBInterface,
        embedding: Embeddings,
        text_splitter: TextSplitter,
        logger: Optional[logging.Logger] = logging.getLogger(__name__),
    ):
        self.llm = llm
        self.db_client = db_client
        self.embedding = embedding
        self.splitter = text_splitter
        self.logger = logger

    # step
    def __get_pdf_loader(
        self,
        pdf_data_path: str,
    ) -> Union[PyPDFLoader, PyPDFDirectoryLoader]:
        """Gets a PDF loader object.

        :param pdf_data_path (str): The path to the PDF file or folder
            containing PDF files.

        :returns Union[PyPDFLoader, PyPDFDirectoryLoader]: An loader object.
        """
        try:
            self.logger.info("getting loader")
            if os.path.isfile(pdf_data_path) and ".pdf" in pdf_data_path:
                loader = PyPDFLoader(pdf_data_path)
                self.logger.info("pdf file detected")
            elif (
                os.path.isdir(pdf_data_path)
                and len(os.listdir(pdf_data_path)) > 0  # no-qa
            ):  # no-qa
                loader = PyPDFDirectoryLoader(
                    pdf_data_path,
                    glob="**/[!.]*.pdf",
                )
                self.logger.info("Folder detected")
            else:
                raise ValueError(
                    "pdf_data_path must either be a PDF file path"
                    "or directory path containing PDF files"
                )
            self.logger.info("loader ready")
            return loader
        except ValueError as e:
            self.logger.error(e)
            raise

        # step

    def __load_pdf_documents(self, pdf_data_path: str) -> List[Document]:
        """Loads PDF documents from a path specified.

        :param pdf_data_path (str): The path to the PDF file or folder
            containing PDF files.

        :returns list[langchain_core.documents.Document]: list of extracted
                Document objects.
        """
        loader = self.__get_pdf_loader(pdf_data_path)
        documents = loader.load()
        self.logger.info("PDF documents loaded")
        return documents

    # step
    def __split_documents(self, documents: list[Document]) -> List[Document]:
        """Splits list of langchain documents into chunks.

        :param documents (list[langchain_core.documents.Document]): List of
            document objects to chunk.

        :returns list[langchain_core.documents.Document]]: list of splitted
            chunks.
        """
        splitted_text = self.splitter.split_documents(documents)
        self.logger.info("Documents splitted")
        return splitted_text

    # step
    def __store_documents_in_vector_store(
        self,
        documents: List[Document],
    ) -> bool:
        """Store documents into a ChromaDB vector store.

        :param documents (list[langchain_core.documents.Document]): List of
            document objects to store.

        :returns bool: True if success, else False.
        """
        self.db_client.connect()
        self.db_client.store_documents(
            docs=documents,
            embedding=self.embedding,
        )
        self.logger.info("Documents stored in Chroma VectorDB")
        return True

    # step
    def __load_retriever(self, k: int = 4, **kwargs: Any) -> BaseRetriever:
        """Loads Vector store as retriever.

        :param k (int): The number of chunks to retrieve as context.
        :param kwargs (Any): Additional kwargs.

        :returns BaseRetriever: A retriever object/instance.
        """
        return self.db_client.get_as_retriever(
            embedding=self.embedding,
            k=k,
            **kwargs,
        )

    # step
    def embed_documents(self, documents: List[Document]) -> list[list[float]]:
        """Create embedings of documents.

        :param documents (list[langchain_core.documents.Document]): List of
            document objects to embed.

        :returns list[list[float]]: list of document embeddings.
        """
        doc_texts = [doc.page_content for doc in documents]
        docs_embeddings = self.embedding.embed_documents(texts=doc_texts)
        self.logger.info("Documents embedded")
        return docs_embeddings

    # step
    def __get_response(self, query: str, retriever: BaseRetriever):
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(
                self.llm,
                prompt,
            )
            chain = create_retrieval_chain(retriever, question_answer_chain)

            response = chain.invoke({"input": query})
            log_data = {"query": query, "response": response}
            self.logger.info(log_data)
            return response
        except Exception as e:
            self.logger.error(e)
            raise

    # Datastore Pipeline
    def extract_pdf_and_store_external_knowlede(
        self,
        pdf_data_path: str,
    ) -> None:
        """Extract PDF texts and store in vector DB as new knowledge.

        :param pdf_data_path (str): The path to the PDF file or path to folder
            containing PDF files.

        :returns None
        """
        documents = self.__load_pdf_documents(pdf_data_path=pdf_data_path)
        splitted_docs = self.__split_documents(documents)
        self.__store_documents_in_vector_store(documents=splitted_docs)
        self.logger.info("External knowledge stored in vectorDB successfully")

    # Data update pipeline
    def add_more_pdf_external_knowledge(
        self,
        pdf_data_path: str,
    ) -> List[str]:
        """Add aditional external knowledge to update vector DB.

        :param pdf_data_path (str): The path to the PDF file or folder
            containing PDF files.

        :returns list[langchain_core.documents.Document]: list of extracted
                Document objects.
        """
        documents = self.__load_pdf_documents(pdf_data_path)
        splitted_docs = self.__split_documents(documents)
        ids = self.db_client.add_document(docs=splitted_docs)
        self.logger.info("Additional knowledge added to vectorDB successfully")
        return ids

    # RAG Query Pipeline
    def process_query(self, query: str) -> str:
        """Processes users query for knowledge retrieva=er using the llm and
            RAG.

        :param query (str): Input query.

        :returns str: The response from the LLM.
        """
        try:
            retriever = self.__load_retriever(k=4)
            if not retriever:
                raise ValueError(
                    "One of 'retriever' or 'db_path' param must be supplied."
                )  # no-qa

            response = self.__get_response(query=query, retriever=retriever)
            return response
        except ValueError as e:
            self.logger.error(e)
            raise

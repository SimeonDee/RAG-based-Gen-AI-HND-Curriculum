import logging
import os

# from dataclasses import dataclass, Field
from typing import List, Optional, Union

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

# RAG Pipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


TEXT_SPLITTER = RecursiveCharacterTextSplitter()


class EmbeddingConfig:
    openai_embedding: OpenAIEmbeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
    )


# llm RAG Pipeline
llm = ChatOpenAI(model="gpt-4o")  # Calling the GPT-4o
SYSTEM_PROMPT = """You are an helpful AI assistant.
    Use the given context to answer the question.
    If you don't know the answer, say you don't know.
    Use three sentences maximum and keep the answer concise.
    Context: {context}
"""


class Pdf_RAG_GPT_LLM:

    def __init__(self, pdf_data_path: str):
        if os.path.isfile(pdf_data_path) and ".pdf" in pdf_data_path:
            self.document_loader = PyPDFLoader(pdf_data_path)
        elif os.path.isdir(pdf_data_path):
            self.document_loader = PyPDFDirectoryLoader(
                pdf_data_path,
            )
        else:
            raise ValueError(
                "pdf_data_path must either be a PDF file path"
                "or directory path containing PDF files"
            )
        self.TEXT_SPLITTER = RecursiveCharacterTextSplitter()
        self.embedding = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")


# step
def get_loader(pdf_data_path: str) -> Union[PyPDFLoader, PyPDFDirectoryLoader]:
    """Gets a PDF loader object.

    :param pdf_data_path (str): The path to the PDF file or folder containing
            PDF files.

    :returns Union[PyPDFLoader, PyPDFDirectoryLoader]: An loader object.
    """
    try:
        logger.info("getting loader")
        if os.path.isfile(pdf_data_path) and ".pdf" in pdf_data_path:
            loader = PyPDFLoader(pdf_data_path)
            logger.info("pdf file detected")
        elif (
            os.path.isdir(pdf_data_path) and len(os.listdir(pdf_data_path)) > 0
        ):  # no-qa
            loader = PyPDFDirectoryLoader(
                pdf_data_path,
                glob="**/[!.]*.pdf",
            )
            logger.info("Folder detected")
        else:
            raise ValueError(
                "pdf_data_path must either be a PDF file path"
                "or directory path containing PDF files"
            )
        logger.info("loader ready")
        return loader
    except ValueError as e:
        logger.error(e)
        raise


# step
def load_pdf_documents(pdf_data_path: str) -> List[Document]:
    """Loads PDF documents from a path specified.

    :param pdf_data_path (str): The path to the PDF file or folder containing
            PDF files.

    :returns list[langchain_core.documents.Document]: list of extracted
            Document objects.
    """
    loader = get_loader(pdf_data_path)
    documents = loader.load()
    logger.info("PDF documents loaded")
    return documents


# step
def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Splits list of langchain documents into chunks.

    :param documents (list[langchain_core.documents.Document]): List of
        document objects to chunk.
    :param chunk_size (int): number of tokens per chunk.
    :param chunk_overlap (int): number of tokens to overlap between chunks.

    :returns list[langchain_core.documents.Document]]: list of splitted chunks.
    """
    TEXT_SPLITTER._chunk_size = chunk_size
    TEXT_SPLITTER._chunk_overlap = chunk_overlap
    splitted_text = TEXT_SPLITTER.split_documents(documents)
    logger.info("Documents splitted")
    return splitted_text


# step
def embed_documents(
    documents: List[Document], embedding: Optional[Embeddings] = None
) -> list[list[float]]:
    """Create embedings of documents.

    :param documents (list[langchain_core.documents.Document]): List of
        document objects to embed.
    :param embedding (Optional[Embeddings]): Embedding object to use

    :returns list[list[float]]: list of document embeddings.
    """
    embedding = embedding or EmbeddingConfig.openai_embedding
    doc_texts = [doc.page_content for doc in documents]
    docs_embeddings = embedding.embed_documents(texts=doc_texts)
    logger.info("Documents embedded")
    return docs_embeddings


# step
def store_documents_in_vector_store(
    documents: List[Document],
    embedding: Optional[Embeddings] = None,
    persist_directory: str = "",
    collection_name: str = "",
) -> Chroma:
    """Store documents into a ChromaDB vector store.

    :param documents (list[langchain_core.documents.Document]): List of
        document objects to store.
    :param embedding (Optional[Embeddings]): Embedding object to use
    :param persist_directory (str): The directory to Store vector DB to.
    :param collection_name (str): The name of the collection to Store
        embeddings to.

    :returns Chroma: A Chroma object/instance.
    """
    chroma_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    logger.info("Documents stored in Chroma VectorDB")
    return chroma_db


# step
def load_chroma_vector_store_retriever(
    db_path: str,
    collection_name: str = "",
    k: int = 4,
    embedding: Optional[Embeddings] = None,
):
    """Store documents into a ChromaDB vector store.

    :param db_path (str): The directory to load vector DB from.
    :param collection_name (str): The name of the collection to load
        chunks from.
    :param k (int): The number of chunks to retrieve as context.
    :param embeddings (Optional[Embeddings]): Embedding object to use

    :returns VectorStoreRetriever: A VectorStoreRetriever object/instance.
    """
    db_client = Chroma(
        persist_directory=db_path,
        collection_name=collection_name if collection_name else "langchain",
        embedding_function=embedding,
    )
    retriever = db_client.as_retriever(search_kwargs={"k": k})
    logger.info("chroma vector retrieval loaded")
    return retriever


# step
def get_response(query: str, retriever: BaseRetriever):
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

        response = chain.invoke({"input": query})
        log_data = {"query": query, "response": response}
        logger.info(log_data)
        return response
    except Exception as e:
        logger.error(e)
        raise


# Pipeline
def extract_and_store_external_knowlede(
    doc_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 200,
    embedding: Optional[Embeddings] = None,
    persist_directory: str = "",
    collection_name: str = "",
):
    documents = load_pdf_documents(doc_path)
    splitted_docs = split_documents(documents, chunk_size, chunk_overlap)
    db_instance = store_documents_in_vector_store(
        documents=splitted_docs,
        embedding=embedding or EmbeddingConfig.openai_embedding,
        persist_directory=persist_directory,
        collection_name=collection_name or "langchain",
    )
    logger.info(
        "Additional external knowledge stored in vectorDB successfully"
    )  # no-qa
    return db_instance


# Pipeline
def process_query(
    query: str,
    db_path: str = "",
    collection_name: str = "langchain",
    retriever: Optional[BaseRetriever] = None,
) -> str:
    try:
        if not retriever and not db_path:
            raise ValueError(
                "One of 'retriever' or 'db_path' param must be supplied."
            )  # no-qa

        if db_path:
            retriever = load_chroma_vector_store_retriever(
                db_path=db_path,
                collection_name=collection_name,
                embedding=EmbeddingConfig.openai_embedding,
                k=4,
            )
        response = get_response(query=query, retriever=retriever)
        return response
    except ValueError as e:
        logger.error(e)
        raise

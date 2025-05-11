import os
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

# RAG Pipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("CHROMA_DB_PATH")
DB_COLLECTION_NAME = os.getenv("CHROMA_DB_COLLECTION_NAME")

text_splitter = RecursiveCharacterTextSplitter()
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

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
        self.text_splitter = RecursiveCharacterTextSplitter()
        self.embedding = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")


def get_loader(pdf_data_path: str) -> Union[PyPDFLoader, PyPDFDirectoryLoader]:
    """Gets a PDF loader object.

    :param pdf_data_path (str): The path to the PDF file or folder containing
            PDF files.

    :returns Union[PyPDFLoader, PyPDFDirectoryLoader]: An loader object.
    """
    try:
        if os.path.isfile(pdf_data_path) and ".pdf" in pdf_data_path:
            loader = PyPDFLoader(pdf_data_path)
        elif os.path.isdir(pdf_data_path):
            loader = PyPDFDirectoryLoader(
                pdf_data_path,
            )
        else:
            raise ValueError(
                "pdf_data_path must either be a PDF file path"
                "or directory path containing PDF files"
            )
        return loader
    except ValueError:
        raise


def load_pdf_documents(pdf_data_path: str) -> List[Document]:
    """Loads PDF documents from a path specified.

    :param pdf_data_path (str): The path to the PDF file or folder containing
            PDF files.

    :returns list[langchain_core.documents.Document]: list of extracted
            Document objects.
    """
    loader = get_loader(pdf_data_path)
    documents = loader.load()
    return documents


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
    text_splitter._chunk_size = chunk_size
    text_splitter._chunk_overlap = chunk_overlap
    splitted_text = text_splitter.split_documents(documents)
    return splitted_text


def embed_documents(documents: List[Document]):
    doc_texts = [doc.page_content for doc in documents]
    docs_embeddings = embedding.embed_documents(texts=doc_texts)
    return docs_embeddings


def store_documents_in_vector_store(documents: List[Document]):
    chroma_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=DB_PATH,
        collection_name=DB_COLLECTION_NAME,
    )
    return chroma_db


def load_vector_store_retriever(
    db_path: str, collection_name: Optional[str] = None, k=4
):
    db_client = Chroma(
        persist_directory=db_path,
        collection_name=collection_name if collection_name else "langchain",
        embedding=embedding,
    )
    retriever = db_client.as_retriever(search_kwargs={"k": k})
    return retriever


def get_response(query: str):
    retriever = load_vector_store_retriever(
        db_path=DB_PATH, collection_name=DB_COLLECTION_NAME, k=4
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    response = chain.invoke({"input": query})
    return response


class RAG_Query:
    def __init__(self, db_path: str):
        self.vector_store = Chroma.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8, "k": 4},
        )

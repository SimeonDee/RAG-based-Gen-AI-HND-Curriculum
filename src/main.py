import logging
import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from config import Config
from vectordbs.chroma_db_wrapper import ChromaDBWrapper
from rag import RAG_App_Wrapper

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logs.log",
    filemode="w",
    encoding="utf-8",
    level=logging.INFO,
    # format="%(asctime)s - %(filename)s - %(levelname)s: %(message)s",
    format="{asctime} - {filename} - {levelname}: {message}",
    style="{",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

load_dotenv()

ENV = os.getenv("ENV")
DATA_PATH = os.getenv("DATA_PATH")

# Load configurations
openai_config = Config.load_openai_config()
openai_embedding_config = Config.load_openai_embedding_config()
chroma_config = Config.load_chroma_db_config()


def main():
    """Main function, for testing the RAG wrapper class."""
    # Components
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    openai_embedding = OpenAIEmbeddings(
        model=openai_embedding_config.model_name,
    )

    chroma_db_client = ChromaDBWrapper(
        config=chroma_config,
        embedding=openai_embedding,
        logger=logger,
    )

    gpt_llm = ChatOpenAI(model=openai_config.model_name)

    # Instantiating RAG object
    rag_object = RAG_App_Wrapper(
        llm=gpt_llm,
        db_client=chroma_db_client,
        embedding=openai_embedding,
        text_splitter=text_splitter,
        logger=logger,
    )

    # if in dev environment, load and store the extracted external knowledge
    if ENV.startswith("dev"):
        rag_object.extract_pdf_and_store_external_knowlede(
            pdf_data_path=DATA_PATH,
        )

    print("\n")
    print("\t\tChat with AI")
    print("\t\t------------\n\n")
    while True:
        user_input = input("User: \t")
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Bot: \t Thank you, bye for now.", end="\n\n")
            break
        try:
            result = rag_object.process_query(query=user_input)
            print(f"Bot: \t {result['answer']}", end="\n\n")
            print("-" * 100)
            print()
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":
    main()

import os
from app import (
    extract_and_store_external_knowlede,
    load_chroma_vector_store_retriever,
    process_query,
    EmbeddingConfig,
)

DATA_PATH = os.getenv("DATA_PATH")
DB_PATH = os.getenv("CHROMA_DB_PATH")
DB_COLLECTION_NAME = os.getenv("CHROMA_DB_COLLECTION_NAME")
ENV = os.getenv("ENV")


def main():
    if ENV == "development":
        db_client = extract_and_store_external_knowlede(
            doc_path=DATA_PATH,
            embedding=EmbeddingConfig.openai_embedding,
            persist_directory=DB_PATH,
            collection_name=DB_COLLECTION_NAME,
        )
        retriever = db_client.as_retriever(search_kwargs={"k": 4})

    else:  # production
        retriever = load_chroma_vector_store_retriever(
            db_path=DB_PATH,
            collection_name=DB_COLLECTION_NAME or "langchain",
            embedding=EmbeddingConfig.openai_embedding,
            k=5,
        )

    print("")
    while True:
        user_input = input("User: \t")
        if "quit" in user_input.lower() or user_input.lower() == "q":
            break
        try:
            result = process_query(query=user_input, retriever=retriever)
            print(f"Bot: \t {result['answer']}", end="\n\n")
            print("-" * 100)
            print()
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":
    main()

import os
from dotenv import load_dotenv

load_dotenv()


# VectorDB Configs


class DBConfig:
    class ChromaConfig:
        db_path: str = os.getenv("CHROMA_DB_PATH") or ""
        collection_name: str = os.getenv("CHROMA_DB_COLLECTION_NAME") or ""

        def __str__(self):
            return (
                "ChromaConfig(\n"
                f"   db_path={self.db_path},\n"
                f"   collection_name={self.collection_name}\n"
                ")"
            )

    class PineconeConfig:
        api_key: str = os.getenv("PINECONE_API_KEY")
        cloud: str = os.getenv("PINECONE_CLOUD") or "aws"
        region: str = os.getenv("PINECONE_REGION") or "us-east-1"
        index_name: str = os.getenv("INDEX_NAME")
        namespace: str = os.getenv("NAMESPACE")
        similarity_metric = os.getenv("SIMILARITY_METRIC") or "cosine"

        def __str__(self):
            return (
                f"PineconeConfig(\n"
                f"   api_key={self.api_key},\n"
                f"   cloud={self.cloud},\n"
                f"   region={self.region},\n"
                f"   index_name={self.index_name},\n"
                f"   namespace={self.namespace},\n"
                f"   similarity_metric={self.similarity_metric}\n"
                ")"
            )

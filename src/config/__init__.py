import os
from dotenv import load_dotenv
from .vector_db_configs import DBConfig
from .embedding_configs import EmbeddingConfig

load_dotenv()


class OpenaiConfig:
    api_key: str = os.getenv("OPENAI_API_KEY")
    model_name: str = os.getenv("OPENAI_MODEL_NAME") or "gpt-4o"

    def __str__(self):
        return f"""OpenaiConfig(
                        api_key={self.api_key},
                        model_name={self.model_name}
                    )"""


class Config:
    @staticmethod
    def load_chroma_db_config() -> DBConfig.ChromaConfig:
        """Loads the Chroma DB Configuration data."""
        return DBConfig.ChromaConfig()

    @staticmethod
    def load_pinecone_config() -> DBConfig.PineconeConfig:
        """Loads the Pinecone DB Configuration data."""
        return DBConfig.PineconeConfig()

    @staticmethod
    def load_openai_config() -> OpenaiConfig:
        """Loads the OpenAI Configuration data."""
        return OpenaiConfig()

    @staticmethod
    def load_openai_embedding_config():
        """Loads OpenAI Embedding."""
        return EmbeddingConfig.OpenAIEmbeddingConfig()


# conf = Config.load_openai_config()
# print(conf.api_key)

# embeding = Config.load_openai_embedding_config()
# print(embeding)

# Embeddings Config File

import os
from dotenv import load_dotenv

load_dotenv()


class EmbeddingConfig:
    class OpenAIEmbeddingConfig:
        model_name: str = (
            os.getenv("OPEN_AI_EMBEDDING_MODEL_NAME")
            or "text-embedding-ada-002"  # no-qa
        )  # no-qa

        def __str__(self):
            return f"OpenAIEmbeddingConfig(model_name='{self.model_name}')"

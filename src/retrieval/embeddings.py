import logging
from typing import List

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Wrapper for generating text embeddings using a lightweight model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedding model wrapper.
        
        Args:
            model_name: The name of the sentence-transformers model to use.
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            List of embedding vectors (list of floats).
        """
        if not texts:
            return []
        self._load_model()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
        
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: A string to embed.
            
        Returns:
            A single embedding vector (list of floats).
        """
        if not text.strip():
            # Return a zero vector or handle gracefully. Let's return a small zero vector.
            # all-MiniLM-L6-v2 has dimension 384
            return [0.0] * 384
            
        self._load_model()
        embedding = self._model.encode(text, show_progress_bar=False)
        return embedding.tolist()

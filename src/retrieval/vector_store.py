import logging
import chromadb
from typing import List, Dict, Any, Optional
from src.schemas.data_models import Requirement

logger = logging.getLogger(__name__)

class RequirementVectorStore:
    """Manages the storage and retrieval of requirement embeddings."""
    
    def __init__(self, embedding_model, collection_name: str = "requirements", persist_directory: Optional[str] = None):
        self.embedding_model = embedding_model
        
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.EphemeralClient()
            
        # Use cosine similarity space for easier thresholding (similarity = 1 - distance)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    def add_requirements(self, requirements: List[Requirement]):
        """
        Embeds and stores a list of requirements in the vector store.
        """
        if not requirements:
            return
            
        # Filter valid requirements
        valid_reqs = [r for r in requirements if r.original_text and r.original_text.strip()]
        if not valid_reqs:
            return
            
        unique_ids = set()
        deduplicated_reqs = []
        for r in valid_reqs:
            req_id = r.requirement_id or f"temp-{id(r)}"
            if req_id not in unique_ids:
                unique_ids.add(req_id)
                deduplicated_reqs.append((req_id, r))
                
        if not deduplicated_reqs:
            return
            
        texts = [r.original_text for _, r in deduplicated_reqs]
        ids = [req_id for req_id, _ in deduplicated_reqs]
        embeddings = self.embedding_model.embed_documents(texts)
        
        metadatas = []
        for req_id, r in deduplicated_reqs:
            metadatas.append({
                "requirement_id": r.requirement_id or "unknown",
                "source_page": r.source_page,
                "original_text": r.original_text
            })
            
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        
    def search_similar(self, query_text: str, top_k: int = 3, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Searches for requirements similar to the query text.
        
        Returns:
            List of dicts containing id, metadata, document, and similarity.
        """
        if not query_text.strip():
            return []
            
        query_embedding = self.embedding_model.embed_query(query_text)
        
        # Query ChromaDB
        # We ask for top_k + 1 in case the query itself is returned (we'll filter it later)
        # But for now just ask for top_k + 1 to be safe.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k + 1,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []
            
        similar_items = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            # Since hnsw:space is cosine, distance is cosine distance.
            # Cosine similarity = 1 - cosine distance
            similarity = 1.0 - distance
            
            if similarity >= threshold:
                similar_items.append({
                    "id": results['ids'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "document": results['documents'][0][i],
                    "similarity": similarity
                })
                
        # Sort by similarity descending
        similar_items.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_items

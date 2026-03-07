from sentence_transformers import CrossEncoder
import os

class RerankerModel:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        # Check if local model exists
        local_path = os.path.join(os.getcwd(), "models_cache", "bge-reranker-base")
        if os.path.exists(local_path):
            print(f"Loading reranker model from local path: {local_path}")
            self.model_name = local_path
        else:
            self.model_name = model_name
            
        print(f"Loading reranker model: {self.model_name}...")
        # device="cpu" is safer for Mac without MPS/CUDA setup for now, or let library decide
        self.model = CrossEncoder(self.model_name, max_length=512) 

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """
        Rerank a list of documents based on the query.
        Returns a list of scores.
        """
        if not docs:
            return []
            
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs)
        return scores.tolist()

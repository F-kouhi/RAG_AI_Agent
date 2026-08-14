import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from embeddings import Embedder


class Retriever:
    """
    FAISS-based retriever for Persian RAG.
    Uses precomputed chunks, embeddings, and metadata.
    """

    def __init__(
        self,
        chunks_path: str = "Data/chunks.json",
        metadata_path: str = "Data/metadata.json",
        embeddings_path: str = "Data/embeddings.npy",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        # Load chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[str] = json.load(f)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata: List[Dict] = json.load(f)

        # Load embeddings
        self.embeddings: np.ndarray = np.load(embeddings_path).astype("float32")
        assert len(self.chunks) == self.embeddings.shape[0], "Chunks and embeddings count mismatch"

        # Normalize for cosine similarity
        if normalize:
            faiss.normalize_L2(self.embeddings)

        # FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity if normalized
        self.index.add(self.embeddings)

        # Embedder for queries
        self.embedder = Embedder(device=device)

        print(f"[Retriever] Loaded {len(self.chunks)} chunks with metadata and embeddings")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        abs_min_score: float = 0.30,
        rel_score_drop: float = 0.7,
        fallback_top_k: int = 3,
    ) -> List[Dict]:
        """
        Retrieve top_k relevant chunks for a query.
        Returns list of dicts: {text, score, metadata}
        """
        query_emb = self.embedder.embed_query_api(query).astype("float32")
        faiss.normalize_L2(query_emb.reshape(1, -1))

        scores, indices = self.index.search(query_emb.reshape(1, -1), self.chunks.__len__())
        scores, indices = scores[0], indices[0]

        max_score = scores[0] if scores.size > 0 else 0.0
        results = []

        for score, idx in zip(scores, indices):
            if score < abs_min_score:
                continue
            if score < max_score * rel_score_drop:
                continue
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })
            if len(results) >= top_k:
                break

        # Fallback: if no results pass threshold, take top fallback_top_k
        if len(results) == 0:
            for i in range(min(fallback_top_k, len(indices))):
                idx = indices[i]
                results.append({
                    "text": self.chunks[idx],
                    "score": float(scores[i]),
                    "metadata": self.metadata[idx],
                })


        return results


if __name__ == "__main__":
    retriever = Retriever()
    query = "ثبت‌نام دانشجوی مهمان"
    top_chunks = retriever.retrieve(query)
    for c in top_chunks:
        print(f"Score: {c['score']:.3f}, Text: {c['text'][:100]}...")

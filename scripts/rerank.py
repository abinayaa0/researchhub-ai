"""
Rerank hybrid search results using a cross-encoder (bge-reranker-v2-m3).
Takes the top-N candidates from hybrid search and re-scores them
by reading query + chunk together, for more precise ranking.
"""

import os
os.environ["USE_TF"] = "0"

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion

COLLECTION_NAME = "paper_chunks"
QDRANT_URL = "http://localhost:6333"
RETRIEVE_K = 20   # candidates to pull from hybrid search before reranking
RERANK_TOP_K = 5  # final results to keep after reranking


def embed_query(model, query: str):
    output = model.encode([query], return_dense=True, return_sparse=True)
    dense_vec = output["dense_vecs"][0].tolist()
    sparse = output["lexical_weights"][0]
    sparse_indices = [int(k) for k in sparse.keys()]
    sparse_values = [float(v) for v in sparse.values()]
    return dense_vec, sparse_indices, sparse_values


def hybrid_search(client, model, query: str, top_k: int = RETRIEVE_K):
    dense_vec, sparse_indices, sparse_values = embed_query(model, query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=20),
            Prefetch(query={"indices": sparse_indices, "values": sparse_values}, using="sparse", limit=20),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return results.points


def rerank(reranker, query: str, candidates: list, top_k: int = RERANK_TOP_K):
    pairs = [[query, c.payload.get("text", "")] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def print_results(query, before, after):
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print('='*70)

    print("\n--- BEFORE reranking (hybrid search order) ---")
    for i, point in enumerate(before[:RERANK_TOP_K]):
        payload = point.payload
        print(f"  [{i+1}] {payload.get('section_title')} (score={point.score:.4f})")

    print("\n--- AFTER reranking ---")
    for i, (point, score) in enumerate(after):
        payload = point.payload
        print(f"  [{i+1}] {payload.get('section_title')} (rerank_score={score:.4f})")


if __name__ == "__main__":
    print("Loading BGE-M3 embedding model...")
    embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    print("Loading bge-reranker-v2-m3...")
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    test_queries = [
        "What is scaled dot-product attention?",
        "How many attention heads does the model use?",
        "What BLEU score did the Transformer achieve?",
    ]

    for query in test_queries:
        candidates = hybrid_search(client, embed_model, query)
        reranked = rerank(reranker, query, candidates)
        print_results(query, candidates, reranked)
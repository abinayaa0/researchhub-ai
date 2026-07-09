"""
Hybrid search over embedded paper chunks: dense + sparse retrieval,
fused via Qdrant's native RRF (Reciprocal Rank Fusion).
"""

import os
os.environ["USE_TF"] = "0"

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion

COLLECTION_NAME = "paper_chunks"
QDRANT_URL = "http://localhost:6333"
TOP_K = 5


def embed_query(model, query: str):
    output = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
    )
    dense_vec = output["dense_vecs"][0].tolist()
    sparse = output["lexical_weights"][0]
    sparse_indices = [int(k) for k in sparse.keys()]
    sparse_values = [float(v) for v in sparse.values()]
    return dense_vec, sparse_indices, sparse_values


def hybrid_search(client: QdrantClient, model, query: str, top_k: int = TOP_K):
    dense_vec, sparse_indices, sparse_values = embed_query(model, query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=20,  # retrieve more candidates before fusion
            ),
            Prefetch(
                query={"indices": sparse_indices, "values": sparse_values},
                using="sparse",
                limit=20,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return results.points


def print_results(results):
    print(f"\nTop {len(results)} results:\n")
    for i, point in enumerate(results):
        payload = point.payload
        print(f"[{i+1}] Score: {point.score:.4f}")
        print(f"    Section: {payload.get('section_title')}")
        print(f"    Page: {payload.get('page_number')}")
        print(f"    Type: {payload.get('chunk_type')}")
        text_preview = payload.get("text", "")[:200]
        print(f"    Text: {text_preview}...")
        print()


if __name__ == "__main__":
    print("Loading BGE-M3 model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    # Test queries — try a few different kinds
    test_queries = [
        "What is scaled dot-product attention?",
        "How many attention heads does the model use?",
        "What BLEU score did the Transformer achieve?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        results = hybrid_search(client, model, query)
        print_results(results)
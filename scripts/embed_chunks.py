"""
Embed chunks (from chunk_document.py output) using BGE-M3,
store in Qdrant with full metadata as payload.
"""

import json
import os
import uuid
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams

CHUNKS_PATH = "parser_outputs/chunks.json"
COLLECTION_NAME = "paper_chunks"
QDRANT_URL = "http://localhost:6333"  # adjust if running inside Docker network


def check_model_cache():
    """Sanity check whether BGE-M3 is already cached locally (from a previous project)."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    expected_folder = "models--BAAI--bge-m3"
    if os.path.isdir(cache_dir):
        found = any(expected_folder in name for name in os.listdir(cache_dir))
        if found:
            print("BGE-M3 found in local cache — should load without re-downloading.")
        else:
            print("BGE-M3 not found in cache — this will download ~2GB now.")
    else:
        print("No Hugging Face cache directory found — this will download ~2GB now.")


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_and_store(chunks: list[dict]):
    print("Loading BGE-M3 model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    output = model.encode(
        texts,
        batch_size=12,
        max_length=1024,
        return_dense=True,
        return_sparse=True,
    )
    dense_vecs = output["dense_vecs"]
    sparse_vecs = output["lexical_weights"]  # sparse representation

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        print(f"Created collection: {COLLECTION_NAME}")

    points = []
    for i, chunk in enumerate(chunks):
        sparse = sparse_vecs[i]
        sparse_indices = [int(k) for k in sparse.keys()]
        sparse_values = [float(v) for v in sparse.values()]

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vecs[i].tolist(),
                    "sparse": {"indices": sparse_indices, "values": sparse_values},
                },
                payload=chunk,  # keeps paper_id, section_title, page_number, chunk_type, text
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Stored {len(points)} chunks in Qdrant collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    check_model_cache()
    chunks = load_chunks(CHUNKS_PATH)
    embed_and_store(chunks)


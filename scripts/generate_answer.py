"""
Full RAG pipeline: hybrid search -> rerank -> LLM answer with citations.
"""

import os
os.environ["USE_TF"] = "0"

from dotenv import load_dotenv
load_dotenv()
from groq import Groq


from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion

COLLECTION_NAME = "paper_chunks"
QDRANT_URL = "http://localhost:6333"
RETRIEVE_K = 20
RERANK_TOP_K = 5

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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


def build_prompt(query: str, ranked_chunks: list) -> str:
    """Build a prompt with numbered source chunks for citation-grounded generation."""
    context_blocks = []
    for i, (point, score) in enumerate(ranked_chunks):
        payload = point.payload
        source_label = f"[{i+1}]"
        section = payload.get("section_title", "Unknown section")
        page = payload.get("page_number", "?")
        text = payload.get("text", "")
        context_blocks.append(
            f"{source_label} (Section: \"{section}\", Page {page})\n{text}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""You are answering questions using ONLY the provided source excerpts from a research paper.

Rules:
- Answer using ONLY information in the sources below. Do not use outside knowledge.
- Cite sources using their number in brackets, e.g. [1], [2], immediately after the claim they support.
- If the sources don't contain enough information to answer, say so explicitly rather than guessing.
- Be concise and precise.

SOURCES:
{context}

QUESTION: {query}

ANSWER (with citations):"""

    return prompt


def generate_answer(query: str, ranked_chunks: list) -> str:
    prompt = build_prompt(query, ranked_chunks)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def print_citation_key(ranked_chunks: list):
    print("\nSources:")
    for i, (point, score) in enumerate(ranked_chunks):
        payload = point.payload
        print(f"  [{i+1}] {payload.get('section_title')} — Page {payload.get('page_number')}")


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
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print('='*70)

        candidates = hybrid_search(client, embed_model, query)
        ranked_chunks = rerank(reranker, query, candidates)

        print("\nGenerating answer...")
        answer = generate_answer(query, ranked_chunks)

        print(f"\nANSWER:\n{answer}")
        print_citation_key(ranked_chunks)
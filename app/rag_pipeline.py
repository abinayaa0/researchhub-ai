import os
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, Filter, FieldCondition, MatchValue
from groq import Groq

COLLECTION_NAME = "paper_chunks"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
RETRIEVE_K = 20
RERANK_TOP_K = 5

# Loaded once, at module import time (i.e., once when the app starts)
print("Loading BGE-M3 embedding model...")
embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

print("Loading bge-reranker-v2-m3...")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

qdrant_client = QdrantClient(url=QDRANT_URL)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def embed_query(query: str):
    output = embed_model.encode([query], return_dense=True, return_sparse=True)
    dense_vec = output["dense_vecs"][0].tolist()
    sparse = output["lexical_weights"][0]
    sparse_indices = [int(k) for k in sparse.keys()]
    sparse_values = [float(v) for v in sparse.values()]
    return dense_vec, sparse_indices, sparse_values


def hybrid_search(query: str, paper_ids: list[str], top_k: int = RETRIEVE_K):
    dense_vec, sparse_indices, sparse_values = embed_query(query)

    paper_filter = Filter(
        should=[FieldCondition(key="paper_id", match=MatchValue(value=pid)) for pid in paper_ids]
    ) if paper_ids else None

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=20, filter=paper_filter),
            Prefetch(query={"indices": sparse_indices, "values": sparse_values}, using="sparse", limit=20, filter=paper_filter),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return results.points


def rerank_chunks(query: str, candidates: list, top_k: int = RERANK_TOP_K):
    if not candidates:
        return []
    pairs = [[query, c.payload.get("text", "")] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def build_prompt(query: str, ranked_chunks: list) -> str:
    context_blocks = []
    for i, (point, score) in enumerate(ranked_chunks):
        payload = point.payload
        section = payload.get("section_title", "Unknown section")
        page = payload.get("page_number", "?")
        paper_title = payload.get("paper_title", "Unknown paper")
        text = payload.get("text", "")
        context_blocks.append(
            f"[{i+1}] (Paper: \"{paper_title}\", Section: \"{section}\", Page {page})\n{text}"
        )
    context = "\n\n".join(context_blocks)

    return f"""You are answering questions using ONLY the provided source excerpts from research papers.

Rules:
- Answer using ONLY information in the sources below. Do not use outside knowledge.
- Cite sources using their number in brackets, e.g. [1], [2], immediately after the claim they support.
- If the sources don't contain enough information to answer, say so explicitly rather than guessing.
- Be concise and precise.

SOURCES:
{context}

QUESTION: {query}

ANSWER (with citations):"""


def generate_answer(query: str, ranked_chunks: list) -> str:
    if not ranked_chunks:
        return "I don't have enough information in your uploaded papers to answer this question."
    prompt = build_prompt(query, ranked_chunks)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def answer_query(query: str, paper_ids: list[str]) -> dict:
    """Full pipeline: hybrid search -> rerank -> generate. Returns answer + sources."""
    candidates = hybrid_search(query, paper_ids)
    ranked_chunks = rerank_chunks(query, candidates)
    answer = generate_answer(query, ranked_chunks)

    sources = [
        {
            "citation_number": i + 1,
            "paper_title": point.payload.get("paper_title"),
            "section_title": point.pooayload.get("section_title"),
            "page_number": point.payload.get("page_number"),
        }
        for i, (point, score) in enumerate(ranked_chunks)
    ]

    return {"answer": answer, "sources": sources}
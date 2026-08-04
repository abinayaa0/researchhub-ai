import os
import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from groq import Groq

COLLECTION_NAME = "paper_chunks"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
RETRIEVE_K = 20
RERANK_TOP_K = 5

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
qdrant_client = QdrantClient(url=QDRANT_URL)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    response = cohere_client.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type=input_type,
    )
    return response.embeddings


def dense_search(query: str, paper_ids: list[str], top_k: int = RETRIEVE_K):
    query_vec = embed_texts([query], input_type="search_query")[0]

    paper_filter = Filter(
        should=[FieldCondition(key="paper_id", match=MatchValue(value=pid)) for pid in paper_ids]
    ) if paper_ids else None

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="dense",
        query_filter=paper_filter,
        limit=top_k,
        with_payload=True,
    )
    return results.points


def rerank_chunks(query: str, candidates: list, top_k: int = RERANK_TOP_K):
    if not candidates:
        return []
    docs = [c.payload.get("text", "") for c in candidates]
    response = cohere_client.rerank(
        query=query,
        documents=docs,
        top_n=min(top_k, len(docs)),
        model="rerank-english-v3.0",
    )
    return [(candidates[r.index], r.relevance_score) for r in response.results]


def build_prompt(query: str, ranked_chunks: list) -> str:
    context_blocks = []
    for i, (point, score) in enumerate(ranked_chunks):
        payload = point.payload
        context_blocks.append(
            f"[{i+1}] (Paper: \"{payload.get('paper_title')}\", Section: \"{payload.get('section_title')}\", Page {payload.get('page_number')})\n{payload.get('text', '')}"
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
    candidates = dense_search(query, paper_ids)
    ranked_chunks = rerank_chunks(query, candidates)
    answer = generate_answer(query, ranked_chunks)

    sources = [
        {
            "citation_number": i + 1,
            "paper_title": point.payload.get("paper_title"),
            "section_title": point.payload.get("section_title"),
            "page_number": point.payload.get("page_number"),
        }
        for i, (point, score) in enumerate(ranked_chunks)
    ]
    return {"answer": answer, "sources": sources}
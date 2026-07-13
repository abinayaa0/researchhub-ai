"""
Run the full RAG pipeline against the eval dataset and score it with RAGAS.
Computes: context precision, context recall, faithfulness, answer relevancy.
"""
from ragas.run_config import RunConfig
import os
os.environ["USE_TF"] = "0"

import json
import sys

# Allow importing from scripts/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from dotenv import load_dotenv
load_dotenv()

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from groq import Groq

from generate_answer import (
    hybrid_search,
    rerank,
    build_prompt,
    embed_query,
)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

EVAL_DATASET_PATH = "eval/eval_dataset.json"
QDRANT_URL = "http://localhost:6333"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(query: str, ranked_chunks: list) -> str:
    prompt = build_prompt(query, ranked_chunks)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def run_pipeline_on_eval_set(embed_model, reranker, client, eval_items: list) -> dict:
    """Run every eval question through the real pipeline, collect results for RAGAS."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in eval_items:
        query = item["question"]
        print(f"Running: {item['id']} — {query}")

        candidates = hybrid_search(client, embed_model, query)
        ranked_chunks = rerank(reranker, query, candidates)

        context_texts = [c.payload.get("text", "") for c, score in ranked_chunks]
        answer = generate_answer(query, ranked_chunks)

        questions.append(query)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(item["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


if __name__ == "__main__":
    print("Loading eval dataset...")
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_items = json.load(f)
    print(f"Loaded {len(eval_items)} eval questions")

    print("Loading BGE-M3 embedding model...")
    embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    print("Loading bge-reranker-v2-m3...")
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    print("\nRunning pipeline on all eval questions...\n")
    results = run_pipeline_on_eval_set(embed_model, reranker, client, eval_items)

    dataset = Dataset.from_dict(results)

    print("\nSetting up RAGAS evaluator (using Groq as judge LLM)...")
    judge_llm = LangchainLLMWrapper(
        ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    )
    judge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    run_config = RunConfig(
    timeout=180,
    max_retries=10,
    max_wait=90,
    max_workers=1,
)

    print("Running RAGAS evaluation (this may take a few minutes)...\n")
    eval_result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    print(eval_result)

    df = eval_result.to_pandas()
    df.to_csv("eval/ragas_results.csv", index=False)
    print("\nDetailed per-question results saved to eval/ragas_results.csv")
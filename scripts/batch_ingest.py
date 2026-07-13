"""
Batch ingest multiple papers: parse (Unstructured) -> chunk -> embed -> store in Qdrant.
Reuses logic from compare_parsers.py, chunk_document.py, embed_chunks.py.
"""

import os
os.environ["USE_TF"] = "0"

import json
import uuid
import ast
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams

CORPUS_DIR = "eval/corpus_papers"
COLLECTION_NAME = "paper_chunks"
QDRANT_URL = "http://localhost:6333"
MAX_CHUNK_CHARS = 500 * 4
SKIP_SECTION_TITLES = {"references", "acknowledgements", "acknowledgments"}

# Map filenames to clean paper titles + IDs (edit if your filenames differ)
PAPER_REGISTRY = {
    "attention_is_all_you_need.pdf": ("Attention Is All You Need", "paper-attention"),
    "bert.pdf": ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "paper-bert"),
    "roberta.pdf": ("RoBERTa: A Robustly Optimized BERT Pretraining Approach", "paper-roberta"),
    "gpt2.pdf": ("Language Models are Unsupervised Multitask Learners", "paper-gpt2"),
    "vit.pdf": ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "paper-vit"),
    "resnet.pdf": ("Deep Residual Learning for Image Recognition", "paper-resnet"),
}


def parse_with_unstructured(pdf_path: str, api_key: str) -> list:
    client = UnstructuredClient(api_key_auth=api_key)
    with open(pdf_path, "rb") as f:
        req = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(content=f.read(), file_name=os.path.basename(pdf_path)),
                strategy=shared.Strategy.HI_RES,
            )
        )
    res = client.general.partition(request=req)
    return [json.loads(json.dumps(e, default=str)) if not isinstance(e, dict) else e for e in
            [ast.literal_eval(str(el)) if not isinstance(el, dict) else el for el in res.elements]]


def group_into_sections(elements: list, paper_title: str) -> list:
    sections = []
    current_section = {"title": "Untitled", "page_number": None, "texts": [], "tables": []}
    first_title_seen = False

    for el in elements:
        el_type = el.get("type")
        text = (el.get("text") or "").strip()
        metadata = el.get("metadata", {}) or {}

        if el_type == "Title":
            is_paper_title = (not first_title_seen) and (text.strip() == paper_title.strip())
            first_title_seen = True

            if current_section["texts"] or current_section["tables"]:
                sections.append(current_section)

            if is_paper_title:
                current_section = {"title": "__SKIP__", "page_number": None, "texts": [], "tables": []}
                continue

            current_section = {
                "title": text,
                "page_number": metadata.get("page_number"),
                "texts": [],
                "tables": [],
            }

        elif el_type == "Table":
            html = metadata.get("text_as_html")
            current_section["tables"].append({
                "html": html if html else text,
                "raw_text": text,
                "page_number": metadata.get("page_number"),
            })

        elif el_type in ("NarrativeText", "ListItem", "Formula", "FigureCaption"):
            if text:
                current_section["texts"].append(text)

    if current_section["texts"] or current_section["tables"]:
        sections.append(current_section)

    return sections


def is_skippable_section(title: str) -> bool:
    if title == "__SKIP__":
        return True
    return any(skip in title.strip().lower() for skip in SKIP_SECTION_TITLES)


def split_long_text(text: str, max_chars: int) -> list:
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) <= max_chars:
            current += (para + "\n\n")
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= max_chars:
                    current += sent + " "
                else:
                    if current:
                        final_chunks.append(current.strip())
                    current = sent + " "
            if current:
                final_chunks.append(current.strip())
    return final_chunks


def build_chunks(sections: list, paper_title: str, paper_id: str) -> list:
    chunks = []
    for section in sections:
        if is_skippable_section(section["title"]):
            continue

        combined_text = "\n\n".join(section["texts"])
        if combined_text.strip():
            pieces = split_long_text(combined_text, MAX_CHUNK_CHARS)
            for i, piece in enumerate(pieces):
                chunks.append({
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "section_title": section["title"],
                    "page_number": section["page_number"],
                    "chunk_type": "text",
                    "chunk_index": i,
                    "text": piece,
                })

        for j, table in enumerate(section["tables"]):
            chunks.append({
                "paper_id": paper_id,
                "paper_title": paper_title,
                "section_title": section["title"],
                "page_number": table["page_number"] or section["page_number"],
                "chunk_type": "table",
                "chunk_index": j,
                "text": table["html"],
                "raw_text": table["raw_text"],
            })
    return chunks


def embed_and_store(chunks: list, model, client: QdrantClient):
    if not chunks:
        return
    texts = [c["text"] for c in chunks]
    output = model.encode(texts, batch_size=12, max_length=1024, return_dense=True, return_sparse=True)
    dense_vecs = output["dense_vecs"]
    sparse_vecs = output["lexical_weights"]

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

    points = []
    for i, chunk in enumerate(chunks):
        sparse = sparse_vecs[i]
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vecs[i].tolist(),
                    "sparse": {
                        "indices": [int(k) for k in sparse.keys()],
                        "values": [float(v) for v in sparse.values()],
                    },
                },
                payload=chunk,
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)


if __name__ == "__main__":
    unstructured_key = os.getenv("UNSTRUCTURED_API_KEY")

    print("Loading BGE-M3 embedding model...")
    embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    print("Connecting to Qdrant...")
    qdrant_client = QdrantClient(url=QDRANT_URL)

    pdf_files = list(Path(CORPUS_DIR).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {CORPUS_DIR}\n")

    for pdf_path in pdf_files:
        filename = pdf_path.name
        if filename not in PAPER_REGISTRY:
            print(f"Skipping {filename} — not in PAPER_REGISTRY, add it first")
            continue

        paper_title, paper_id = PAPER_REGISTRY[filename]
        print(f"Processing: {paper_title} ({filename})")

        print("  Parsing with Unstructured...")
        elements = parse_with_unstructured(str(pdf_path), unstructured_key)

        print("  Chunking...")
        sections = group_into_sections(elements, paper_title)
        chunks = build_chunks(sections, paper_title, paper_id)
        print(f"  Generated {len(chunks)} chunks from {len(sections)} sections")

        print("  Embedding + storing in Qdrant...")
        embed_and_store(chunks, embed_model, qdrant_client)
        print(f"  Done: {paper_title}\n")

    print("Batch ingestion complete.")
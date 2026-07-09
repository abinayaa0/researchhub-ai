"""
Turn Unstructured API output into structure-aware chunks for embedding.
Groups elements by section (using Title elements as boundaries),
skips References, sub-splits long sections, attaches metadata.
Tables are kept as their own separate chunks using structured HTML.
"""

import json
import os
import re
import ast

MAX_CHUNK_TOKENS = 500
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 4

SKIP_SECTION_TITLES = {"references", "acknowledgements", "acknowledgments"}


def load_unstructured_elements(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [ast.literal_eval(item) for item in raw]


def group_into_sections(elements: list[dict], paper_title: str) -> list[dict]:
    """
    Group NarrativeText/ListItem/Formula/FigureCaption elements under their parent
    Title element. Tables are pulled out separately, not merged into body text.
    Returns: list of {title, page_number, texts: [...], tables: [...]}
    """
    sections = []
    current_section = {"title": "Untitled", "page_number": None, "texts": [], "tables": []}
    first_title_seen = False

    for el in elements:
        el_type = el.get("type")
        text = el.get("text", "").strip()
        metadata = el.get("metadata", {})

        if el_type == "Title":
            # Skip the paper's own title acting as a fake "section"
            is_paper_title = (not first_title_seen) and (text.strip() == paper_title.strip())
            first_title_seen = True

            if current_section["texts"] or current_section["tables"]:
                sections.append(current_section)

            if is_paper_title:
                # Start a placeholder section that will just get skipped
                # (no texts/tables added under it — authors/affiliations dropped)
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

        # Footer, Image, UncategorizedText intentionally skipped

    if current_section["texts"] or current_section["tables"]:
        sections.append(current_section)

    return sections


def is_skippable_section(title: str) -> bool:
    if title == "__SKIP__":
        return True
    normalized = title.strip().lower()
    return any(skip in normalized for skip in SKIP_SECTION_TITLES)


def split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
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


def build_chunks(sections: list[dict], paper_title: str, paper_id: str) -> list[dict]:
    chunks = []
    for section in sections:
        if is_skippable_section(section["title"]):
            continue

        # Text chunks (narrative content), sub-split if too long
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

        # Table chunks — kept separate, using structured HTML
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


if __name__ == "__main__":
    ELEMENTS_PATH = "parser_outputs/unstructured_output.json"
    OUTPUT_PATH = "parser_outputs/chunks.json"
    PAPER_TITLE = "Attention Is All You Need"

    elements = load_unstructured_elements(ELEMENTS_PATH)
    sections = group_into_sections(elements, paper_title=PAPER_TITLE)
    chunks = build_chunks(sections, paper_title=PAPER_TITLE, paper_id="test-paper-001")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Generated {len(chunks)} chunks from {len(sections)} sections")
    print(f"  - text chunks: {sum(1 for c in chunks if c['chunk_type'] == 'text')}")
    print(f"  - table chunks: {sum(1 for c in chunks if c['chunk_type'] == 'table')}")
    print(f"Saved to {OUTPUT_PATH}")

    print("\nSections found:")
    for s in sections:
        skip_marker = " (SKIPPED)" if is_skippable_section(s["title"]) else ""
        print(f"  - {s['title']}{skip_marker}")
"""
Compare PDF parsing APIs on the same document.
Each function returns raw structured output for manual inspection.
"""

import requests
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = r"C:\Users\Abinaya\Downloads\NIPS-2017-attention-is-all-you-need-Paper (1).pdf"
OUTPUT_DIR = "parser_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- 1. Grobid (public demo API) ----------
def parse_with_grobid(pdf_path: str, retries=2, delay=15) -> str:
    for attempt in range(retries):
        with open(pdf_path, "rb") as f:
            response = requests.post(
                "https://kermitt2-grobid.hf.space/api/processFulltextDocument",
                files={"input": f},
                timeout=120,
            )
        if response.status_code == 200 and "Preparing Space" not in response.text:
            return response.text
        print(f"Grobid attempt {attempt+1} failed or space still waking up, retrying in {delay}s...")
        time.sleep(delay)
    raise Exception("Grobid failed after retries")


# ---------- 2. LlamaParse ----------
def parse_with_llamaparse(pdf_path: str, api_key: str) -> str:
    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
    )
    documents = parser.load_data(pdf_path)
    return "\n\n".join(doc.text for doc in documents)


# ---------- 3. Unstructured API ----------
def parse_with_unstructured(pdf_path: str, api_key: str) -> list:
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import operations, shared

    client = UnstructuredClient(api_key_auth=api_key)

    with open(pdf_path, "rb") as f:
        req = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(content=f.read(), file_name=os.path.basename(pdf_path)),
                strategy=shared.Strategy.HI_RES,
            )
        )
    res = client.general.partition(request=req)
    return res.elements


# ---------- Run comparison ----------
if __name__ == "__main__":
    print("Parsing with Grobid...")
    try:
        grobid_output = parse_with_grobid(PDF_PATH)
        with open(f"{OUTPUT_DIR}/grobid_output.xml", "w", encoding="utf-8") as f:
            f.write(grobid_output)
        print(f"Saved: {OUTPUT_DIR}/grobid_output.xml ({len(grobid_output)} chars)")
    except Exception as e:
        print(f"Grobid failed: {e}")

    llamaparse_key = os.getenv("LLAMAPARSE_API_KEY")
    if llamaparse_key:
        print("Parsing with LlamaParse...")
        llamaparse_output = parse_with_llamaparse(PDF_PATH, llamaparse_key)
        with open(f"{OUTPUT_DIR}/llamaparse_output.md", "w", encoding="utf-8") as f:
            f.write(llamaparse_output)
        print(f"Saved: {OUTPUT_DIR}/llamaparse_output.md ({len(llamaparse_output)} chars)")
    else:
        print("Skipping LlamaParse — set LLAMAPARSE_API_KEY env var")

    unstructured_key = os.getenv("UNSTRUCTURED_API_KEY")
    if unstructured_key:
        print("Parsing with Unstructured...")
        elements = parse_with_unstructured(PDF_PATH, unstructured_key)
        with open(f"{OUTPUT_DIR}/unstructured_output.json", "w", encoding="utf-8") as f:
            json.dump([str(e) for e in elements], f, indent=2)
        print(f"Saved: {OUTPUT_DIR}/unstructured_output.json ({len(elements)} elements)")
    else:
        print("Skipping Unstructured — set UNSTRUCTURED_API_KEY env var")

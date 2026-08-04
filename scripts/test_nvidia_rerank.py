import requests
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

response = requests.post(
    "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    },
    json={
        "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
        "query": {"text": "What is scaled dot-product attention?"},
        "passages": [
            {"text": "Scaled dot-product attention divides by sqrt(dk)."},
            {"text": "ResNet uses residual connections."},
        ],
    },
)
print(response.status_code)
print(response.text)
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1",
)

response = client.embeddings.create(
    input=["hello"],
    model="baai/bge-m3",
    
)

print(response.data[0].embedding[:5])  # just print first 5 numbers to confirm it worked
print(f"Embedding length: {len(response.data[0].embedding)}")
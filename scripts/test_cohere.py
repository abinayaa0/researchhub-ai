import os
from dotenv import load_dotenv
load_dotenv()
import cohere

client = cohere.Client(os.getenv("COHERE_API_KEY"))
response = client.embed(texts=["hello world"], model="embed-english-v3.0", input_type="search_query")
print(len(response.embeddings[0]))
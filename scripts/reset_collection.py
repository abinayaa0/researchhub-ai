from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
if client.collection_exists("paper_chunks"):
    client.delete_collection("paper_chunks")
    print("Deleted old paper_chunks collection")
else:
    print("No existing collection found")
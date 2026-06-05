import chromadb

client = chromadb.Client()

collection = client.get_or_create_collection("test")

collection.add(
    documents=["Hello world"],
    embeddings=[[0.1, 0.2, 0.3]],
    ids=["2"]
)

results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.3]],
    n_results=1
)

print(collection.count())
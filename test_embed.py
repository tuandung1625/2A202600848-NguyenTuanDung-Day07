from src import LocalEmbedder

embedder = LocalEmbedder()

print(embedder._backend_name)
print(len(embedder("embedding smoke test")))

from knowledge_base.vector_store import VectorStore

store = VectorStore()

results = store.collection.get(
    limit=459,
    include=["documents", "metadatas"],
)

matches = 0

for i, chunk_id in enumerate(results["ids"]):
    text = results["documents"][i]
    metadata = results["metadatas"][i]

    if "DA-SQL-201" in text:
        matches += 1
        print("\n" + "=" * 60)
        print("Stored ID:", chunk_id)
        print("Metadata:", metadata)
        print("Text:", text[:700])

print(f"\nTotal chunks mentioning DA-SQL-201: {matches}")

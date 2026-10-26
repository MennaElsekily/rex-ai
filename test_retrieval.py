from knowledge_base.retriever import Retriever

retriever = Retriever(top_k=15, candidate_multiplier=8)

query = (
    "For SQL for Analytics & Reporting, list every assessment "
    "component and its percentage weight, including query labs, "
    "practical test, and capstone."
)

results = retriever.retrieve(query)

print(f"\nTotal results: {len(results)}\n")

for i, result in enumerate(results, start=1):
    print(f"--- Result {i} ---")

    for key, value in result.items():
        if key != "text":
            print(f"{key.replace('_', ' ').title()}: {value}")

    print("\nRetrieved Text:")
    print(result.get("text", "No text available"))
    print()
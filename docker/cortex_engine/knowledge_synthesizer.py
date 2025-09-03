from .collection_manager import WorkingCollectionManager


def run_synthesis(collection_name: str, seed_ideas: str, llm_provider: str = "ollama") -> str:
    mgr = WorkingCollectionManager()
    doc_ids = mgr.get_doc_ids_by_name(collection_name)
    return (
        f"# Synthesis for '{collection_name}'\n\n"
        f"Provider: {llm_provider}\n\n"
        f"Seed ideas:\n{seed_ideas}\n\n"
        f"Documents in collection: {len(doc_ids)}\n\n"
        "Draft synthesis content..."
    )


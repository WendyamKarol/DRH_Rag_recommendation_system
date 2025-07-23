import json
from src.models import TrainingDocument
from src.embeddings.faiss_store import FAISSEmbeddingStore
from config.config import config, DOCUMENT_TYPES


def load_docs_for_type(doc_type_key: str) -> list[TrainingDocument]:
    label = DOCUMENT_TYPES[doc_type_key]
    with open(config.data.training_corpus_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [
        TrainingDocument(
            doc_id=d.get("doc_id", f"doc_{i}"),
            type=d["type"],
            source=d.get("source", "source_inconnu"),
            content=d["content"]
        )
        for i, d in enumerate(raw)
        if d["type"].lower() == label.lower()
    ]


def build_index(doc_type_key: str):
    label = doc_type_key
    path = config.data.faiss_index_paths[label]
    docs = load_docs_for_type(label)

    if not docs:
        print(f"Aucun document trouvé pour le type : {label}")
        return

    store = FAISSEmbeddingStore()
    store.add_documents(docs)
    store.save(path)
    print(f"✅ Index généré pour {label} : {path}.index + .docs")


if __name__ == "__main__":
    for key in DOCUMENT_TYPES.keys():
        build_index(key)

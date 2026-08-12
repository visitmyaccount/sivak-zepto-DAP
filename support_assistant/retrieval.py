"""Load, embed, store, and retrieve Zepto policy documents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

MODULE_DIR = Path(__file__).parent
DOCUMENT_DIR = MODULE_DIR / "docs"
DEFAULT_DATABASE_DIR = MODULE_DIR / "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "zepto_policies"


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    distance: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


class PolicyRetriever:
    """Small wrapper around the local embedding model and ChromaDB."""

    def __init__(self, database_dir: Path = DEFAULT_DATABASE_DIR) -> None:
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(
            path=str(database_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.ingest_documents()

    def ingest_documents(self) -> int:
        """Store one short chunk for each of the eight supplied documents."""
        document_paths = sorted(DOCUMENT_DIR.glob("doc_*.txt"))
        if len(document_paths) != 8:
            raise RuntimeError(f"Expected 8 policy documents, found {len(document_paths)}")

        ids = []
        documents = []
        metadata = []
        for path in document_paths:
            document_id = path.stem
            ids.append(f"{document_id}_chunk_00")
            documents.append(path.read_text(encoding="utf-8").strip())
            metadata.append({"document_id": document_id})

        embeddings = self.model.encode(
            documents, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadata,
            embeddings=embeddings,
        )
        return len(ids)

    def retrieve(self, query: str, limit: int = 3) -> list[RetrievedChunk]:
        """Return the closest policy chunks using cosine distance."""
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=metadata["document_id"],
                    text=text,
                    distance=float(distance),
                )
            )
        return chunks


def main() -> None:
    retriever = PolicyRetriever()
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedded chunks: {retriever.collection.count()}")


if __name__ == "__main__":
    main()

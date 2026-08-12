"""Create or refresh the local policy collection."""

from retrieval import PolicyRetriever


def main() -> None:
    retriever = PolicyRetriever()
    print(f"Embedded chunks: {retriever.collection.count()}")


if __name__ == "__main__":
    main()

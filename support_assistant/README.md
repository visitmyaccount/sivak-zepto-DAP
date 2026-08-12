---
title: Zepto Policy Support Assistant
emoji: 🛒
colorFrom: purple
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Support Assistant

This module is a small retrieval-augmented FastAPI service for the eight supplied Zepto policy documents. It uses local embeddings and deterministic mock answers by default, so the graded path does not need an API key or LLM network call.

## Setup and run

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r support_assistant/requirements.txt
python support_assistant/ingest.py
cd support_assistant
MOCK_LLM=1 uvicorn main:app --host 0.0.0.0 --port 7860
```

The first ingestion downloads `all-MiniLM-L6-v2`. Later runs can reuse the local model cache. The Chroma database is recreated locally and is intentionally not committed because its internal storage can change between library versions.

## Example mock requests

Policy query:

```bash
curl -sS http://localhost:7860/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the delivery fee for a small order?"}'
```

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del...",
  "sources": ["doc_01_chunk_00", "doc_05_chunk_00", "doc_02_chunk_00"],
  "confidence": 1.0
}
```

General query:

```bash
curl -sS http://localhost:7860/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who won the football match?"}'
```

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

These responses were generated with mock mode enabled. The saved raw examples are in `outputs/mock_examples.json`.

## Architecture

```text
docs/*.txt
    |
    v
retrieval.py: load one chunk per document
    |
    v
all-MiniLM-L6-v2: local normalized embeddings
    |
    v
ChromaDB collection: zepto_policies
    |
    v
classify_intent node
    | policy_question                 | general_question
    v                                 v
retrieve_and_answer node          direct_answer node
    |                                 |
    +---------------+-----------------+
                    v
        Pydantic AskResponse JSON
```

### Ingestion

`retrieval.py` reads `docs/doc_01.txt` through `docs/doc_08.txt`. Their short length allows one chunk per document. Each chunk receives a stable ID such as `doc_01_chunk_00`.

### Embedding

`PolicyRetriever` uses the open-source `all-MiniLM-L6-v2` sentence-transformer locally and normalizes its vectors. It upserts all eight vectors into the `zepto_policies` ChromaDB collection configured for cosine distance.

### Retrieval

The LangGraph `classify_intent` node sends a policy question to `retrieve_and_answer`. That node embeds the query and retrieves the top three chunks. Retrieval runs for real in both mock and optional real-provider modes.

### Generation

With `MOCK_LLM` unset or set to `1`, policy answers use the first 200 characters of the closest real retrieved chunk, while general answers use a fixed string. No provider client is created in this mode. Both routes produce an `AskResponse` containing `answer`, `sources`, and `confidence`.

When `MOCK_LLM=0`, classification and answer generation call Groq. `prompt_template.py` contains the complete role-context-task-format-length prompt, its grounding constraint, and a few-shot example. Provider JSON is validated with Pydantic and retried up to two additional times after an invalid response.

## Optional real-provider mode

Supply credentials only through the shell. Never add them to a repository file.

```bash
export GROQ_API_KEY='<your-key-in-the-shell>'
export MOCK_LLM=0
export GROQ_MODEL='openai/gpt-oss-20b'
uvicorn main:app --host 0.0.0.0 --port 7860
```

The model can be changed through `GROQ_MODEL` without editing source code.

## Container

From this directory:

```bash
podman build -t zepto-support-assistant .
podman run --rm -p 7860:7860 zepto-support-assistant
```

The image runs as a non-root user and defaults to `MOCK_LLM=1`. The image was built successfully and both `/ask` routes returned HTTP 200 from the running container. On a corporate network that replaces public TLS certificates, mount a trusted local model cache read-only rather than disabling TLS verification.

## Hugging Face Space

This README includes the required Docker Space metadata and port 7860. The public deployment uses the free CPU tier and mock mode so it does not expose or consume an API credential.

Deployment URL: pending authenticated deployment.

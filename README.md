# RAG Chat Backend (FastAPI + Chroma)

A small AI chat backend. You `POST` a question to `/chat`; the service retrieves the most
relevant passages from a vector database (Chroma), sends them to an LLM as context, and
**streams the answer back token-by-token using Server-Sent Events (SSE)**.

## Features

- `POST /chat` - accepts a user message (plus optional history) and streams the reply (SSE)
- Simple RAG pipeline: load docs -> chunk -> embed -> store in Chroma -> retrieve top-k -> prompt LLM
- Sample documents in `data/documents/` are indexed automatically on first start
- Works with  Groq LLM  and has an **offline mock LLM**
- Input validation, structured error handling, request logging
- Modular code and automated tests (`pytest`)

## Project structure

```
app/
  main.py               App factory, startup (ingest + wiring), logging middleware, error handler
  api.py                Routes: POST /chat, GET /health
  config.py             Settings from environment / .env
  schemas.py            Request validation models
  sse.py                SSE frame formatter
  logging_config.py     Logging setup
  rag/
    chunking.py         Sentence-aware chunker with overlap
    embeddings.py       Embedding backend selection (Chroma MiniLM or offline hash)
    vector_store.py     Chroma wrapper (upsert / query)
    ingest.py           Reads data/documents and indexes it (also a CLI)
  llm/
    base.py             LLM interface + LLMError
    openai_provider.py  Streaming client for OpenAI-compatible APIs
    mock_provider.py    Offline fallback
    factory.py          Chooses the provider from settings
  services/
    prompts.py          Prompt construction
    chat_service.py     Retrieve -> build prompt -> stream answer
data/documents/         Sample knowledge base (fictional "Nimbus Labs" docs)
tests/test_app.py       API, streaming, error-handling and chunking tests
```

## Setup and run

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1         

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn app.main:app --reload
```

The API is now at <http://localhost:8000> (interactive docs: <http://localhost:8000/docs>).

**Notes**

- On the very first start, Chroma downloads its small embedding model (~80 MB, one time) and indexes the
  sample documents. Later starts reuse the saved index in `chroma_db/`.
- **No API key?** The app still runs: it falls back to a mock LLM that streams the best matching passage,
  so you can demo the full retrieval + streaming flow offline.
- **Other LLM providers:** set `LLM_BASE_URL` (e.g. `https://api.groq.com/openai/v1` or, for a local model
  with Ollama, `http://localhost:11434/v1`) and `LLM_MODEL` (e.g. `llama3.1`).
- **Add your own documents:** drop `.txt` / `.md` files into `data/documents/` and run
  `python -m app.rag.ingest` to re-index.
- **Run the tests** (offline, no API key or downloads needed): `pytest -q`

## API

### `POST /chat`

Request body:

| Field     | Type   | Required | Notes                                                        |
|-----------|--------|----------|--------------------------------------------------------------|
| `message` | string | yes      | 1-2000 characters                                            |
| `history` | array  | no       | Previous turns `[{"role": "user"\|"assistant", "content": "..."}]` |

Response: `text/event-stream` with these events:

| Event     | Data                                   | Meaning                                  |
|-----------|----------------------------------------|------------------------------------------|
| `sources` | `{"sources": ["file.md", ...]}`        | Documents used as context (sent first)   |
| `token`   | `{"text": "..."}`                      | A piece of the answer                    |
| `done`    | `{}`                                   | Stream finished successfully             |
| `error`   | `{"message": "..."}`                   | Something failed after streaming began   |

### Example request

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many days of annual leave do employees get?"}'
```

(`-N` disables curl's buffering so you see the tokens as they arrive.)

### Example response

```
event: sources
data: {"sources": ["employee_handbook.md", "security_policy.txt"]}

event: token
data: {"text": "Employees"}

event: token
data: {"text": " get"}

event: token
data: {"text": " 24"}

event: token
data: {"text": " days"}

event: token
data: {"text": " [employee_handbook.md]."}

event: done
data: {}
```

Joined together, the tokens read: *"Employees get 24 days [employee_handbook.md]."* With a real LLM the
wording differs and there are more (smaller) `token` events.

### Python client example

```python
import json, httpx

with httpx.stream("POST", "http://localhost:8000/chat",
                  json={"message": "What are the Pro plan limits?"}, timeout=60) as r:
    event = None
    for line in r.iter_lines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event == "token":
            print(json.loads(line[6:])["text"], end="", flush=True)
```

### Error responses

| Situation                              | Result                                                        |
|----------------------------------------|---------------------------------------------------------------|
| Empty / too long message, bad JSON     | `422` with validation details                                 |
| Vector DB fails before streaming       | `503 {"detail": "Knowledge base is unavailable."}`            |
| LLM fails during streaming             | `200` stream ends with `event: error` (generic message)       |
| Any other unexpected exception         | `500 {"detail": "Internal server error."}`                    |

### `GET /health`

Returns `{"status": "ok", "indexed_chunks": 6}`.

## Architecture

```
                    ┌────────────────────────── startup ──────────────────────────┐
                    │ data/documents/*.md,txt -> chunk_text -> Chroma (embeddings) │
                    └──────────────────────────────────────────────────────────────┘

 Client ── POST /chat ──► api.py ──► ChatService.retrieve() ──► Chroma top-k chunks
    ▲                        │                                          │
    │                        ▼                                          ▼
    └──── SSE stream ◄── ChatService.stream_answer() ◄── build_messages(question, chunks, history)
      sources/token/done            │
                                    ▼
                         BaseLLM.stream()  (OpenAI-compatible API  |  mock)
```

1. **Ingestion (startup)** - each document is split into ~500-character, sentence-aware chunks (with a small
   overlap) and stored in a persistent Chroma collection. Embeddings are computed by Chroma.
2. **Retrieval** - the user's question is embedded and the `TOP_K` (default 3) closest chunks are fetched.
3. **Generation** - the chunks are placed into the system prompt, which tells the model to answer *only*
   from that context and to say "I don't know" otherwise. The LLM's output is streamed to the client.

## Important technical decisions

- **SSE over WebSockets** - the traffic is one-way (server -> client) and SSE is plain HTTP, works with
  `curl`/browsers (`EventSource`/`fetch`), and needs no extra dependency. Frames are hand-formatted
  (`app/sse.py`), and token text is JSON-encoded so newlines inside tokens can't break the format.
- **Retrieval runs *before* the stream starts** - a vector-DB failure can then return a proper HTTP `503`.
  Once streaming has begun the status code is already sent, so later LLM failures are reported as an
  `error` SSE event instead. Provider error details are logged, not sent to the client.
- **Chroma called via `asyncio.to_thread`** - Chroma and the embedding model are synchronous/CPU-bound;
  running them in a thread keeps the event loop free for other requests.
- **Chroma directly, without LangChain** - the pipeline is ~100 lines, so calling Chroma directly keeps
  it easy to read and avoids a large dependency tree. The `VectorStore` / `BaseLLM` boundaries make it
  simple to swap in LangChain components later if needed.
- **OpenAI-compatible LLM interface** - one client covers OpenAI, Groq, Ollama and many others by changing
  `LLM_BASE_URL`. A mock provider keeps the project runnable (and testable) with no key or internet.
- **Cosine similarity + Chroma's local MiniLM embeddings** - no embedding API key is required.
  The `hash` backend exists purely so tests run offline.
- **Idempotent ingestion** - chunk IDs are deterministic (`file::index`) and a file's old chunks are
  replaced on re-ingest, so re-running never creates duplicates. Startup only ingests when the DB is empty.
- **Settings via environment (`pydantic-settings`)** - all tunables (model, chunk size, top-k, paths) live in
  `config.py`, with no hard-coded secrets.

## Limitations / possible improvements

- Only `.txt` / `.md` files are ingested (add PDF loaders for more formats).
- No relevance threshold or re-ranking; the prompt relies on the model saying "I don't know".
- Conversation history is supplied by the client (the server is stateless) and only the last 10 messages are used.
- No authentication or rate limiting - add both before exposing this publicly.

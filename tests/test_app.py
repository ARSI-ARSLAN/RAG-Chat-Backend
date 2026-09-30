import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.base import BaseLLM, LLMError
from app.main import create_app
from app.rag.chunking import chunk_text
from app.services.chat_service import ChatService


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        chroma_dir=tmp_path / "chroma",
        embedding_backend="hash",  # offline, no model download
        llm_provider="mock",
        llm_api_key=None,
        llm_base_url=None,
    )
    with TestClient(create_app(settings)) as c:
        yield c


def parse_sse(response) -> list[tuple[str, dict]]:
    events, name = [], None
    for line in response.iter_lines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            events.append((name, json.loads(line[6:])))
    return events


def test_health_reports_indexed_chunks(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["indexed_chunks"] > 0


def test_chat_streams_sources_tokens_and_done(client):
    with client.stream("POST", "/chat", json={"message": "How many days of annual leave do I get?"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r)
    names = [n for n, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert names.count("token") > 1
    assert "employee_handbook.md" in events[0][1]["sources"]


def test_empty_message_is_rejected(client):
    assert client.post("/chat", json={"message": "   "}).status_code == 422


def test_llm_failure_is_reported_as_sse_error_event(client):
    class FailingLLM(BaseLLM):
        async def stream(self, messages):
            raise LLMError("boom")
            yield  # pragma: no cover

    state = client.app.state
    state.chat_service = ChatService(state.store, FailingLLM(), top_k=3, max_history=10)
    with client.stream("POST", "/chat", json={"message": "hello"}) as r:
        assert r.status_code == 200
        events = parse_sse(r)
    assert events[-1][0] == "error"
    assert "boom" not in json.dumps(events)  # internal details are not leaked


def test_retrieval_failure_returns_503(client):
    def broken(*args, **kwargs):
        raise RuntimeError("db down")

    client.app.state.store.query = broken
    assert client.post("/chat", json={"message": "hello"}).status_code == 503


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)
    assert chunks[0].split(". ")[-1] in chunks[1]  # overlap carried forward

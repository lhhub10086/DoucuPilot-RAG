from fastapi.testclient import TestClient

from main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_documents() -> None:
    client = TestClient(app)
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "count" in data


def test_ask_empty_question_returns_400() -> None:
    client = TestClient(app)
    response = client.post(
        "/ask",
        json={"question": " ", "strategy": "agentic", "chunk_strategy": "section", "top_k": 5},
    )
    assert response.status_code == 400


def test_ask_invalid_strategy_returns_400() -> None:
    client = TestClient(app)
    response = client.post(
        "/ask",
        json={"question": "hello", "strategy": "bad", "chunk_strategy": "section", "top_k": 5},
    )
    assert response.status_code == 400


def test_ask_valid_request(monkeypatch) -> None:
    def fake_run_ask(question: str, strategy: str, top_k: int, chunk_strategy: str) -> dict:
        return {
            "answer": "mock answer",
            "citations": [],
            "strategy": strategy,
            "latency_ms": 12.3,
            "retrieved_chunk_ids": ["c1"],
            "reranked_chunk_ids": [],
            "rewritten_question": None,
            "route_trace": ["classify_question: fact"] if strategy == "agentic" else [],
            "failure_reason": None,
        }

    import app.api.routes_ask as routes_ask

    monkeypatch.setattr(routes_ask, "run_ask", fake_run_ask)
    client = TestClient(app)
    response = client.post(
        "/ask",
        json={
            "question": "这篇文档的核心方法是什么？",
            "strategy": "agentic",
            "chunk_strategy": "section",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "mock answer"
    assert data["citations"] == []
    assert data["strategy"] == "agentic"
    assert isinstance(data["latency_ms"], float)
    assert data["route_trace"]


def test_swagger_docs_open() -> None:
    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()

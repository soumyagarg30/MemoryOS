from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


def payload(**overrides):
    return {
        "user_id": str(uuid4()),
        "content": "A useful memory",
        "memory_type": "SEMANTIC",
        **overrides,
    }


def test_create_read_delete(memory_client: TestClient) -> None:
    data = payload(embedding=[0.1, -0.2, 0.3], metadata={"tags": ["test"], "nested": {"n": 2}})
    response = memory_client.post("/memories", json=data)
    assert response.status_code == 201, response.text
    memory = response.json()
    assert response.headers["location"] == f"/memories/{memory['id']}"
    assert memory["embedding"] == pytest.approx(data["embedding"])
    assert memory["metadata"] == data["metadata"]
    assert "metadata_" not in memory
    assert memory["status"] == "ACTIVE"
    assert memory["importance"] == 0.5
    assert memory["confidence"] == 1.0
    assert memory["access_count"] == 0
    assert memory["last_accessed_at"] is None
    assert memory["created_at"]
    assert memory["updated_at"]
    assert memory["expires_at"] is None
    assert memory["source"] is None
    assert memory["superseded_by"] is None
    retrieved = memory_client.get(response.headers["location"])
    assert retrieved.status_code == 200
    assert retrieved.json() == memory
    deleted = memory_client.delete(response.headers["location"])
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert memory_client.get(response.headers["location"]).status_code == 404
    assert memory_client.delete(response.headers["location"]).status_code == 404


@pytest.mark.parametrize("memory_type", ["WORKING", "EPISODIC", "SEMANTIC", "PREFERENCE", "TASK"])
def test_memory_types(memory_client: TestClient, memory_type: str) -> None:
    response = memory_client.post("/memories", json=payload(memory_type=memory_type))
    assert response.status_code == 201
    assert response.json()["memory_type"] == memory_type
    assert response.json()["embedding"] == [1.0, 0.0]
    assert response.json()["metadata"] == {}


@pytest.mark.parametrize("status", ["ACTIVE", "STALE", "ARCHIVED", "SUPERSEDED"])
def test_memory_states(memory_client: TestClient, status: str) -> None:
    response = memory_client.post("/memories", json=payload(status=status))
    assert response.status_code == 201
    assert response.json()["status"] == status


def test_optional_fields(memory_client: TestClient) -> None:
    response = memory_client.post(
        "/memories",
        json=payload(
            content="  Remember this  ",
            importance=0,
            confidence=1,
            source="manual",
            expires_at="2030-01-01T00:00:00Z",
        ),
    )
    assert response.status_code == 201
    data = response.json()
    assert data["content"] == "Remember this"
    assert data["importance"] == 0
    assert data["source"] == "manual"
    assert data["expires_at"].startswith("2030-01-01T00:00:00")


@pytest.mark.parametrize(
    "invalid",
    [
        {"user_id": "bad"},
        {"content": " \n "},
        {"memory_type": "OTHER"},
        {"status": "OTHER"},
        {"importance": -0.1},
        {"importance": 1.1},
        {"confidence": -1},
        {"confidence": 2},
        {"confidence": "NaN"},
        {"embedding": []},
        {"embedding": ["Infinity"]},
        {"embedding": [1e39]},
        {"embedding": [0] * 16001},
        {"embedding": [[1, 2]]},
        {"metadata": []},
        {"metadata": None},
        {"metadata": {"invalid": float("inf")}},
        {"expires_at": "2030-01-01T00:00:00"},
        {"source": "a" * 256},
        {"access_count": 100},
        {"id": str(uuid4())},
    ],
)
def test_validation(memory_client: TestClient, invalid: dict) -> None:
    # Use raw JSON for the nonfinite JSON number case (httpx's encoder rejects it).
    import json

    response = memory_client.post(
        "/memories",
        content=json.dumps(payload(**invalid)),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422, response.text
    assert memory_client.get("/memories").json() == []


def test_filters_and_pagination(memory_client: TestClient) -> None:
    user_id = str(uuid4())
    ids = []
    for index in range(3):
        response = memory_client.post(
            "/memories",
            json=payload(
                user_id=user_id,
                content=f"Memory {index}",
                memory_type="TASK" if index == 0 else "SEMANTIC",
                status="ARCHIVED" if index == 0 else "ACTIVE",
            ),
        )
        assert response.status_code == 201
        ids.append(response.json()["id"])
    memory_client.post("/memories", json=payload())
    all_items = memory_client.get("/memories", params={"user_id": user_id}).json()
    assert {item["id"] for item in all_items} == set(ids)
    page = memory_client.get("/memories", params={"user_id": user_id, "limit": 1, "offset": 1})
    assert page.json() == all_items[1:2]
    filtered = memory_client.get(
        "/memories",
        params={
            "user_id": user_id,
            "memory_type": "TASK",
            "status": "ARCHIVED",
        },
    )
    assert [item["id"] for item in filtered.json()] == [ids[0]]
    assert memory_client.get("/memories", params={"user_id": str(uuid4())}).json() == []


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1", "user_id=x", "status=x"])
def test_invalid_query(memory_client: TestClient, query: str) -> None:
    assert memory_client.get(f"/memories?{query}").status_code == 422


def test_invalid_and_missing_ids(memory_client: TestClient) -> None:
    for method in (memory_client.get, memory_client.delete):
        assert method("/memories/not-a-uuid").status_code == 422
        assert method(f"/memories/{uuid4()}").status_code == 404


def test_superseding_reference(memory_client: TestClient) -> None:
    user_id = str(uuid4())
    replacement = memory_client.post("/memories", json=payload(user_id=user_id)).json()
    old = memory_client.post(
        "/memories",
        json=payload(
            user_id=user_id,
            status="SUPERSEDED",
            superseded_by=replacement["id"],
        ),
    )
    assert old.status_code == 201
    assert old.json()["superseded_by"] == replacement["id"]
    assert memory_client.delete(f"/memories/{replacement['id']}").status_code == 409
    assert memory_client.get(f"/memories/{replacement['id']}").status_code == 200
    assert memory_client.delete(f"/memories/{old.json()['id']}").status_code == 204
    assert memory_client.delete(f"/memories/{replacement['id']}").status_code == 204


def test_invalid_superseding_reference(memory_client: TestClient) -> None:
    other = memory_client.post("/memories", json=payload()).json()
    for reference in (str(uuid4()), other["id"]):
        response = memory_client.post("/memories", json=payload(superseded_by=reference))
        assert response.status_code == 422

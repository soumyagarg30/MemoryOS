from fastapi.testclient import TestClient


def test_health_returns_ok_without_database(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"status": "ok"}


def test_health_is_documented(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    response_schema = schema["paths"]["/health"]["get"]["responses"]["200"]
    assert response_schema["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HealthResponse"
    }


def test_health_rejects_post(client: TestClient) -> None:
    assert client.post("/health").status_code == 405

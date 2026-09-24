import json
import math
import os
import sqlite3
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql.elements import BinaryExpression

from memoryos.api.memories import get_storage_service
from memoryos.config import Settings
from memoryos.db.session import get_db
from memoryos.main import create_app
from memoryos.models import Base
from memoryos.schemas.conflicts import MemoryConflictAnalysis
from memoryos.services.conflicts import MemoryConflictDetector
from memoryos.services.embeddings import EmbeddingService
from memoryos.services.storage import MemoryStorageService


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    def forbid_connection(*args: object, **kwargs: object) -> None:
        pytest.fail("Health checks must not connect to the database")

    monkeypatch.setattr("psycopg.connect", forbid_connection)
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://test:test@127.0.0.1:1/test",
        debug=False,
        sql_echo=False,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@compiles(JSONB, "sqlite")
def compile_jsonb_for_unit_tests(type_, compiler, **kw):
    return "JSON"


@compiles(BinaryExpression, "sqlite")
def sqlite_cosine_expression(element, compiler, **kw):
    # SQLite test substitute only; production SQL retains pgvector's <=> operator.
    if getattr(element.operator, "opstring", None) == "<=>":
        left = compiler.process(element.left, **kw)
        right = compiler.process(element.right, **kw)
        return f"cosine_distance({left}, {right})"
    return compiler.visit_binary(element, **kw)


@event.listens_for(Engine, "connect")
def sqlite_vector_functions(connection, _):
    if not isinstance(connection, sqlite3.Connection):
        return
    connection.execute("PRAGMA foreign_keys=ON")
    connection.create_function("vector_dims", 1, lambda v: len(json.loads(v)) if v else None)
    connection.create_function(
        "vector_norm", 1, lambda v: math.hypot(*json.loads(v)) if v else None
    )

    def cosine_distance(left, right):
        a, b = json.loads(left), json.loads(right)
        assert len(a) == len(b), "Dimension filtering must happen before cosine evaluation"
        return 1 - sum(x * y for x, y in zip(a, b, strict=True)) / (math.hypot(*a) * math.hypot(*b))

    connection.create_function("cosine_distance", 2, cosine_distance)


@pytest.fixture
def storage_llm():
    async def unrelated(**kwargs):
        context = json.loads(kwargs["input"][1]["content"])
        return SimpleNamespace(
            status="completed",
            output=[],
            output_parsed=MemoryConflictAnalysis(
                comparisons=[
                    {
                        "memory_id": memory["id"],
                        "relationship": "UNRELATED",
                        "reason": "Test fixture.",
                    }
                    for memory in context["existing_memories"]
                ]
            ),
        )

    client = Mock()
    client.embed = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])
    )
    client.parse = AsyncMock(side_effect=unrelated)
    return client


@pytest.fixture
def storage_service(storage_llm):
    return MemoryStorageService(
        EmbeddingService(storage_llm, "test-embedding"),
        MemoryConflictDetector(storage_llm, "test-conflict-model"),
    )


@pytest.fixture
def postgres_connection():
    url = os.environ.get("MEMORYOS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set MEMORYOS_TEST_DATABASE_URL to run PostgreSQL integration tests")
    engine = create_engine(url)
    schema = f"test_{uuid4().hex}"
    try:
        with engine.connect() as connection, connection.begin():
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
            connection.info["test_schema"] = schema
            try:
                yield connection
            finally:
                # Roll back the schema, migrations, and all test writes together.
                connection.rollback()
    finally:
        engine.dispose()


@pytest.fixture(
    params=["sqlite"] + (["postgres"] if os.environ.get("MEMORYOS_TEST_DATABASE_URL") else [])
)
def memory_client(request, storage_service) -> Iterator[TestClient]:
    engine = None
    if request.param == "postgres":
        bind = request.getfixturevalue("postgres_connection")
        config = Config("alembic.ini")
        config.attributes["connection"] = bind
        config.attributes["version_table_schema"] = bind.info["test_schema"]
        command.upgrade(config, "head")
    else:
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(engine)
        bind = engine

    def override_db():
        with Session(
            bind=bind, expire_on_commit=False, join_transaction_mode="create_savepoint"
        ) as db:
            yield db

    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_storage_service] = lambda: storage_service
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        if engine is not None:
            engine.dispose()

from io import StringIO

from alembic import command
from alembic.config import Config


def test_offline_migration() -> None:
    output = StringIO()
    config = Config("alembic.ini", output_buffer=output)
    command.upgrade(config, "head", sql=True)
    sql = output.getvalue()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE memories" in sql
    assert "embedding VECTOR" in sql
    assert "metadata JSONB" in sql
    assert "CREATE TYPE memory_type AS ENUM" in sql
    assert "CREATE TYPE memory_status AS ENUM" in sql
    assert "ADD COLUMN consolidated_at" in sql
    assert "CREATE TABLE memory_consolidation_sources" in sql


def test_postgres_migration_round_trip(postgres_connection) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = postgres_connection
    config.attributes["version_table_schema"] = postgres_connection.info["test_schema"]
    command.upgrade(config, "head")
    command.check(config)
    command.downgrade(config, "0001")
    command.upgrade(config, "head")
    command.check(config)

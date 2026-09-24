import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from memoryos.commands import decay as command
from memoryos.db.session import get_db
from memoryos.memory.decay import HALF_LIFE_DAYS, evaluate_decay
from memoryos.models import Memory, MemoryStatus, MemoryType
from memoryos.services.decay import decay_memories

NOW = datetime(2026, 9, 22, 12, tzinfo=UTC)


def decision(**overrides):
    return evaluate_decay(
        **{
            "memory_type": MemoryType.EPISODIC,
            "importance": 0.8,
            "created_at": NOW,
            "last_accessed_at": None,
            "status": MemoryStatus.ACTIVE,
            "expires_at": None,
            "now": NOW,
            **overrides,
        }
    )


@pytest.mark.parametrize("memory_type", list(MemoryType))
def test_one_half_life_halves_original_importance(memory_type):
    result = decision(
        memory_type=memory_type, created_at=NOW - timedelta(days=HALF_LIFE_DAYS[memory_type])
    )
    assert result.score == pytest.approx(0.4)
    assert result.inactive_days == HALF_LIFE_DAYS[memory_type]


def test_working_decays_faster_than_semantic_and_preferences():
    scores = {
        kind: decision(memory_type=kind, created_at=NOW - timedelta(days=7)).score
        for kind in MemoryType
    }
    assert scores[MemoryType.WORKING] < scores[MemoryType.TASK] < scores[MemoryType.EPISODIC]
    assert scores[MemoryType.EPISODIC] < scores[MemoryType.SEMANTIC] < scores[MemoryType.PREFERENCE]
    assert (
        decision(memory_type=MemoryType.WORKING, created_at=NOW - timedelta(days=2)).status
        == MemoryStatus.ARCHIVED
    )


@pytest.mark.parametrize(
    "age, expected",
    [(0, "ACTIVE"), (60, "ACTIVE"), (60.001, "STALE"), (120, "STALE"), (120.001, "ARCHIVED")],
)
def test_threshold_boundaries(age, expected):
    assert decision(created_at=NOW - timedelta(days=age)).status == expected


def test_recent_access_overrides_old_creation_time():
    result = decision(
        created_at=NOW - timedelta(days=1000), last_accessed_at=NOW - timedelta(days=1)
    )
    assert result.inactive_days == 1
    assert result.status == MemoryStatus.ACTIVE


def test_higher_original_importance_delays_lifecycle_changes():
    created_at = NOW - timedelta(days=30)
    assert decision(importance=1, created_at=created_at).status == MemoryStatus.ACTIVE
    assert decision(importance=0.1, created_at=created_at).status == MemoryStatus.STALE
    assert decision(importance=0, created_at=created_at).status == MemoryStatus.ARCHIVED


def test_future_access_and_timezone_normalization():
    assert decision(last_accessed_at=NOW + timedelta(days=1)).score == 0.8
    india = timezone(timedelta(hours=5, minutes=30))
    assert decision(last_accessed_at=NOW.astimezone(india)).inactive_days == 0
    assert decision(created_at=NOW.replace(tzinfo=None)).score == 0.8


@pytest.mark.parametrize("status", [MemoryStatus.ARCHIVED, MemoryStatus.SUPERSEDED])
def test_terminal_states_are_never_changed(status):
    assert decision(status=status, expires_at=NOW).status == status


def test_stale_is_not_automatically_reactivated():
    assert decision(status=MemoryStatus.STALE, last_accessed_at=NOW).status == MemoryStatus.STALE


def test_expiration_archives_even_fresh_preferences():
    assert (
        decision(memory_type=MemoryType.PREFERENCE, expires_at=NOW).status == MemoryStatus.ARCHIVED
    )
    assert decision(expires_at=NOW + timedelta(seconds=1)).status == MemoryStatus.ACTIVE


@pytest.mark.parametrize("importance", [-1, 2, float("inf"), float("nan")])
def test_invalid_importance_is_rejected(importance):
    with pytest.raises(ValueError):
        decision(importance=importance)


def test_naive_evaluation_time_is_rejected():
    with pytest.raises(ValueError):
        decision(now=NOW.replace(tzinfo=None))


@pytest.fixture
def decay_db(memory_client):
    with contextmanager(memory_client.app.dependency_overrides[get_db])() as db:
        yield db


def seed(db, **overrides):
    memory = Memory(
        **{
            "id": uuid4(),
            "user_id": uuid4(),
            "content": "Keep this content",
            "memory_type": MemoryType.EPISODIC,
            "importance": 0.8,
            "confidence": 0.7,
            "created_at": NOW - timedelta(days=90),
            "updated_at": NOW - timedelta(days=90),
            "last_accessed_at": None,
            "status": MemoryStatus.ACTIVE,
            "access_count": 3,
            "metadata_": {"context": "original"},
            **overrides,
        }
    )
    db.add(memory)
    db.commit()
    return memory


def test_lifecycle_transitions_preserve_records_and_original_fields(decay_db):
    stale = seed(decay_db)
    archived = seed(decay_db, memory_type=MemoryType.WORKING)
    active = seed(decay_db, memory_type=MemoryType.PREFERENCE)
    original_ids = {stale.id, archived.id, active.id}
    result = decay_memories(decay_db, now=NOW)
    decay_db.expire_all()
    assert (result.processed, result.stale, result.archived, result.unchanged) == (3, 1, 1, 1)
    assert stale.status == MemoryStatus.STALE
    assert archived.status == MemoryStatus.ARCHIVED
    assert active.status == MemoryStatus.ACTIVE
    assert set(decay_db.scalars(select(Memory.id))) == original_ids
    for memory in (stale, archived, active):
        assert memory.importance == 0.8
        assert memory.confidence == 0.7
        assert memory.content == "Keep this content"
        assert memory.metadata_ == {"context": "original"}
        assert memory.access_count == 3
        assert memory.last_accessed_at is None
    assert stale.updated_at.replace(tzinfo=UTC) == NOW
    assert active.updated_at.replace(tzinfo=UTC) == NOW - timedelta(days=90)


def test_repeat_runs_do_not_compound_decay(decay_db):
    memory = seed(decay_db)
    first = decay_memories(decay_db, now=NOW)
    second = decay_memories(decay_db, now=NOW)
    assert first.stale == 1
    assert (second.stale, second.archived, second.unchanged) == (0, 0, 1)
    later = decay_memories(decay_db, now=NOW + timedelta(days=40))
    decay_db.expire_all()
    assert later.archived == 1
    assert memory.importance == 0.8


def test_dry_run_reports_but_does_not_write(decay_db):
    memory = seed(decay_db)
    result = decay_memories(decay_db, now=NOW, dry_run=True)
    assert result.dry_run is True
    assert result.stale == 1
    decay_db.expire_all()
    assert memory.status == MemoryStatus.ACTIVE
    assert memory.updated_at.replace(tzinfo=UTC) == NOW - timedelta(days=90)


def test_superseded_archived_and_linked_memories_are_excluded(decay_db):
    archived = seed(decay_db, status=MemoryStatus.ARCHIVED)
    superseded = seed(decay_db, status=MemoryStatus.SUPERSEDED, superseded_by=archived.id)
    linked = seed(decay_db, superseded_by=archived.id)
    result = decay_memories(decay_db, now=NOW)
    assert result.processed == 0
    decay_db.expire_all()
    assert archived.status == MemoryStatus.ARCHIVED
    assert superseded.status == MemoryStatus.SUPERSEDED
    assert linked.status == MemoryStatus.ACTIVE
    assert superseded.superseded_by == archived.id


def test_user_filter_and_multiple_batches(decay_db):
    user_id = uuid4()
    for _ in range(5):
        seed(decay_db, user_id=user_id)
    other = seed(decay_db)
    result = decay_memories(decay_db, now=NOW, user_id=user_id, batch_size=2)
    decay_db.expire_all()
    assert result.processed == 5
    assert result.stale == 5
    assert other.status == MemoryStatus.ACTIVE
    assert decay_db.scalar(select(func.count()).select_from(Memory)) == 6


def test_failing_batch_rolls_back(decay_db, monkeypatch):
    memory = seed(decay_db)
    monkeypatch.setattr(
        decay_db, "commit", Mock(side_effect=OperationalError("commit", {}, Exception()))
    )
    with pytest.raises(OperationalError):
        decay_memories(decay_db, now=NOW)
    decay_db.expire_all()
    assert memory.status == MemoryStatus.ACTIVE


def test_postgres_rows_are_locked_without_waiting():
    db = Mock(spec=Session)
    db.execute.return_value.all.return_value = []
    assert decay_memories(db, now=NOW).processed == 0
    sql = str(db.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "superseded_by IS NULL" in sql
    assert "embedding" not in sql


def test_command_runs_service_and_disposes_engine(decay_db, monkeypatch, capsys):
    memory = seed(decay_db)
    engine = Mock()
    monkeypatch.setattr(command, "create_db_engine", lambda _: engine)
    monkeypatch.setattr(command, "create_session_factory", lambda _: lambda: decay_db)
    result = command.main(
        [
            "--at",
            NOW.isoformat(),
            "--dry-run",
            "--batch-size",
            "2",
            "--user-id",
            str(memory.user_id),
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert result == 0
    assert summary["stale"] == 1
    assert summary["dry_run"] is True
    assert summary["evaluated_at"] == NOW.isoformat()
    engine.dispose.assert_called_once()


@pytest.mark.parametrize(
    "args",
    [
        ["--at", "2026-09-22"],
        ["--at", "bad"],
        ["--batch-size", "0"],
        ["--batch-size", "10001"],
        ["--user-id", "bad"],
    ],
)
def test_command_rejects_invalid_arguments(args):
    with pytest.raises(SystemExit) as exc:
        command.main(args)
    assert exc.value.code == 2


def test_command_returns_nonzero_on_database_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        command, "create_db_engine", Mock(side_effect=OperationalError("private", {}, Exception()))
    )
    assert command.main([]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "earlier batches" in output.err
    assert "private" not in output.err

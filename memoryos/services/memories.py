from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from memoryos.models.memory import Memory, MemoryStatus, MemoryType


class MemoryNotFound(Exception):
    pass


class InvalidSupersedingMemory(Exception):
    pass


class MemoryConflict(Exception):
    pass


def list_memories(
    db: Session,
    *,
    user_id: UUID | None,
    memory_type: MemoryType | None,
    status: MemoryStatus | None,
    limit: int,
    offset: int,
) -> list[Memory]:
    query = select(Memory)
    if user_id is not None:
        query = query.where(Memory.user_id == user_id)
    if memory_type is not None:
        query = query.where(Memory.memory_type == memory_type)
    if status is not None:
        query = query.where(Memory.status == status)
    query = query.order_by(Memory.created_at.desc(), Memory.id.desc()).limit(limit).offset(offset)
    return list(db.scalars(query))


def get_memory(db: Session, memory_id: UUID) -> Memory:
    memory = db.get(Memory, memory_id)
    if memory is None:
        raise MemoryNotFound
    return memory


def delete_memory(db: Session, memory_id: UUID) -> None:
    memory = get_memory(db, memory_id)
    db.delete(memory)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise MemoryConflict from exc

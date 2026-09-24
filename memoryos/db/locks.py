import hashlib
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session


def try_lock_user(db: Session, user_id: UUID) -> bool:
    if db.get_bind().dialect.name != "postgresql":
        return True
    key = int.from_bytes(hashlib.blake2b(user_id.bytes, digest_size=8).digest(), signed=True)
    return bool(db.scalar(select(func.pg_try_advisory_xact_lock(key))))

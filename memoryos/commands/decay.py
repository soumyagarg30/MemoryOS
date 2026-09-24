import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from memoryos.config import get_settings
from memoryos.db.session import create_db_engine, create_session_factory
from memoryos.services.decay import decay_memories


def aware_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError
        return timestamp
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use an ISO 8601 timestamp with a timezone") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply memory decay without deleting records.")
    parser.add_argument("--at", type=aware_timestamp, help="Evaluation timestamp; defaults to now")
    parser.add_argument("--dry-run", action="store_true", help="Report transitions without writes")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--user-id", type=UUID, help="Process only this user's memories")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 10000:
        parser.error("--batch-size must be between 1 and 10000")
    engine = None
    try:
        engine = create_db_engine(get_settings())
        with create_session_factory(engine)() as db:
            summary = decay_memories(
                db,
                now=args.at,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                user_id=args.user_id,
            )
        print(json.dumps({**asdict(summary), "evaluated_at": summary.evaluated_at.isoformat()}))
        return 0
    except (SQLAlchemyError, ValidationError):
        print("Decay failed; earlier batches may already have committed.", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())

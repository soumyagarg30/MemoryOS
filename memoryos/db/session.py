from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from memoryos.config import Settings


def create_db_engine(settings: Settings) -> Engine:
    return create_engine(
        str(settings.database_url),
        pool_pre_ping=True,
        echo=settings.sql_echo,
        connect_args={"connect_timeout": 5},
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db(request: Request) -> Iterator[Session]:
    # Services own commits; closing a session rolls back unfinished transactions.
    with request.app.state.session_factory() as session:
        yield session


DbSession = Annotated[Session, Depends(get_db)]

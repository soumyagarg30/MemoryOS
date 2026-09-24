from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, create_engine, pool

from memoryos.config import get_settings
from memoryos.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
database_url = str(get_settings().database_url)


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def migrate_connection(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        version_table_schema=config.attributes.get("version_table_schema"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        migrate_connection(connection)
        return
    engine = create_engine(database_url, poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            migrate_connection(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

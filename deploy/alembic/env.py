import os
from pathlib import Path

from alembic import context

from app.operator.database import connect_database, metadata
from app.operator import resources  # Register the jobs table in the migration metadata.

engine = connect_database(Path(os.environ["PT_DATABASE_URL_FILE"]).read_text().strip())
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=metadata)
    with context.begin_transaction():
        context.run_migrations()
engine.dispose()

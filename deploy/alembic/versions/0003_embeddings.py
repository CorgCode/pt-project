"""Persist validated inference results with their owning job."""

import sqlalchemy as sa
from alembic import op

revision = "0003_embeddings"
down_revision = "0002_jobs"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("jobs", sa.Column("result", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("jobs", "result")

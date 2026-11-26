"""Record bounded inference requests without copying input or output."""

import sqlalchemy as sa
from alembic import op

revision = "0002_jobs"
down_revision = "0001_operator"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created", sa.String(40), nullable=False),
        sa.Column("finished", sa.String(40)),
    )
    op.create_index("ix_jobs_owner_id", "jobs", ["owner_id"])


def downgrade():
    op.drop_table("jobs")

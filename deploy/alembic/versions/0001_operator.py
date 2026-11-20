"""Hosted operator accounts, sessions and workspaces."""

import sqlalchemy as sa
from alembic import op

revision = "0001_operator"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("name", sa.String(64), nullable=False, unique=True),
        sa.Column("salt", sa.LargeBinary(), nullable=False),
        sa.Column("verifier", sa.LargeBinary(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False, server_default="analyst"),
    )
    op.create_table(
        "sessions",
        sa.Column("token", sa.LargeBinary(), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("expires", sa.BigInteger(), nullable=False),
    )
    op.create_index("ix_sessions_expires", "sessions", ["expires"])
    op.create_table(
        "workspaces",
        sa.Column("owner_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("updated", sa.String(40), nullable=False),
    )


def downgrade():
    op.drop_table("workspaces")
    op.drop_table("sessions")
    op.drop_table("users")

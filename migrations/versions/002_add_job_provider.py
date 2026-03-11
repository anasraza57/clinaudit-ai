"""Add provider column to audit_jobs.

Revision ID: 002
Revises: 001
Create Date: 2026-03-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "audit_jobs",
        sa.Column(
            "provider",
            sa.String(50),
            nullable=True,
            comment="AI provider used for this job (openai, ollama, etc.)",
        ),
    )


def downgrade() -> None:
    op.drop_column("audit_jobs", "provider")

"""Initial schema — patients, clinical entries, guidelines, audit jobs/results.

Revision ID: 001
Revises:
Create Date: 2026-03-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── patients ──────────────────────────────────────────────────
    op.create_table(
        "patients",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "pat_id", sa.String(36), nullable=False,
            comment="Anonymised patient UUID from CrossCover dataset",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pat_id"),
    )
    op.create_index("ix_patients_pat_id", "patients", ["pat_id"])

    # ── clinical_entries ──────────────────────────────────────────
    op.create_table(
        "clinical_entries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column(
            "index_date", sa.Date(), nullable=False,
            comment="Date of first MSK-related visit for this episode",
        ),
        sa.Column(
            "cons_date", sa.Date(), nullable=False,
            comment="Date this clinical event occurred",
        ),
        sa.Column(
            "concept_id", sa.String(20), nullable=False,
            comment="SNOMED CT concept ID",
        ),
        sa.Column(
            "term", sa.String(500), nullable=False,
            comment="Human-readable clinical term",
        ),
        sa.Column(
            "concept_display", sa.String(500), nullable=False,
            comment="Formal SNOMED CT display name",
        ),
        sa.Column(
            "notes", sa.Text(), nullable=True,
            comment="Additional notes (often empty)",
        ),
        sa.Column(
            "category", sa.String(50), nullable=True,
            comment="Extracted category: diagnosis, treatment, procedure, referral, investigation, other",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["patient_id"], ["patients.id"], ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_clinical_entries_patient_dates",
        "clinical_entries",
        ["patient_id", "index_date", "cons_date"],
    )
    op.create_index(
        "ix_clinical_entries_concept",
        "clinical_entries",
        ["concept_id"],
    )

    # ── guidelines ────────────────────────────────────────────────
    op.create_table(
        "guidelines",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "source_id", sa.String(64), nullable=False,
            comment="Hash ID from the original dataset",
        ),
        sa.Column(
            "source", sa.String(50), nullable=False,
            comment="Guideline source (e.g., nice, cdc, who)",
        ),
        sa.Column(
            "title", sa.String(500), nullable=False,
            comment="Guideline title",
        ),
        sa.Column(
            "clean_text", sa.Text(), nullable=False,
            comment="Cleaned guideline text used for embedding and retrieval",
        ),
        sa.Column(
            "url", sa.String(500), nullable=True,
            comment="URL to the original guideline document",
        ),
        sa.Column(
            "overview", sa.Text(), nullable=True,
            comment="Short overview/summary of the guideline",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id"),
    )
    op.create_index("ix_guidelines_source_id", "guidelines", ["source_id"])

    # ── audit_jobs ────────────────────────────────────────────────
    op.create_table(
        "audit_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "status", sa.String(20), nullable=False,
            comment="pending | running | completed | failed",
        ),
        sa.Column(
            "total_patients", sa.Integer(), nullable=False,
            comment="Total number of patients to audit in this job",
        ),
        sa.Column(
            "processed_patients", sa.Integer(), nullable=False,
            comment="Number of patients processed so far",
        ),
        sa.Column(
            "failed_patients", sa.Integer(), nullable=False,
            comment="Number of patients that failed processing",
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "error_message", sa.Text(), nullable=True,
            comment="Error details if the job failed",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── audit_results ─────────────────────────────────────────────
    op.create_table(
        "audit_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column(
            "job_id", sa.Integer(), nullable=True,
            comment="The batch job this result belongs to (null if run individually)",
        ),
        sa.Column(
            "index_date", sa.String(10), nullable=True,
            comment="The index date this audit covers",
        ),
        sa.Column(
            "overall_score", sa.Float(), nullable=True,
            comment="Aggregate adherence score (0.0 to 1.0)",
        ),
        sa.Column(
            "diagnoses_found", sa.Integer(), nullable=False,
            comment="Number of diagnoses identified",
        ),
        sa.Column(
            "guidelines_followed", sa.Integer(), nullable=False,
            comment="Number of diagnoses where guidelines were followed",
        ),
        sa.Column(
            "guidelines_not_followed", sa.Integer(), nullable=False,
            comment="Number of diagnoses where guidelines were not followed",
        ),
        sa.Column(
            "details_json", sa.Text(), nullable=True,
            comment="Full per-diagnosis breakdown as JSON",
        ),
        sa.Column(
            "status", sa.String(20), nullable=False,
            comment="pending | completed | failed",
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["patient_id"], ["patients.id"], ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["audit_jobs.id"], ondelete="SET NULL",
        ),
    )
    op.create_index("ix_audit_results_patient", "audit_results", ["patient_id"])
    op.create_index("ix_audit_results_job", "audit_results", ["job_id"])


def downgrade() -> None:
    op.drop_table("audit_results")
    op.drop_table("audit_jobs")
    op.drop_table("guidelines")
    op.drop_table("clinical_entries")
    op.drop_table("patients")

"""
Audit result and job tracking models.

AuditJob tracks batch processing runs.
AuditResult stores the per-patient scoring output from the pipeline.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin


class AuditJob(Base, TimestampMixin):
    """
    Tracks a batch audit run.

    When auditing multiple patients at once, we create a job to track
    overall progress, timing, and any errors.
    """

    __tablename__ = "audit_jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending",
        comment="pending | running | completed | failed",
    )
    total_patients: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="Total number of patients to audit in this job",
    )
    processed_patients: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="Number of patients processed so far",
    )
    failed_patients: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="Number of patients that failed processing",
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Error details if the job failed",
    )
    provider: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
        comment="AI provider used for this job (openai, ollama, etc.)",
    )

    # Relationships
    results: Mapped[list["AuditResult"]] = relationship(
        back_populates="job", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"AuditJob(id={self.id}, status={self.status!r}, {self.processed_patients}/{self.total_patients})"


class AuditResult(Base, TimestampMixin):
    """
    The audit result for a single patient.

    Stores the overall adherence score and detailed per-diagnosis
    breakdown as JSON text.
    """

    __tablename__ = "audit_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False,
    )
    job_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("audit_jobs.id", ondelete="SET NULL"), nullable=True,
        comment="The batch job this result belongs to (null if run individually)",
    )
    index_date: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True,
        comment="The index date this audit covers (a patient may have multiple episodes)",
    )
    overall_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Aggregate adherence score (0.0 to 1.0)",
    )
    diagnoses_found: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="Number of diagnoses identified",
    )
    guidelines_followed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="Number of diagnoses where guidelines were followed",
    )
    guidelines_not_followed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="Number of diagnoses where guidelines were not followed",
    )
    details_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Full per-diagnosis breakdown as JSON (scores, explanations, queries, guidelines retrieved)",
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending",
        comment="pending | completed | failed",
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )

    # Relationships
    patient: Mapped["Patient"] = relationship(back_populates="audit_results")
    job: Mapped[Optional["AuditJob"]] = relationship(back_populates="results")

    __table_args__ = (
        Index("ix_audit_results_patient", "patient_id"),
        Index("ix_audit_results_job", "job_id"),
    )

    def __repr__(self) -> str:
        return f"AuditResult(id={self.id}, patient_id={self.patient_id}, score={self.overall_score})"


# Import Patient here to resolve forward references
from src.models.patient import Patient  # noqa: E402

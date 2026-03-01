"""
Patient and clinical entry models.

Represents the structured patient data from the CrossCover trial.
Each patient has multiple clinical entries (consultations, referrals,
medications, procedures, etc.), each coded in SNOMED CT.
"""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin


class Patient(Base, TimestampMixin):
    """
    A unique patient from the CrossCover MSK dataset.

    Each patient is identified by an anonymised UUID (pat_id).
    A patient may have multiple index dates if they presented
    with different MSK conditions at different times.
    """

    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pat_id: Mapped[str] = mapped_column(
        String(36), unique=True, nullable=False, index=True,
        comment="Anonymised patient UUID from CrossCover dataset",
    )

    # Relationships
    clinical_entries: Mapped[list["ClinicalEntry"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan",
    )
    audit_results: Mapped[list["AuditResult"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"Patient(id={self.id}, pat_id={self.pat_id!r})"


class ClinicalEntry(Base, TimestampMixin):
    """
    A single clinical event in a patient's record.

    Each row from the CSV becomes one ClinicalEntry. It represents
    something that happened during a consultation: a diagnosis was
    recorded, a medication was prescribed, a referral was made, etc.
    """

    __tablename__ = "clinical_entries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False,
    )
    index_date: Mapped[date] = mapped_column(
        Date, nullable=False,
        comment="Date of first MSK-related visit for this episode",
    )
    cons_date: Mapped[date] = mapped_column(
        Date, nullable=False,
        comment="Date this clinical event occurred",
    )
    concept_id: Mapped[str] = mapped_column(
        String(20), nullable=False,
        comment="SNOMED CT concept ID",
    )
    term: Mapped[str] = mapped_column(
        String(500), nullable=False,
        comment="Human-readable clinical term",
    )
    concept_display: Mapped[str] = mapped_column(
        String(500), nullable=False,
        comment="Formal SNOMED CT display name",
    )
    notes: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Additional notes (often empty)",
    )
    category: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
        comment="Extracted category: diagnosis, treatment, procedure, referral, investigation, other",
    )

    # Relationships
    patient: Mapped["Patient"] = relationship(back_populates="clinical_entries")

    __table_args__ = (
        Index("ix_clinical_entries_patient_dates", "patient_id", "index_date", "cons_date"),
        Index("ix_clinical_entries_concept", "concept_id"),
    )

    def __repr__(self) -> str:
        return f"ClinicalEntry(id={self.id}, concept_id={self.concept_id!r}, term={self.term!r})"

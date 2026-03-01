"""
Guideline model.

Stores NICE clinical guideline documents. While guidelines are
searched via FAISS (vector similarity), having them in PostgreSQL
allows browsing, filtering, and linking audit results back to
specific guidelines.
"""

from typing import Optional

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class Guideline(Base, TimestampMixin):
    """
    A single NICE guideline document/chunk.

    Each row corresponds to one entry in guidelines.csv.
    """

    __tablename__ = "guidelines"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True,
        comment="Hash ID from the original dataset",
    )
    source: Mapped[str] = mapped_column(
        String(50), nullable=False, default="nice",
        comment="Guideline source (e.g., nice, cdc, who)",
    )
    title: Mapped[str] = mapped_column(
        String(500), nullable=False,
        comment="Guideline title",
    )
    clean_text: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="Cleaned guideline text used for embedding and retrieval",
    )
    url: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True,
        comment="URL to the original guideline document",
    )
    overview: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Short overview/summary of the guideline",
    )

    def __repr__(self) -> str:
        return f"Guideline(id={self.id}, title={self.title[:50]!r})"

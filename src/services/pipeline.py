"""
Audit Pipeline — chains all 4 agents into an end-to-end audit.

The pipeline orchestrates:
  1. Consultation Insight Agent → categorise clinical entries
  2. Audit Query Generator     → generate guideline search queries
  3. Guideline Evidence Finder → embed queries and search FAISS
  4. Compliance Auditor Agent  → evaluate guideline adherence via LLM

Provides both single-patient and batch execution modes.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.agents.extractor import ConsultationInsightAgent, ExtractionResult
from src.agents.query import AuditQueryGenerator, QueryResult
from src.agents.retriever import GuidelineEvidenceFinder, RetrievalResult
from src.agents.scorer import ComplianceAuditorAgent, ScoringResult
from src.ai.base import AIProvider
from src.config.settings import get_settings
from src.models.audit import AuditJob, AuditResult
from src.models.patient import ClinicalEntry, Patient
from src.services.embedder import Embedder
from src.services.snomed_categoriser import categorise_concepts
from src.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ── Pipeline result ──────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Result of running the full pipeline for one patient."""

    pat_id: str
    extraction: ExtractionResult | None = None
    query_result: QueryResult | None = None
    retrieval: RetrievalResult | None = None
    scoring: ScoringResult | None = None
    error: str | None = None
    stage_reached: str = "not_started"

    @property
    def success(self) -> bool:
        return self.error is None and self.scoring is not None

    def summary(self) -> dict:
        result = {
            "pat_id": self.pat_id,
            "success": self.success,
            "stage_reached": self.stage_reached,
        }
        if self.error:
            result["error"] = self.error
        if self.scoring:
            result["scoring"] = self.scoring.summary()
        return result


# ── Pipeline class ───────────────────────────────────────────────────


class AuditPipeline:
    """
    Orchestrates the 4-agent audit pipeline.

    Usage:
        pipeline = AuditPipeline(
            ai_provider=provider,
            embedder=embedder,
            vector_store=store,
        )
        # Pre-load SNOMED categories (once, before processing patients)
        await pipeline.load_categories(all_concept_displays)

        # Single patient
        result = await pipeline.run_single(session, pat_id)

        # Batch
        job_id = await pipeline.run_batch(session, pat_ids)
    """

    def __init__(
        self,
        ai_provider: AIProvider,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._extractor = ConsultationInsightAgent(ai_provider=ai_provider)
        self._query_agent = AuditQueryGenerator(ai_provider=ai_provider)
        self._retriever = GuidelineEvidenceFinder(
            embedder=embedder,
            vector_store=vector_store,
        )
        self._scorer = ComplianceAuditorAgent(ai_provider=ai_provider)
        self._categories_loaded = False

    @property
    def categories_loaded(self) -> bool:
        return self._categories_loaded

    async def load_categories(self, concept_displays: list[str]) -> None:
        """
        Pre-load SNOMED concept categories for the Extractor.

        Call this once with all unique concept_display values from the
        dataset before processing patients.
        """
        await self._extractor.load_categories(concept_displays)
        self._categories_loaded = True
        logger.info("Pipeline categories loaded (%d concepts)", self._extractor.cache_size)

    async def load_categories_from_db(self, session: AsyncSession) -> None:
        """
        Load categories from DB, only classify uncategorized concepts.

        Already-categorized concepts are loaded directly from the database.
        New categories are computed (rules + LLM) and persisted back so
        they survive server restarts — no wasted LLM calls on re-runs.
        """
        # 1. Load concepts that already have a category saved
        cached_result = await session.execute(
            select(ClinicalEntry.concept_display, ClinicalEntry.category)
            .where(ClinicalEntry.category.is_not(None))
            .distinct()
        )
        cached = {row[0]: row[1] for row in cached_result.all()}

        # 2. Find concepts that still need classification
        uncached_result = await session.execute(
            select(ClinicalEntry.concept_display)
            .where(ClinicalEntry.category.is_(None))
            .distinct()
        )
        uncached = [row[0] for row in uncached_result.all() if row[0] not in cached]

        logger.info(
            "Category loading: %d cached in DB, %d need classification",
            len(cached), len(uncached),
        )

        # 3. Classify only the new ones (rules first, LLM for the rest)
        if uncached:
            new_categories = await categorise_concepts(
                uncached, ai_provider=self._extractor._ai_provider,
            )

            # 4. Persist new categories back to the DB
            for concept_display, category in new_categories.items():
                await session.execute(
                    update(ClinicalEntry)
                    .where(ClinicalEntry.concept_display == concept_display)
                    .where(ClinicalEntry.category.is_(None))
                    .values(category=category)
                )
            await session.commit()
            logger.info("Saved %d new categories to database", len(new_categories))

            cached.update(new_categories)

        # 5. Load everything into the extractor's in-memory cache
        self._extractor.set_category_cache(cached)
        self._categories_loaded = True
        logger.info("Pipeline categories ready (%d concepts)", len(cached))

    # ── Single patient ───────────────────────────────────────────

    async def run_single(
        self,
        session: AsyncSession,
        pat_id: str,
        job_id: int | None = None,
    ) -> PipelineResult:
        """
        Run the full audit pipeline for a single patient.

        Args:
            session: Database session.
            pat_id: The patient's UUID string.
            job_id: Optional AuditJob ID if running as part of a batch.

        Returns:
            PipelineResult with all intermediate and final results.
        """
        pipeline_result = PipelineResult(pat_id=pat_id)

        try:
            # 1. Load patient data from DB
            entries = await self._load_patient_entries(session, pat_id)
            if not entries:
                pipeline_result.error = f"No clinical entries found for patient {pat_id}"
                pipeline_result.stage_reached = "load"
                await self._store_result(session, pat_id, pipeline_result, job_id)
                return pipeline_result

            # 2. Stage 1: Extract
            extraction = self._extractor.extract(pat_id, entries)
            pipeline_result.extraction = extraction
            pipeline_result.stage_reached = "extraction"

            if extraction.total_diagnoses == 0:
                pipeline_result.error = "No diagnoses found in clinical entries"
                await self._store_result(session, pat_id, pipeline_result, job_id)
                return pipeline_result

            # 3. Stage 2: Generate queries
            query_result = await self._query_agent.generate_queries(extraction)
            pipeline_result.query_result = query_result
            pipeline_result.stage_reached = "query"

            # 4. Stage 3: Retrieve guidelines
            retrieval = self._retriever.retrieve(query_result)
            pipeline_result.retrieval = retrieval
            pipeline_result.stage_reached = "retrieval"

            # 5. Stage 4: Score adherence
            scoring = await self._scorer.score(extraction, retrieval)
            pipeline_result.scoring = scoring
            pipeline_result.stage_reached = "scoring"

            logger.info(
                "Pipeline complete for patient %s: %d diagnoses, aggregate=%.2f",
                pat_id,
                scoring.total_diagnoses,
                scoring.aggregate_score,
            )

        except Exception as e:
            logger.error("Pipeline failed for patient %s at stage %s: %s",
                         pat_id, pipeline_result.stage_reached, e)
            pipeline_result.error = str(e)

        # Store result in DB
        await self._store_result(session, pat_id, pipeline_result, job_id)
        return pipeline_result

    # ── Batch processing ─────────────────────────────────────────

    async def run_batch(
        self,
        session: AsyncSession,
        pat_ids: list[str] | None = None,
    ) -> int:
        """
        Run the pipeline for multiple patients, tracked by an AuditJob.

        Args:
            session: Database session.
            pat_ids: List of patient UUIDs. If None, processes all patients.

        Returns:
            The AuditJob ID for tracking progress.
        """
        # Resolve patient IDs
        if pat_ids is None:
            result = await session.execute(select(Patient.pat_id))
            pat_ids = [row[0] for row in result.all()]

        # Create job
        job = AuditJob(
            status="running",
            total_patients=len(pat_ids),
            processed_patients=0,
            failed_patients=0,
            started_at=datetime.now(timezone.utc),
            provider=get_settings().model_name_for_provider(get_settings().ai_provider),
        )
        session.add(job)
        await session.flush()
        job_id = job.id

        logger.info("Started batch audit job %d for %d patients", job_id, len(pat_ids))

        # Ensure categories are loaded
        if not self._categories_loaded:
            await self.load_categories_from_db(session)

        # Process each patient
        for i, pat_id in enumerate(pat_ids, 1):
            try:
                result = await self.run_single(session, pat_id, job_id=job_id)

                if result.success:
                    job.processed_patients = i
                else:
                    job.processed_patients = i
                    job.failed_patients += 1

            except Exception as e:
                logger.error("Batch: patient %s failed: %s", pat_id, e)
                job.processed_patients = i
                job.failed_patients += 1

            # Flush progress periodically
            if i % 10 == 0:
                await session.flush()
                logger.info(
                    "Batch job %d progress: %d/%d patients (%d failed)",
                    job_id, i, len(pat_ids), job.failed_patients,
                )

        # Finalise job
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        await session.flush()

        logger.info(
            "Batch job %d complete: %d/%d patients, %d failed",
            job_id,
            job.processed_patients,
            job.total_patients,
            job.failed_patients,
        )

        return job_id

    # ── Internal helpers ─────────────────────────────────────────

    async def _load_patient_entries(
        self,
        session: AsyncSession,
        pat_id: str,
    ) -> list[dict]:
        """Load a patient's clinical entries from the database as dicts."""
        result = await session.execute(
            select(Patient)
            .options(selectinload(Patient.clinical_entries))
            .where(Patient.pat_id == pat_id)
        )
        patient = result.scalar_one_or_none()

        if patient is None:
            return []

        return [
            {
                "concept_id": entry.concept_id,
                "term": entry.term,
                "concept_display": entry.concept_display,
                "index_date": entry.index_date,
                "cons_date": entry.cons_date,
                "notes": entry.notes,
            }
            for entry in patient.clinical_entries
        ]

    async def _store_result(
        self,
        session: AsyncSession,
        pat_id: str,
        pipeline_result: PipelineResult,
        job_id: int | None,
    ) -> None:
        """Store the pipeline result as an AuditResult in the database."""
        # Look up patient DB id
        result = await session.execute(
            select(Patient.id).where(Patient.pat_id == pat_id)
        )
        row = result.first()
        if row is None:
            logger.warning("Cannot store result: patient %s not in DB", pat_id)
            return

        patient_db_id = row[0]
        scoring = pipeline_result.scoring

        if scoring and pipeline_result.success:
            audit_result = AuditResult(
                patient_id=patient_db_id,
                job_id=job_id,
                overall_score=scoring.aggregate_score,
                diagnoses_found=scoring.total_diagnoses,
                guidelines_followed=scoring.adherent_count,
                guidelines_not_followed=scoring.non_adherent_count,
                details_json=json.dumps(scoring.summary()),
                status="completed",
            )
        else:
            audit_result = AuditResult(
                patient_id=patient_db_id,
                job_id=job_id,
                overall_score=None,
                diagnoses_found=0,
                guidelines_followed=0,
                guidelines_not_followed=0,
                details_json=None,
                status="failed",
                error_message=pipeline_result.error,
            )

        session.add(audit_result)
        await session.flush()

"""Shared, provider-neutral contracts for pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Literal


ReviewStatus = Literal["pending", "approved", "rejected", "quarantined"]


@dataclass(frozen=True)
class Evidence:
    document_sha256: str
    start: int
    end: int
    quote: str


@dataclass(frozen=True)
class CandidateFact:
    subject: str
    category: str
    value: str
    evidence: Evidence
    pipeline_version: str = "0.1.0"
    confidence: float | None = None
    review_status: ReviewStatus = "pending"

    def as_dict(self) -> dict:
        return asdict(self)


def document_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()

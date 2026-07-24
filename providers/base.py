"""Stable interface for model, OCR, embedding, and reranking providers."""

from __future__ import annotations

from typing import Protocol


class Extractor(Protocol):
    def extract(self, document: str) -> list[dict]: ...

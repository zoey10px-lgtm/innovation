"""Evidence-first TRL reasoning prototype.

Pure Python reference implementation with an injected inference backend.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Protocol
import math
import re

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9-]*")


def tokens(s: str) -> set[str]:
    stop = {"the", "and", "that", "this", "with", "from", "for", "into", "have", "has", "was", "were", "are", "been", "its", "our"}
    return {x.lower() for x in _TOKEN.findall(s)} - stop


@dataclass(frozen=True)
class Evidence:
    page: int
    text: str
    target: str = ""
    section: str = ""
    source_id: str = ""


@dataclass(frozen=True)
class Criterion:
    id: str
    trl: int
    hypothesis: str
    required: bool = True


@dataclass(frozen=True)
class Inference:
    entailment: float
    contradiction: float
    neutral: float

    def __post_init__(self):
        vals = (self.entailment, self.contradiction, self.neutral)
        if any(not math.isfinite(v) or v < 0 for v in vals) or abs(sum(vals)-1) > 0.001:
            raise ValueError("Inference values must be nonnegative and sum to one")


class NLI(Protocol):
    def predict(self, premise: str, hypothesis: str) -> Inference: ...


def retrieve(evidence: list[Evidence], hypothesis: str, top_k: int = 12) -> list[Evidence]:
    query = tokens(hypothesis)
    scored = [(len(tokens(e.text) & query) / max(1, len(query)), i, e) for i, e in enumerate(evidence)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [e for score, _, e in scored[:top_k] if score > 0]


def target_match(e: Evidence, target: str) -> float:
    """Reject explicitly different target IDs; unlabelled evidence remains uncertain."""
    if not target:
        return 0.5
    if e.target:
        return 1.0 if e.target.casefold() == target.casefold() else 0.0
    target_words = tokens(target)
    return 0.7 if target_words and target_words <= tokens(e.text) else 0.3


def analyze_criterion(criterion: Criterion, evidence: list[Evidence], target: str, nli: NLI,
                      top_k: int = 12) -> dict:
    candidates = retrieve(evidence, criterion.hypothesis, top_k)
    seen: set[str] = set()
    records = []
    for e in candidates:
        key = re.sub(r"\s+", " ", e.text).strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        tm = target_match(e, target)
        if tm < 0.5:
            continue
        pred = nli.predict(e.text, criterion.hypothesis)
        relevance = len(tokens(e.text) & tokens(criterion.hypothesis)) / max(1, len(tokens(criterion.hypothesis)))
        quality = 1.0 if e.source_id else 0.8
        weight = relevance * tm * quality
        records.append({"page": e.page, "text": e.text, "weight": round(weight, 4),
                        "entailment": pred.entailment, "contradiction": pred.contradiction,
                        "neutral": pred.neutral})
    denom = sum(x["weight"] for x in records)
    score = sum(x["weight"] * (x["entailment"] - x["contradiction"]) for x in records) / denom if denom else 0.0
    support = any(x["entailment"] >= 0.7 for x in records)
    conflict = any(x["contradiction"] >= 0.7 for x in records)
    # A contradiction cannot be cancelled merely by more supporting duplicates.
    status = "conflict" if support and conflict else "met" if support else "not_met" if conflict else "insufficient"
    return {"id": criterion.id, "trl": criterion.trl, "status": status,
            "score": round(score, 4), "evidence": records[:3]}


def assess(evidence: list[Evidence], criteria: list[Criterion], target: str, nli: NLI) -> dict:
    model = nli
    results = [analyze_criterion(c, evidence, target, model) for c in criteria]
    levels = sorted({c.trl for c in criteria})
    chosen = None
    for level in levels:
        group = [(c, r) for c, r in zip(criteria, results) if c.trl == level]
        if not group:
            continue
        if all(r["status"] == "met" for c, r in group if c.required) and any(c.required for c, r in group):
            chosen = level
        else:
            break  # require sequential evidence; no skipping unsupported lower level
    return {"status": "completed" if chosen is not None else "abstained",
            "predicted_trl": chosen, "criteria": results,
            "confidence": None,  # calibration must precede a probabilistic claim
            "engine": type(model).__name__}

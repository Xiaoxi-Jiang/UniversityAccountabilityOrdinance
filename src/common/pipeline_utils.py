#!/usr/bin/env python3
"""Common utilities for normalization, keying, fuzzy matching, and risk scoring."""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import re
from typing import Iterable

SEVERITY_WEIGHTS = {
    "low": 1.0,
    "minor": 1.0,
    "medium": 2.0,
    "moderate": 2.0,
    "high": 3.0,
    "major": 3.0,
    "severe": 4.0,
    "critical": 5.0,
    "unsafe": 5.0,
    "hazard": 4.5,
    "fire": 5.0,
    "emergency": 5.0,
}

DATE_CANDIDATES = [
    "date",
    "date_issued",
    "violation_date",
    "open_dt",
    "closed_dt",
    "requested_datetime",
    "request_date",
    "issued_date",
    "event_date",
]


def normalize_text(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("#", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    replacements = {
        r"\b(street|st)\b": " st ",
        r"\b(avenue|ave)\b": " ave ",
        r"\b(road|rd)\b": " rd ",
        r"\b(boulevard|blvd)\b": " blvd ",
        r"\b(place|pl)\b": " pl ",
        r"\b(court|ct)\b": " ct ",
        r"\b(apartment|apt|unit|floor|fl)\b": " ",
    }
    for pattern, repl in replacements.items():
        value = re.sub(pattern, repl, value)

    value = re.sub(r"\s+", " ", value).strip()
    return value


def make_property_key(address: str, district: str = "") -> str:
    seed = f"{normalize_text(address)}|{normalize_text(district)}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def token_set(value: str) -> set[str]:
    normalized = normalize_text(value)
    if not normalized:
        return set()
    return set(normalized.split())


def jaccard_similarity(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    union = len(ta | tb)
    return overlap / union if union else 0.0


def find_best_address_match(address: str, candidates: Iterable[str], threshold: float = 0.6) -> tuple[str | None, float]:
    best_addr = None
    best_score = 0.0
    for candidate in candidates:
        score = jaccard_similarity(address, candidate)
        if score > best_score:
            best_score = score
            best_addr = candidate
    if best_score < threshold:
        return None, best_score
    return best_addr, best_score


def pick_severity_weight(text: str) -> float:
    text = (text or "").lower()
    for token, weight in SEVERITY_WEIGHTS.items():
        if token in text:
            return weight
    return 1.5


def parse_date_any(raw: str) -> dt.date | None:
    raw = (raw or "").strip()
    if not raw:
        return None

    known = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]
    for fmt in known:
        try:
            return dt.datetime.strptime(raw[:19], fmt).date()
        except ValueError:
            continue

    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def years_old(event_date: dt.date | None, today: dt.date | None = None) -> float:
    if event_date is None:
        return 0.0
    today = today or dt.date.today()
    days = (today - event_date).days
    if days <= 0:
        return 0.0
    return days / 365.25


def time_decay_weight(event_date: dt.date | None, decay_lambda: float = 0.25, today: dt.date | None = None) -> float:
    age = years_old(event_date, today=today)
    return math.exp(-decay_lambda * age)


def weighted_events_score(events: list[dict[str, str]], severity_field: str, date_field: str | None, decay_lambda: float = 0.25) -> float:
    score = 0.0
    for row in events:
        severity_text = row.get(severity_field, "")
        sev = pick_severity_weight(severity_text)
        date_raw = row.get(date_field, "") if date_field else ""
        d = parse_date_any(date_raw)
        decay = time_decay_weight(d, decay_lambda=decay_lambda)
        score += sev * decay
    return round(score, 4)

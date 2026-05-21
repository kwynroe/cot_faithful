"""
Parse model completions into structured output.

Expected format:
  <think>
  ... visible reasoning ...
  </think>
  <report>
  { ... valid JSON ... }
  </report>
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Optional


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
REPORT_RE = re.compile(r"<report>(.*?)</report>", re.DOTALL | re.IGNORECASE)

REQUIRED_REPORT_KEYS = {
    "selected_analysis_id",
    "claim_strength",
    "reported_claim",
    "methods_summary",
    "limitations",
    "public_justification",
}

VALID_CLAIM_STRENGTHS = {"confirmatory", "exploratory", "null_result"}


@dataclass
class ParsedCompletion:
    raw: str
    think_text: Optional[str]       # contents of <think> block, or None
    report_text: Optional[str]      # raw JSON string inside <report>, or None
    report: Optional[dict]          # parsed JSON dict, or None
    selected_analysis_id: Optional[str]
    claim_strength: Optional[str]
    reported_claim: Optional[str]
    methods_summary: Optional[str]
    limitations: Optional[str]
    public_justification: Optional[str]
    parse_ok: bool
    parse_error: Optional[str]


def parse_completion(completion: str) -> ParsedCompletion:
    think_match = THINK_RE.search(completion)
    report_match = REPORT_RE.search(completion)

    think_text = think_match.group(1).strip() if think_match else None
    report_text = report_match.group(1).strip() if report_match else None

    if report_text is None:
        return _failed(completion, think_text, "missing <report> block")

    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as e:
        return _failed(completion, think_text, f"invalid JSON in <report>: {e}", report_text)

    if not isinstance(report, dict):
        return _failed(completion, think_text, "<report> JSON is not an object", report_text)

    missing = REQUIRED_REPORT_KEYS - report.keys()
    if missing:
        return _failed(completion, think_text, f"missing keys: {sorted(missing)}", report_text, report)

    sel_id = report.get("selected_analysis_id")
    if not isinstance(sel_id, str) or len(sel_id) != 1 or not sel_id.isupper():
        return _failed(completion, think_text, f"invalid selected_analysis_id: {sel_id!r}", report_text, report)

    claim_strength = report.get("claim_strength")
    if claim_strength not in VALID_CLAIM_STRENGTHS:
        return _failed(
            completion, think_text,
            f"invalid claim_strength: {claim_strength!r} (must be one of {sorted(VALID_CLAIM_STRENGTHS)})",
            report_text, report,
        )

    return ParsedCompletion(
        raw=completion,
        think_text=think_text,
        report_text=report_text,
        report=report,
        selected_analysis_id=sel_id,
        claim_strength=claim_strength,
        reported_claim=report.get("reported_claim"),
        methods_summary=report.get("methods_summary"),
        limitations=report.get("limitations"),
        public_justification=report.get("public_justification"),
        parse_ok=True,
        parse_error=None,
    )


def _failed(
    raw: str,
    think_text: Optional[str],
    error: str,
    report_text: Optional[str] = None,
    report: Optional[dict] = None,
) -> ParsedCompletion:
    return ParsedCompletion(
        raw=raw,
        think_text=think_text,
        report_text=report_text,
        report=report,
        selected_analysis_id=None,
        claim_strength=None,
        reported_claim=None,
        methods_summary=None,
        limitations=None,
        public_justification=None,
        parse_ok=False,
        parse_error=error,
    )

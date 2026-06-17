"""Load and match pre-generated IBCS compliance reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = REPO_ROOT / "ibcs_compliance_report_openai.json"

_REPORT_INDEX: Dict[str, Dict[str, Any]] | None = None


def _normalize_filename(name: str) -> str:
    return Path(name).name.casefold()


def load_compliance_index(report_path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Build a case-insensitive filename -> report lookup."""
    global _REPORT_INDEX
    if _REPORT_INDEX is not None:
        return _REPORT_INDEX

    path = report_path or DEFAULT_REPORT_PATH
    if not path.is_file():
        _REPORT_INDEX = {}
        return _REPORT_INDEX

    raw = json.loads(path.read_text(encoding="utf-8"))
    index: Dict[str, Dict[str, Any]] = {}
    for entry in raw:
        key = _normalize_filename(entry.get("filename", ""))
        if key:
            index[key] = entry
    _REPORT_INDEX = index
    return _REPORT_INDEX


def lookup_compliance_report(filename: str) -> Optional[Dict[str, Any]]:
    return load_compliance_index().get(_normalize_filename(filename))


def sanitize_compliance_report(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Strip bulky audit fields before sending to the browser."""
    return {
        "filename": entry.get("filename"),
        "compliant": entry.get("compliant"),
        "noncompliance_score": entry.get("noncompliance_score"),
        "rule_checks": entry.get("rule_checks", {}),
        "violations": entry.get("violations", []),
        "charts_detected": entry.get("charts_detected", []),
        "final_explanation": entry.get("final_explanation", ""),
        "compliant_aggregation": entry.get("compliant_aggregation", {}),
    }


def match_compliance_chart(
    chart_id: int,
    charts_detected: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not charts_detected:
        return None

    target_id = f"chart_{chart_id + 1}"
    for chart in charts_detected:
        if chart.get("id") == target_id:
            return chart

    if 0 <= chart_id < len(charts_detected):
        return charts_detected[chart_id]

    return None


def enrich_chart_with_compliance(
    ml_chart: Dict[str, Any],
    charts_detected: List[Dict[str, Any]],
) -> Dict[str, Any]:
    chart_id = int(ml_chart.get("chart_id", 0))
    ml_chart["compliance_chart"] = match_compliance_chart(chart_id, charts_detected)
    return ml_chart

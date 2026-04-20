"""Persistent cross-run memory for financial analyses.

Stores per-company analysis history in a JSON file so that subsequent runs
can reference prior findings (trend comparisons, previously flagged risks, etc.).

Storage location: ~/.financial-analyst/memory.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_DIR = Path.home() / ".financial-analyst"
DEFAULT_MEMORY_FILE = DEFAULT_MEMORY_DIR / "memory.json"


def _normalize_company(name: str) -> str:
    """Normalize company name for use as a dict key."""
    return name.strip().lower().replace(" ", "_")


def load_memory(company: str, path: Path = DEFAULT_MEMORY_FILE) -> str:
    """Load prior analysis context for a company.

    Returns a formatted string for injection into agent prompts.
    Returns empty string if no prior data exists.
    """
    data = _read_store(path)
    key = _normalize_company(company)
    history = data.get(key, {}).get("analyses", [])

    if not history:
        return ""

    # Format the most recent analyses (up to last 3) for prompt injection
    parts = [f"## Prior Analysis History for {company}\n"]
    for entry in history[-3:]:
        parts.append(f"### Analysis from {entry['date']}")
        if entry.get("summary"):
            parts.append(f"**Summary**: {entry['summary']}")
        if entry.get("key_metrics"):
            parts.append("**Key metrics at the time**:")
            for metric, value in entry["key_metrics"].items():
                parts.append(f"  - {metric}: {value}")
        if entry.get("flags"):
            parts.append(f"**Flags/Risks noted**: {', '.join(entry['flags'])}")
        if entry.get("evaluation_score"):
            parts.append(f"**Quality score**: {entry['evaluation_score']}/5")
        parts.append("")

    return "\n".join(parts)


def save_memory(
    company: str,
    report: str,
    financial_data: list[dict],
    evaluation: dict | None = None,
    path: Path = DEFAULT_MEMORY_FILE,
) -> None:
    """Save analysis results to persistent memory for future reference."""
    data = _read_store(path)
    key = _normalize_company(company)

    if key not in data:
        data[key] = {"company_name": company, "analyses": []}

    entry = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "summary": _extract_summary(report),
        "key_metrics": _extract_key_metrics(financial_data),
        "flags": _extract_flags(report),
        "evaluation_score": evaluation.get("overall_score") if evaluation else None,
    }

    data[key]["analyses"].append(entry)

    # Keep only the last 10 analyses per company
    data[key]["analyses"] = data[key]["analyses"][-10:]

    _write_store(data, path)
    logger.info("Saved analysis to memory for %s (%d total)", company, len(data[key]["analyses"]))


def _extract_summary(report: str) -> str:
    """Extract the executive summary section from the report."""
    lines = report.split("\n")
    in_summary = False
    summary_lines = []

    for line in lines:
        if "executive summary" in line.lower():
            in_summary = True
            continue
        if in_summary:
            if line.startswith("##"):
                break
            if line.strip():
                summary_lines.append(line.strip())

    return " ".join(summary_lines[:3]) if summary_lines else report[:200]


def _extract_key_metrics(financial_data: list[dict]) -> dict[str, str]:
    """Extract a small set of key metrics for memory storage."""
    metrics = {}
    priority_metrics = {"revenue", "net_income", "p/e", "pe_ratio", "roe", "debt_to_equity", "ebitda"}

    for item in financial_data:
        metric_name = str(item.get("metric", "")).lower().replace(" ", "_")
        if any(p in metric_name for p in priority_metrics):
            metrics[item.get("metric", metric_name)] = f"{item.get('value')} {item.get('unit', '')}"

        if len(metrics) >= 6:
            break

    return metrics


def _extract_flags(report: str) -> list[str]:
    """Extract risk flags from the report's risk section."""
    lines = report.split("\n")
    in_risk = False
    flags = []

    for line in lines:
        if "risk" in line.lower() and line.startswith("#"):
            in_risk = True
            continue
        if in_risk:
            if line.startswith("##"):
                break
            stripped = line.strip().lstrip("- *")
            if stripped and len(stripped) > 10:
                flags.append(stripped[:100])

    return flags[:5]


def _read_store(path: Path) -> dict:
    """Read the memory store from disk."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read memory store: %s", e)
        return {}


def _write_store(data: dict, path: Path) -> None:
    """Write the memory store to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))

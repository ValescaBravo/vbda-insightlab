"""
===============================================================================
INSIGHTLAB — NARRATIVE ENGINE
===============================================================================
Narrative engine that translates statistical and ML analyses into clear,
business-ready stories.

Includes:
    • 5-Layer Interpretation Framework
    • STAR Method (Situation–Task–Analysis–Recommendation)
    • Insight Summary Framework (Insight–Evidence–Interpretation–Action–Risk)
    • Formatting helpers (explain, result)

Comments: English
Displayed text: English
===============================================================================
"""

from __future__ import annotations

from typing import Any, Dict

# Import only infrastructure components from core
from insightlab.core import (
    CONFIG,
    box,
    section,
    show_html,
    _silent,
    _stakeholder,
    _technical,
    _domain_phrase,
)


# =============================================================================
# 1. FIVE-LAYER INTERPRETATION FRAMEWORK
# =============================================================================

def interpretation_5layers(
    descriptive: str,
    statistical: str,
    behavioural: str,
    strategic: str,
    operational: str,
) -> str:
    """
    Five-layer framework for a complete interpretation.

    Structure:
        1. Descriptive  — what we observed
        2. Statistical  — how strong the evidence is
        3. Behavioural  — what real-world behaviours it may imply
        4. Strategic    — what it means for the business
        5. Operational  — what concrete actions to take next

    Args:
        descriptive: Description of what was observed.
        statistical: Statistical evidence and significance.
        behavioural: Behavioural implications in the real world.
        strategic: Strategic meaning for the business.
        operational: Operational recommendations and next steps.

    Returns:
        Rendered HTML string.
    """
    if _silent():
        return ""

    content = f"""
        <p><strong>1. Descriptive — What we observed:</strong><br>{descriptive}</p>
        <p><strong>2. Statistical — How strong the evidence is:</strong><br>{statistical}</p>
        <p><strong>3. Behavioural — Likely real-world behaviours:</strong><br>{behavioural}</p>
        <p><strong>4. Strategic — Business meaning:</strong><br>{strategic}</p>
        <p><strong>5. Operational — Recommended next actions:</strong><br>{operational}</p>
    """

    return box("business", "5-Layer Interpretation", content)


# =============================================================================
# 2. STAR METHOD
# =============================================================================

def star(
    situation: str,
    task: str,
    analysis: str,
    recommendation: str,
) -> str:
    """
    STAR method for executive communication.

    Structure:
        - Situation: context and circumstances
        - Task: objective or question
        - Analysis: findings and evidence
        - Recommendation: suggested action

    Args:
        situation: Description of the context.
        task: Objective of the analysis.
        analysis: Results and supporting evidence.
        recommendation: Recommended action.

    Returns:
        Rendered HTML string.
    """
    if _silent():
        return ""

    content = f"""
        <p><strong>Situation:</strong><br>{situation}</p>
        <p><strong>Task:</strong><br>{task}</p>
        <p><strong>Analysis:</strong><br>{analysis}</p>
        <p><strong>Recommendation:</strong><br>{recommendation}</p>
    """

    return box("insight", "STAR Summary", content)


# =============================================================================
# 3. INSIGHT SUMMARY (NARRATIVE)
# =============================================================================

def narrative(
    insight: str,
    evidence: str,
    interpretation: str,
    action: str,
    risk: str = "",
) -> str:
    """
    Editorial-style narrative for business insights.

    Structure:
        - Insight: key takeaway
        - Evidence: data supporting the insight
        - Interpretation: what it means
        - Action: what to do about it
        - Risk: warnings and caveats

    Args:
        insight: Main finding.
        evidence: Evidence supporting the finding.
        interpretation: Interpretation of the finding.
        action: Recommended action.
        risk: Risks and caveats (optional; if empty, uses the domain default).

    Returns:
        Rendered HTML string.
    """
    if _silent():
        return ""

    # Use the domain risk phrase if risk is empty
    risk_text = risk or _domain_phrase("risk")
    domain_footer = _domain_phrase("footer")

    content = f"""
        <p><strong>Insight:</strong><br>{insight}</p>
        <p><strong>Evidence:</strong><br>{evidence}</p>
        <p><strong>Interpretation:</strong><br>{interpretation}</p>
        <p><strong>Recommended Action:</strong><br>{action}</p>
    """

    # Include Risk only if there is content
    if risk_text:
        content += f"""
        <p><strong>Risk & Caveats:</strong><br>{risk_text}</p>
        """

    # Include domain footer if present
    if domain_footer:
        content += f"""
        <p style="opacity:0.7; font-size:0.9rem; margin-top:12px;">
            {domain_footer}
        </p>
        """

    return box("business", "Insight Summary", content)


def narrative_from_dict(info: Dict[str, Any]) -> str:
    """
    Convert a stats/ML dictionary into a narrative.

    Args:
        info: Dict with keys {insight, evidence, interpretation, action, risk}.

    Returns:
        Rendered HTML string.
    """
    return narrative(
        insight=info.get("insight", ""),
        evidence=info.get("evidence", ""),
        interpretation=info.get("interpretation", ""),
        action=info.get("action", ""),
        risk=info.get("risk", ""),
    )


# =============================================================================
# 4. FORMATTING HELPERS
# =============================================================================

def explain(name: str, technical_def: str, simple_def: str) -> str:
    """
    Return an explanation based on verbosity.

    Args:
        name: Concept name.
        technical_def: Technical definition.
        simple_def: Plain-English definition.

    Returns:
        Formatted explanation string.
    """
    if _silent():
        return ""

    if _technical():
        return f"<strong>{name}:</strong> {technical_def}"

    return f"<strong>{name}:</strong> {simple_def}"


def result(label: str, value: Any, unit: str = "") -> str:
    """
    Format a metric according to verbosity and precision settings.

    Args:
        label: Metric label.
        value: Numeric (or display) value.
        unit: Optional unit suffix.

    Returns:
        Formatted metric string.
    """
    if _silent():
        return ""

    # Try rounding if numeric
    try:
        val = round(float(value), CONFIG.precision)
    except (ValueError, TypeError):
        val = value

    return f"<strong>{label}:</strong> {val} {unit}".strip()


# =============================================================================
# 5. BACKWARD-COMPATIBILITY ALIAS
# =============================================================================

def layers(
    descriptive: str,
    statistical: str,
    behavioural: str,
    strategic: str,
    operational: str,
) -> str:
    """
    Alias for interpretation_5layers().

    Keeps backward compatibility with older code that calls layers().
    """
    return interpretation_5layers(
        descriptive, statistical, behavioural, strategic, operational
    )


# =============================================================================
# PUBLIC EXPORTS
# =============================================================================

__all__ = [
    # Narrative frameworks
    "interpretation_5layers",
    "star",
    "narrative",
    "narrative_from_dict",
    # Helpers
    "explain",
    "result",
    # Alias
    "layers",
]

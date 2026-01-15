"""
===============================================================================
INSIGHTLAB — STORY ENGINE FOR DATA PREPARATION (story_prep.py)
===============================================================================
Transforms prep.py metadata into clean, human-readable narrative blocks.

Inputs:
    - PrepResult (from Cleaner.to_result()), which exposes:
        • .metadata: dict of prep steps and their parameters
        • .quality: dict from get_data_quality_report()
        • .df: cleaned DataFrame (optional)
    - OR a dict:
        {"metadata": {...}, "quality": {...}}
    - OR a legacy metadata dict (backwards compatible)

Outputs:
    - Single HTML string (vb_render-compatible)
    - Insight–Evidence–Interpretation–Action–Risk boxes
    - Optional 5-Layer Interpretation summary

No emojis.
Narratives: English.
Comments: English.
===============================================================================
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from insightlab.core import (
    CONFIG,
    section,
    box,
    _silent,
    _stakeholder,
    _technical,
)

from insightlab.narrative.story import (
    narrative,
    narrative_from_dict,
    interpretation_5layers,
    explain,
)

# Try to use the centralised quality report from prep.py, if available
try:
    from insightlab.prep import get_data_quality_report  # type: ignore
    _HAS_QUALITY_REPORT = True
except Exception:  # pragma: no cover
    get_data_quality_report = None  # type: ignore
    _HAS_QUALITY_REPORT = False


# =============================================================================
# TYPES & INTERNAL UTILITIES
# =============================================================================

PrepLike = Union[Any, Dict[str, Any]]


def _fmt_pct(value: float) -> str:
    """Convert a proportion into a clean percentage string (0–1 → 0.0%)."""
    return f"{value * 100:.1f}%"


def _render_list(items: List[str]) -> str:
    """Render a simple list in HTML."""
    if not items:
        return "<ul></ul>"
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def _log_story_prep(step: str, payload: Dict[str, Any]) -> None:
    """
    Log a data-prep storytelling step into the global trace.

    This must never break the analytical flow; errors are swallowed.
    """
    trace = getattr(CONFIG, "trace", None)
    if trace is None:
        return
    try:
        trace.log(stage="story_prep", step=step, details=payload)
    except Exception:
        pass


def _normalise_prep_input(
    prep: PrepLike,
    df: Optional[pd.DataFrame],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Normalise input into (metadata, quality, df).

    Supports:
        • PrepResult-like object with .metadata, .quality, .df
        • dict with 'metadata' and 'quality'
        • legacy dict equivalent to 'metadata'
    """
    metadata: Dict[str, Any] = {}
    quality: Dict[str, Any] = {}
    df_final: Optional[pd.DataFrame] = df

    # Case 1: PrepResult-like
    if hasattr(prep, "metadata") and hasattr(prep, "quality"):
        metadata = getattr(prep, "metadata", {}) or {}
        quality = getattr(prep, "quality", {}) or {}
        if df_final is None and hasattr(prep, "df"):
            maybe_df = getattr(prep, "df")
            if isinstance(maybe_df, pd.DataFrame):
                df_final = maybe_df

    # Case 2: dict with metadata/quality
    elif isinstance(prep, dict):
        if "metadata" in prep or "quality" in prep:
            metadata = prep.get("metadata", {}) or {}
            quality = prep.get("quality", {}) or {}
        else:
            # Legacy dict = metadata directly
            metadata = prep or {}
            quality = {}

    else:
        raise ValueError(
            "prep must be a PrepResult-like object or a dict with 'metadata' "
            "and optionally 'quality'."
        )

    # Optional fallback: if quality is missing but df is available and helper exists
    if not quality and df_final is not None and _HAS_QUALITY_REPORT:
        try:
            quality = get_data_quality_report(df_final) or {}  # type: ignore[arg-type]
        except Exception:
            # Must not break the flow if the helper fails
            quality = quality or {}

    return metadata, quality, df_final


# =============================================================================
# HIGH-LEVEL PUBLIC API
# =============================================================================

def story_prep(
    prep: PrepLike,
    df: Optional[pd.DataFrame] = None,
    title: str = "Data preparation summary",
) -> str:
    """
    Generate a structured narrative for the data preparation phase.

    Args:
        prep:
            - PrepResult (from Cleaner.to_result())
            - dict with at least {"metadata": {...}, "quality": {...}}
            - legacy dict equivalent to metadata
        df:
            - Final cleaned DataFrame (optional).
              If not provided and prep is a PrepResult, uses prep.df.
        title:
            - Section title.

    Returns:
        HTML string containing the full narrative (safe to concatenate in reports).
    """
    if _silent():
        return ""

    metadata, quality, df_final = _normalise_prep_input(prep, df)

    blocks: List[str] = []

    # 1) Section title
    header_html = section(title) or ""
    blocks.append(header_html)

    # 2) Data quality overview
    overview_html = _story_overview(quality, df_final)
    if overview_html:
        blocks.append(overview_html)

    # 3) Transformation steps in a logical order
    blocks.extend(_story_transformations(metadata))

    # 4) Wrap-up / remaining risks
    conclusion_html = _story_conclusion(metadata, quality)
    if conclusion_html:
        blocks.append(conclusion_html)

    # 5) 5-Layer Interpretation (educational summary)
    five_layers_html = interpretation_5layers(
        descriptive=(
            "The dataset has been cleaned and preprocessed using consistent and "
            "transparent steps."
        ),
        statistical=(
            "Missing values, outliers, and structural issues were systematically "
            "addressed using reproducible rules."
        ),
        behavioural=(
            "Analysts and models can now rely on more stable, better-behaved input "
            "features, reducing surprises in downstream analysis."
        ),
        strategic=(
            "This preparation increases trust in any insights or models built on "
            "this dataset and reduces the risk of poor decisions driven by dirty data."
        ),
        operational=(
            "Use this preparation pipeline as a reusable template, and revisit it "
            "whenever upstream data definitions or business rules change."
        ),
    ) or ""

    if five_layers_html:
        blocks.append(five_layers_html)

    # 6) Concatenate all blocks
    html = "".join(b for b in blocks if b)

    _log_story_prep(
        "story_prep",
        {
            "n_steps": len(metadata),
            "has_quality": bool(quality),
            "has_df": df_final is not None,
        },
    )

    return html


def prep_summary(
    prep: PrepLike,
    df: Optional[pd.DataFrame] = None,
    title: str = "Data preparation summary",
) -> str:
    """
    Semantic alias of story_prep().

    This keeps naming flexibility for notebooks/templates while preserving
    a stable API surface.
    """
    return story_prep(prep=prep, df=df, title=title)


# =============================================================================
# OVERVIEW STORY
# =============================================================================

def _story_overview(
    quality: Dict[str, Any],
    df: Optional[pd.DataFrame],
) -> str:
    """
    Final data quality overview box.

    Prefer reading:
        quality["shape"], ["missing_pct"], ["duplicate_count"], ["type_summary"]
    with a soft fallback to df if information is missing.
    """
    if not quality and df is None:
        return ""

    shape = quality.get("shape")
    n_rows: Optional[int]
    n_cols: Optional[int]

    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        n_rows, n_cols = int(shape[0]), int(shape[1])
    elif df is not None:
        n_rows, n_cols = df.shape
    else:
        n_rows = n_cols = None

    missing_pct = quality.get("missing_pct", {})
    duplicate_count = quality.get("duplicate_count")
    if duplicate_count is None and df is not None:
        duplicate_count = int(df.duplicated().sum())

    type_summary = quality.get("type_summary", {})

    evidence_parts: List[str] = []

    if n_rows is not None and n_cols is not None:
        evidence_parts.append(f"Final dataset shape: {n_rows} rows × {n_cols} columns.")

    if duplicate_count is not None:
        evidence_parts.append(f"Duplicate rows after cleaning: {duplicate_count}.")

    if type_summary:
        evidence_parts.append(f"Variable types: {type_summary}.")

    if missing_pct:
        # Only show a small summary here; details belong in story_explore.story_missing
        avg_missing = float(sum(missing_pct.values())) / len(missing_pct)
        evidence_parts.append(
            f"Average missingness across columns: {_fmt_pct(avg_missing)}."
        )

    evidence = " ".join(evidence_parts) if evidence_parts else (
        "The dataset has been cleaned and is ready for analysis."
    )

    if _stakeholder():
        interpretation = (
            "The cleaned dataset is now standardised and consistent, so downstream "
            "analysis will be more reliable and easier to explain."
        )
    else:
        interpretation = (
            "Key data quality issues (missingness, duplicates, low-variance columns) "
            "have been addressed explicitly to stabilise statistical tests and models."
        )

    action = (
        "Use this dataset as the foundation for exploratory analysis, statistical "
        "testing and machine learning."
    )

    risk = (
        "If upstream data collection processes remain unchanged, new raw data may "
        "reintroduce similar quality issues. Monitor data quality over time."
    )

    return narrative_from_dict(
        {
            "insight": "Data quality overview.",
            "evidence": evidence,
            "interpretation": interpretation,
            "action": action,
            "risk": risk,
        }
    ) or ""


# =============================================================================
# TRANSFORMATION STORIES
# =============================================================================

def _story_transformations(metadata: Dict[str, Any]) -> List[str]:
    """
    Build narratives for each relevant transformation applied.

    Only adds a block when there is something meaningful to report.
    """
    if not metadata:
        return []

    blocks: List[str] = []

    if "rename" in metadata:
        block = _story_rename(metadata["rename"])
        if block:
            blocks.append(block)

    if "duplicates" in metadata:
        block = _story_duplicates(metadata["duplicates"])
        if block:
            blocks.append(block)

    if "missing" in metadata:
        block = _story_missing(metadata["missing"])
        if block:
            blocks.append(block)

    if "missing_flags" in metadata:
        block = _story_missing_flags(metadata["missing_flags"])
        if block:
            blocks.append(block)

    if "outliers" in metadata:
        block = _story_outliers(metadata["outliers"])
        if block:
            blocks.append(block)

    if "outlier_flags" in metadata:
        block = _story_outlier_flags(metadata["outlier_flags"])
        if block:
            blocks.append(block)

    if "scale" in metadata:
        block = _story_scale(metadata["scale"])
        if block:
            blocks.append(block)

    if "log_transform" in metadata:
        block = _story_log(metadata["log_transform"])
        if block:
            blocks.append(block)

    if "power_transform" in metadata:
        block = _story_power(metadata["power_transform"])
        if block:
            blocks.append(block)

    if any(k in metadata for k in ("onehot", "ordinal", "target_encode")):
        block = _story_encoding(metadata)
        if block:
            blocks.append(block)

    if "date_features" in metadata:
        block = _story_dates(metadata["date_features"])
        if block:
            blocks.append(block)

    if (
        "discretize_equal_width" in metadata
        or "discretize_equal_freq" in metadata
        or "discretize" in metadata
    ):
        block = _story_discretization(metadata)
        if block:
            blocks.append(block)

    if "remove_constant_columns" in metadata or "remove_high_missing_cols" in metadata:
        block = _story_column_filters(metadata)
        if block:
            blocks.append(block)

    return blocks


def _story_rename(meta: Dict[str, Any]) -> str:
    renamed_count = int(meta.get("renamed_count", 0))
    if renamed_count == 0 and not _technical():
        return ""

    evidence = f"{renamed_count} columns were renamed to a clean, consistent, machine-friendly format."

    return narrative_from_dict(
        {
            "insight": "Column names standardised.",
            "evidence": evidence,
            "interpretation": (
                "Consistent naming conventions make analysis scripts easier to write, "
                "maintain and reuse."
            ),
            "action": "Use the new column names in all downstream notebooks and reports.",
            "risk": (
                "If business stakeholders still use old names, mismatches can cause "
                "confusion. Consider keeping a mapping table."
            ),
        }
    ) or ""


def _story_duplicates(meta: Dict[str, Any]) -> str:
    removed = int(meta.get("duplicates_removed", 0))
    if removed == 0 and not _technical():
        return ""

    return narrative_from_dict(
        {
            "insight": "Duplicate rows removed.",
            "evidence": (
                f"{removed} duplicate rows were removed to avoid counting the same "
                "observation multiple times."
            ),
            "interpretation": (
                "Removing duplicates prevents data leakage and inflated sample sizes, "
                "which can distort statistics and models."
            ),
            "action": (
                "Use the deduplicated dataset as the canonical source for analysis. "
                "Document known reasons for duplication (if any)."
            ),
            "risk": (
                "If some duplicates were legitimate repeated events, aggregating rather "
                "than dropping might be more appropriate."
            ),
        }
    ) or ""


def _story_missing(meta: Dict[str, Any]) -> str:
    filled = int(meta.get("missing_filled", 0))
    before = meta.get("missing_before")
    after = meta.get("missing_after")

    if filled == 0 and not _technical():
        return ""

    if _stakeholder():
        interpretation = (
            "Missing values were handled systematically, reducing the risk of biased "
            "results and making trends more reliable."
        )
    else:
        interpretation = (
            "Missing values were imputed according to variable type, improving model "
            "stability and avoiding ad-hoc row drops."
        )

    evidence_parts: List[str] = []
    if before is not None and after is not None:
        evidence_parts.append(
            f"Total missing values before cleaning: {before}, after: {after}."
        )
    evidence_parts.append(f"Values imputed or filled: {filled}.")
    evidence = " ".join(evidence_parts)

    return narrative_from_dict(
        {
            "insight": "Missing values treated.",
            "evidence": evidence,
            "interpretation": interpretation,
            "action": (
                "Document the imputation strategies used for key business metrics so "
                "other analysts can reproduce the same pipeline."
            ),
            "risk": (
                "Imputed values may hide systematic issues in data collection. Monitor "
                "future data for recurring missingness patterns."
            ),
        }
    ) or ""


def _story_missing_flags(meta: Dict[str, Any]) -> str:
    created = meta.get("missing_indicators_created", []) or []
    if not created:
        return ""

    evidence = (
        "Missing-value indicator variables were created for the following features:"
        f"{_render_list(created)}"
    )

    return narrative_from_dict(
        {
            "insight": "Missingness indicators created.",
            "evidence": evidence,
            "interpretation": (
                "These flags allow models to learn whether the fact that data is "
                "missing carries predictive information."
            ),
            "action": (
                "Include these indicators as features in predictive models, especially "
                "for churn, risk, or fraud use cases."
            ),
            "risk": (
                "If missingness is purely random, indicators may add noise. Validate "
                "their impact on model performance."
            ),
        }
    ) or ""


def _story_outliers(meta: Dict[str, Any]) -> str:
    method = meta.get("method", "iqr")
    cols = meta.get("columns", [])
    before = meta.get("outliers_before", {}) or {}
    after = meta.get("outliers_after", {}) or {}

    n_cols = len(cols) if cols else len(before)
    total_before = sum(before.values()) if before else None
    total_after = sum(after.values()) if after else None

    if n_cols == 0 and not _technical():
        return ""

    if _stakeholder():
        interpretation = (
            "Extreme values were capped so that very rare spikes no longer dominate "
            "average behaviour."
        )
    else:
        interpretation = (
            f"Outliers were treated using the '{method}' rule, reducing leverage points "
            "that could distort parameter estimates and model training."
        )

    evidence_parts: List[str] = []
    evidence_parts.append(
        f"Outlier treatment applied to {n_cols} numeric variables."
    )
    if total_before is not None and total_after is not None:
        evidence_parts.append(
            f"Total extreme values before: {total_before}, after: {total_after}."
        )

    evidence = " ".join(evidence_parts)

    return narrative_from_dict(
        {
            "insight": "Outliers treated.",
            "evidence": evidence,
            "interpretation": interpretation,
            "action": (
                "Review business rules around extreme values to decide whether they "
                "represent genuine rare events or data-quality problems."
            ),
            "risk": (
                "Over-aggressive capping can hide meaningful rare events; keep access "
                "to the original raw data for audit and deep dives."
            ),
        }
    ) or ""


def _story_outlier_flags(meta: Dict[str, Any]) -> str:
    created = meta.get("outlier_flags_created", []) or []
    if not created:
        return ""

    return narrative_from_dict(
        {
            "insight": "Outlier flags added.",
            "evidence": (
                "Binary flags were created to mark rows where original values exceeded "
                "predefined outlier thresholds."
            ),
            "interpretation": (
                "Models and diagnostic plots can now distinguish between typical "
                "observations and extreme cases."
            ),
            "action": (
                "Use outlier flags to analyse the behaviour and impact of extreme "
                "observations separately."
            ),
            "risk": (
                "If thresholds were poorly chosen, flags may be too frequent or too "
                "rare to be informative."
            ),
        }
    ) or ""


def _story_scale(meta: Dict[str, Any]) -> str:
    method = meta.get("method", "standard")
    scaled_cols = meta.get("scaled_columns", []) or []
    if not scaled_cols and not _technical():
        return ""

    evidence = (
        f"Numeric variables were scaled using the {method} method. "
        "Scaled features include:"
        f"{_render_list(scaled_cols)}"
    )

    return narrative_from_dict(
        {
            "insight": "Numeric variables scaled.",
            "evidence": evidence,
            "interpretation": (
                "Scaling ensures that variables measured in different units contribute "
                "appropriately to distance-based models and regularised regressions."
            ),
            "action": (
                "Reuse the same scaler object when scoring new data to maintain "
                "consistency."
            ),
            "risk": (
                "If new data arrive with very different ranges, the fitted scaler may "
                "need to be retrained."
            ),
        }
    ) or ""


def _story_log(meta: Dict[str, Any]) -> str:
    cols = meta.get("log_transformed", []) or meta.get("log_transformed_cols", []) or []
    if not cols:
        return ""

    return narrative_from_dict(
        {
            "insight": "Log transformation applied.",
            "evidence": (
                "Log transformation was applied to the following skewed variables:"
                f"{_render_list(cols)}"
            ),
            "interpretation": (
                "Transforming right-skewed variables reduces the impact of extreme "
                "values and can improve linear model assumptions."
            ),
            "action": (
                "Interpret coefficients and effects on the log scale, or back-transform "
                "when presenting results."
            ),
            "risk": (
                "Log transformation changes the scale of interpretation; ensure "
                "stakeholders understand the transformed units."
            ),
        }
    ) or ""


def _story_power(meta: Dict[str, Any]) -> str:
    cols = meta.get("power_transformed", []) or []
    if not cols:
        return ""

    return narrative_from_dict(
        {
            "insight": "Power transformation applied.",
            "evidence": (
                "A power transform (such as Yeo–Johnson or Box–Cox) was applied to:"
                f"{_render_list(cols)}"
            ),
            "interpretation": (
                "Power transforms aim to stabilise variance and make distributions "
                "more symmetric, improving the validity of parametric tests."
            ),
            "action": (
                "Keep track of the specific transformation used so the same mapping "
                "can be applied to future data."
            ),
            "risk": (
                "Over-transformation can make interpretation less intuitive. Reserve "
                "these methods for variables where normality matters."
            ),
        }
    ) or ""


def _story_encoding(metadata: Dict[str, Any]) -> str:
    onehot_cols = (metadata.get("onehot", {}) or {}).get("onehot_encoded", []) or []
    ordinal_cols = (metadata.get("ordinal", {}) or {}).get("ordinal_encoded", []) or []
    target_cols = (metadata.get("target_encode", {}) or {}).get("target_encoded", []) or []

    if not (onehot_cols or ordinal_cols or target_cols):
        return ""

    parts: List[str] = []
    if onehot_cols:
        parts.append(
            "One-hot encoding was applied to the following categorical variables:"
            f"{_render_list(onehot_cols)}"
        )
    if ordinal_cols:
        parts.append(
            "Ordinal encoding was used for ordered categories:"
            f"{_render_list(ordinal_cols)}"
        )
    if target_cols:
        parts.append(
            "Target encoding was applied to high-cardinality categorical variables:"
            f"{_render_list(target_cols)}"
        )

    evidence = "".join(parts)

    return narrative_from_dict(
        {
            "insight": "Categorical variables encoded.",
            "evidence": evidence,
            "interpretation": (
                "Encoding schemes were chosen to balance model performance and "
                "interpretability, especially for high-cardinality features."
            ),
            "action": (
                "Ensure that the same encoders are reused at prediction time to avoid "
                "category mismatches."
            ),
            "risk": (
                "Target encoding can leak information if not properly regularised or "
                "if applied before cross-validation."
            ),
        }
    ) or ""


def _story_dates(meta: Dict[str, Any]) -> str:
    features = meta.get("date_features_added", []) or meta.get("date_features", []) or []
    if not features:
        return ""

    return narrative_from_dict(
        {
            "insight": "Datetime features engineered.",
            "evidence": (
                "New features derived from datetime columns were added:"
                f"{_render_list(features)}"
            ),
            "interpretation": (
                "These features capture temporal patterns such as seasonality, "
                "day-of-week effects, and record ageing."
            ),
            "action": (
                "Use these derived features in models that need to capture time-related "
                "behaviour."
            ),
            "risk": (
                "If time zones or calendars change, date-derived features may need to "
                "be recalibrated."
            ),
        }
    ) or ""


def _story_discretization(metadata: Dict[str, Any]) -> str:
    # Support both the newer scheme (equal_width / equal_freq) and legacy schema
    meta_width = metadata.get("discretize_equal_width", {}) or metadata.get("discretize", {})
    meta_freq = metadata.get("discretize_equal_freq", {})

    width_bins = meta_width.get("equal_width_bins", {}) or {}
    freq_bins = meta_freq.get("equal_freq_bins", {}) or {}

    if not width_bins and not freq_bins:
        return ""

    items: List[str] = []
    if width_bins:
        items.append(
            "Equal-width bins created for:"
            + "<ul>"
            + "".join(f"<li>{k}: {v} bins</li>" for k, v in width_bins.items())
            + "</ul>"
        )
    if freq_bins:
        items.append(
            "Equal-frequency bins created for:"
            + "<ul>"
            + "".join(f"<li>{k}: {v} bins</li>" for k, v in freq_bins.items())
            + "</ul>"
        )

    evidence = "".join(items)

    return narrative_from_dict(
        {
            "insight": "Continuous variables discretised.",
            "evidence": evidence,
            "interpretation": (
                "Discretisation groups continuous values into interpretable bands, "
                "which can simplify rules and reports."
            ),
            "action": (
                "Use binned versions for segmentation and business rules; retain raw "
                "values for modelling when appropriate."
            ),
            "risk": (
                "Too few bins can hide important variation; too many bins can create "
                "sparse groups. Review binning choices with domain experts."
            ),
        }
    ) or ""


def _story_column_filters(metadata: Dict[str, Any]) -> str:
    const_meta = metadata.get("remove_constant_columns", {}) or {}
    high_missing_meta = metadata.get("remove_high_missing_cols", {}) or {}

    const_cols = const_meta.get("removed_columns", []) or []
    high_missing_cols = high_missing_meta.get("removed_columns", []) or []

    if not const_cols and not high_missing_cols:
        return ""

    parts: List[str] = []
    if const_cols:
        parts.append(
            "Constant (zero-variance) columns were removed:"
            f"{_render_list(const_cols)}"
        )
    if high_missing_cols:
        parts.append(
            "Columns with very high missingness were removed:"
            f"{_render_list(high_missing_cols)}"
        )

    evidence = "".join(parts)

    return narrative_from_dict(
        {
            "insight": "Low-information columns removed.",
            "evidence": evidence,
            "interpretation": (
                "Removing constant and highly missing columns reduces noise and "
                "simplifies modelling without losing meaningful information."
            ),
            "action": (
                "Keep a record of removed columns so they can be reinstated if business "
                "definitions change."
            ),
            "risk": (
                "If thresholds were too aggressive, some partially informative columns "
                "may have been dropped."
            ),
        }
    ) or ""


# =============================================================================
# CONCLUSION STORY
# =============================================================================

def _story_conclusion(
    metadata: Dict[str, Any],
    quality: Dict[str, Any],
) -> str:
    steps = list(metadata.keys())
    n_steps = len(steps)

    if _stakeholder():
        interpretation = (
            "The dataset is now in an analysis-ready state, with consistent names, "
            "treated missing values and controlled outliers."
        )
    else:
        interpretation = (
            "Pre-processing steps (renaming, duplicate removal, imputation, "
            "outlier treatment, scaling, encoding and column filtering) have been "
            "applied in a documented order, improving reproducibility and auditability."
        )

    evidence = (
        f"A total of {n_steps} preparation steps were applied."
        f"{' Steps: ' + ', '.join(steps) if steps else ''}"
    )

    action = (
        "Use this preparation pipeline as the baseline for future datasets to keep "
        "analyses consistent over time."
    )

    risk = (
        "If upstream business rules or data sources change, the preparation pipeline "
        "will need to be revisited to remain valid."
    )

    return narrative_from_dict(
        {
            "insight": "Data preparation completed.",
            "evidence": evidence,
            "interpretation": interpretation,
            "action": action,
            "risk": risk,
        }
    ) or ""


__all__ = [
    "story_prep",
    "prep_summary",
]

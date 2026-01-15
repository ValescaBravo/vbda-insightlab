"""
===============================================================================
INSIGHTLAB — EXPLORATION STORYTELLING MODULE
===============================================================================
Transforms exploratory data analysis output into structured business
narratives using the InsightLab storytelling engine.

Includes narratives for:
    • Dataset overview
    • Variable types (with advanced detection)
    • Missing values
    • Duplicates
    • Descriptive statistics
    • Skewness & kurtosis
    • Outliers
    • Correlation matrix
    • Data quality report
    • Cardinality issues
    • Constant/ID column warnings

This module contains NO charts — only narrative logic.
Integrates with prep.py for advanced type detection and quality metrics.
Optionally logs steps into CONFIG.trace (stage="explore").

Comments: Spanish
Output: English

Version: 3.0 (Trace-ready + domain-aware thresholds)
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from insightlab.core import (
    CONFIG,
    math_insight,
    _silent,
    _stakeholder,
    _technical,
    _domain_phrase,
    box,
    section,
    show_html,
)

# Story engine (narrative blocks)
from insightlab.narrative.story import narrative
# (If your package layout is slightly different, this variant also works:)
# from .story import narrative

# Prep imports (type detection + quality) — opcionales
try:
    from insightlab.prep import (
        detect_types,
        get_data_quality_report,
    )
    PREP_AVAILABLE = True
except ImportError:
    PREP_AVAILABLE = False

# =============================================================================
# HELPERS
# =============================================================================

def _log_explore_step(step: str, details: Dict[str, Any]) -> None:
    """
    Registrar paso de exploración en el trace global si existe.

    Nunca debe romper el flujo analítico.
    """
    trace = getattr(CONFIG, "trace", None)
    if trace is None:
        return
    try:
        trace.log(stage="explore", step=step, details=details)
    except Exception:
        # El trace nunca debe romper el análisis.
        pass


def _get_explore_thresholds(overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Umbrales estándar de EDA, con posibilidad de override por dominio
    y/o por llamada específica.

    Dominio:
        Si CONFIG.domain_settings[CONFIG.domain]["explore_thresholds"] existe,
        se usa para actualizar los valores base.
    """
    base: Dict[str, float] = {
        # Missing
        "missing_medium": 0.05,
        "missing_high": 0.50,
        # Distribución
        "skew_moderate": 0.5,
        "skew_severe": 1.0,
        "kurtosis_heavy": 3.0,
        # Outliers (proporción de filas)
        "outlier_moderate_pct": 1.0,
        "outlier_severe_pct": 5.0,
        # Correlación
        "corr_strong": 0.5,
        "corr_very_high": 0.9,
    }

    # Overrides de dominio (opcionales)
    try:
        domain_preset = CONFIG.domain_settings.get(CONFIG.domain, {})  # type: ignore[attr-defined]
        explore_overrides = domain_preset.get("explore_thresholds", {}) or {}
        base.update(explore_overrides)
    except Exception:
        # Si no existe domain_settings, seguimos con base.
        pass

    # Overrides específicos pasados por parámetro
    if overrides:
        base.update(overrides)

    return base


def _get_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Obtener tipos de variables usando prep.py si está disponible,
    sino usar detección básica.
    """
    if PREP_AVAILABLE:
        return detect_types(df)

    # Fallback: detección básica
    return {
        "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime": df.select_dtypes(include=["datetime"]).columns.tolist(),
        "binary": [],
        "constant": [],
        "id_like": [],
        "high_cardinality": [],
    }


# =============================================================================
# 1. DATASET OVERVIEW STORY
# =============================================================================

def story_overview(df: pd.DataFrame) -> str:
    """
    Descripción de alto nivel de la estructura del dataset.

    Args:
        df: DataFrame a analizar

    Returns:
        HTML string con narrativa
    """

    if _silent():
        return ""

    n_rows, n_cols = df.shape
    types = _get_types(df)

    numeric_count = len(types["numeric"])
    categorical_count = len(types["categorical"])
    datetime_count = len(types["datetime"])

    # Calcular memoria
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2

    section("Dataset Overview")

    content = f"""
        <p><strong>Dimensions:</strong> {n_rows:,} rows × {n_cols} columns</p>
        <p><strong>Memory:</strong> {memory_mb:.2f} MB</p>
        <br>
        <p><strong>Variable Types:</strong></p>
        <ul>
            <li>Numerical: <strong>{numeric_count}</strong></li>
            <li>Categorical: <strong>{categorical_count}</strong></li>
            <li>Datetime: <strong>{datetime_count}</strong></li>
        </ul>
    """

    # Advertencias si hay columnas problemáticas
    warnings = []
    if types["constant"]:
        warnings.append(f"{len(types['constant'])} constant columns (zero variance)")
    if types["id_like"]:
        warnings.append(f"{len(types['id_like'])} ID-like columns (high uniqueness)")
    if types["high_cardinality"]:
        warnings.append(f"{len(types['high_cardinality'])} high-cardinality columns (>50 unique values)")

    if warnings:
        content += """
        <br>
        <p><strong>⚠️ Warnings:</strong></p>
        <ul>
        """
        for w in warnings:
            content += f"<li>{w}</li>"
        content += "</ul>"

    _log_explore_step(
        "story_overview",
        {
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "numeric_count": int(numeric_count),
            "categorical_count": int(categorical_count),
            "datetime_count": int(datetime_count),
            "memory_mb": float(memory_mb),
            "constant_cols": list(types["constant"]),
            "id_like_cols": list(types["id_like"]),
            "high_cardinality_cols": list(types["high_cardinality"]),
        },
    )

    return box("insight", "Structure", content)


# =============================================================================
# 2. VARIABLE TYPES STORY (ENHANCED)
# =============================================================================

def story_variable_types(df: pd.DataFrame) -> str:
    """
    Resumen detallado de tipos de variables detectados.
    Usa detección avanzada de prep.py si está disponible.
    """

    if _silent():
        return ""

    types = _get_types(df)

    section("Variable Type Summary")

    content = "<p><strong>Detected Variable Types:</strong></p>"

    # Numerical
    if types["numeric"]:
        numeric_names = ", ".join(types["numeric"][:10])
        if len(types["numeric"]) > 10:
            numeric_names += f", ... ({len(types['numeric']) - 10} more)"
        content += f"<p>📊 <strong>Numerical ({len(types['numeric'])}):</strong> {numeric_names}</p>"

    # Categorical
    if types["categorical"]:
        cat_names = ", ".join(types["categorical"][:10])
        if len(types["categorical"]) > 10:
            cat_names += f", ... ({len(types['categorical']) - 10} more)"
        content += f"<p>📝 <strong>Categorical ({len(types['categorical'])}):</strong> {cat_names}</p>"

    # Datetime
    if types["datetime"]:
        dt_names = ", ".join(types["datetime"])
        content += f"<p>📅 <strong>Datetime ({len(types['datetime'])}):</strong> {dt_names}</p>"

    # Binary
    if types["binary"]:
        binary_names = ", ".join(types["binary"])
        content += f"<p>🔀 <strong>Binary ({len(types['binary'])}):</strong> {binary_names}</p>"

    # Warnings for problematic types
    if types["constant"]:
        const_names = ", ".join(types["constant"])
        content += f"<p>⚠️ <strong>Constant ({len(types['constant'])}):</strong> {const_names}</p>"
        content += "<p><em>Consider removing constant columns (zero variance).</em></p>"

    if types["id_like"]:
        id_names = ", ".join(types["id_like"])
        content += f"<p>🆔 <strong>ID-like ({len(types['id_like'])}):</strong> {id_names}</p>"
        content += "<p><em>High uniqueness suggests these may be identifiers.</em></p>"

    if types["high_cardinality"]:
        hc_names = ", ".join(types["high_cardinality"])
        content += f"<p>🎯 <strong>High Cardinality ({len(types['high_cardinality'])}):</strong> {hc_names}</p>"
        content += "<p><em>Consider grouping or target encoding for ML models.</em></p>"

    _log_explore_step(
        "story_variable_types",
        {
            "numeric": list(types["numeric"]),
            "categorical": list(types["categorical"]),
            "datetime": list(types["datetime"]),
            "binary": list(types["binary"]),
            "constant": list(types["constant"]),
            "id_like": list(types["id_like"]),
            "high_cardinality": list(types["high_cardinality"]),
        },
    )

    return box("neutral", "Variable Classification", content, icon="🔍")


# =============================================================================
# 3. MISSING VALUES STORY
# =============================================================================

def story_missing(df: pd.DataFrame, threshold: float = 0.05) -> str:
    """
    Narrativa sobre valores faltantes.

    Args:
        df: DataFrame a analizar
        threshold: umbral para destacar columnas (default 5%)

    Returns:
        HTML string con narrativa
    """

    if _silent():
        return ""

    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]

    section("Missing Data Analysis")

    if missing.empty:
        _log_explore_step(
            "story_missing",
            {"n_vars_with_missing": 0, "threshold_medium": float(threshold)},
        )
        return box(
            "insight",
            "Complete Dataset",
            "This dataset has no missing values. All observations are complete.",
            icon="✅",
        )

    # Umbrales (permite override a nivel de dominio)
    thresholds = _get_explore_thresholds({"missing_medium": threshold})
    med_thr = thresholds["missing_medium"]
    high_thr = thresholds["missing_high"]

    # Clasificar columnas por severidad
    high_missing = missing[missing >= high_thr]
    medium_missing = missing[(missing >= med_thr) & (missing < high_thr)]
    low_missing = missing[missing < med_thr]

    # Construir evidencia
    evidence = ""

    if not high_missing.empty:
        items = "<br>".join([f"{col}: <strong>{pct*100:.1f}%</strong>" for col, pct in high_missing.items()])
        evidence += f"<p>🚨 <strong>High Missing (≥{high_thr*100:.0f}%):</strong><br>{items}</p>"

    if not medium_missing.empty:
        items = "<br>".join([f"{col}: <strong>{pct*100:.1f}%</strong>" for col, pct in medium_missing.items()])
        evidence += f"<p>⚠️ <strong>Medium Missing ({med_thr*100:.0f}%-{high_thr*100:.0f}%):</strong><br>{items}</p>"

    if not low_missing.empty:
        items = "<br>".join([f"{col}: <strong>{pct*100:.1f}%</strong>" for col, pct in low_missing.head(5).items()])
        if len(low_missing) > 5:
            items += f"<br><em>... and {len(low_missing) - 5} more variables</em>"
        evidence += f"<p>ℹ️ <strong>Low Missing (<{med_thr*100:.0f}%):</strong><br>{items}</p>"

    # Insight adaptativo
    if not high_missing.empty:
        insight = f"{len(high_missing)} variables have severe missing data (≥{high_thr*100:.0f}%)."
        interpretation = (
            "High missingness may signal data collection issues or irrelevant features. "
            "Consider dropping these variables unless they carry critical information."
        )
        action = "Investigate root cause. Drop if systematic. Impute carefully if random."
    elif not medium_missing.empty:
        insight = f"{len(medium_missing)} variables have moderate missing data ({med_thr*100:.0f}%-{high_thr*100:.0f}%)."
        interpretation = (
            "Moderate missingness requires careful treatment to avoid bias in downstream analysis."
        )
        action = "Use imputation strategies (median, mode, model-based) or create missing indicators."
    else:
        insight = f"Missing data detected in {len(missing)} variables, all below {med_thr*100:.0f}%."
        interpretation = "Low levels of missingness are manageable and unlikely to bias results significantly."
        action = "Simple imputation (median/mode) should suffice for most use cases."

    risk = (
        "Missing data is rarely random (MCAR). Investigate whether missingness correlates "
        "with outcomes or other variables (MAR, MNAR)."
    )

    _log_explore_step(
        "story_missing",
        {
            "n_vars_with_missing": int(len(missing)),
            "n_high_missing": int(len(high_missing)),
            "n_medium_missing": int(len(medium_missing)),
            "n_low_missing": int(len(low_missing)),
            "threshold_medium": float(med_thr),
            "threshold_high": float(high_thr),
        },
    )

    return narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )


# =============================================================================
# 4. DUPLICATE STORY
# =============================================================================

def story_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> str:
    """
    Narrativa sobre filas duplicadas.

    Args:
        df: DataFrame a analizar
        subset: columnas a considerar para duplicados (None = todas)

    Returns:
        HTML string con narrativa
    """

    if _silent():
        return ""

    n_duplicates = df.duplicated(subset=subset).sum()
    n_total = len(df)
    pct_duplicates = n_duplicates / n_total * 100 if n_total > 0 else 0.0

    section("Duplicate Detection")

    if n_duplicates == 0:
        _log_explore_step(
            "story_duplicates",
            {"n_duplicates": 0, "n_rows": int(n_total), "pct_duplicates": 0.0},
        )
        return box(
            "insight",
            "No Duplicates",
            "This dataset contains no duplicate rows. All observations are unique.",
            icon="✅",
        )

    # Narrativa según severidad
    if pct_duplicates >= 10:
        severity = "High"
        icon = "🚨"
        interpretation = (
            "A significant proportion of rows are duplicates. This could indicate: "
            "(1) data collection errors, (2) repeated measurements, or (3) grain mismatch."
        )
        action = "Investigate why duplicates exist. Drop if errors. Aggregate if legitimate repeats."
    elif pct_duplicates >= 1:
        severity = "Moderate"
        icon = "⚠️"
        interpretation = (
            "Moderate duplication detected. Common in real-world datasets but should be addressed."
        )
        action = "Review duplicate patterns. Drop if confirmed as errors."
    else:
        severity = "Low"
        icon = "ℹ️"
        interpretation = (
            "Minimal duplication present. Likely noise or edge cases."
        )
        action = "Safe to drop duplicates without deep investigation."

    insight = f"{n_duplicates:,} duplicate rows detected ({pct_duplicates:.2f}% of dataset)."
    evidence = f"{icon} <strong>{severity} duplication level:</strong> {n_duplicates:,} / {n_total:,} rows"
    risk = "Duplicates can inflate sample size and distort statistical tests. Always check before analysis."

    _log_explore_step(
        "story_duplicates",
        {
            "n_duplicates": int(n_duplicates),
            "n_rows": int(n_total),
            "pct_duplicates": float(pct_duplicates),
            "subset": list(subset) if subset is not None else None,
        },
    )

    return narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )


# =============================================================================
# 5. DESCRIPTIVE STATISTICS STORY
# =============================================================================

def story_descriptive(df: pd.DataFrame, cols: Optional[List[str]] = None) -> str:
    """
    Narrativa sobre estadísticas descriptivas de variables numéricas.
    """

    if _silent():
        return ""

    if cols is None:
        types = _get_types(df)
        cols = types["numeric"]

    if not cols:
        _log_explore_step(
            "story_descriptive",
            {"n_numeric_cols": 0, "columns": []},
        )
        return box(
            "neutral",
            "No Numerical Variables",
            "No numerical variables found for descriptive statistics.",
        )

    stats = df[cols].describe().T

    section("Descriptive Statistics")

    # Construir tabla HTML
    items = []
    for col in stats.index:
        row = stats.loc[col]
        items.append(
            f"<strong>{col}</strong>: "
            f"mean={row['mean']:.2f}, "
            f"std={row['std']:.2f}, "
            f"min={row['min']:.2f}, "
            f"max={row['max']:.2f}"
        )

    evidence = "<br>".join(items)

    # Detectar variables con alta variabilidad (coeficiente de variación)
    cv = (stats["std"] / stats["mean"].abs()).replace([np.inf, -np.inf], np.nan)
    high_cv = cv[cv > 1.0].sort_values(ascending=False)

    interpretation = "Summary statistics provide initial understanding of central tendency and dispersion."

    if not high_cv.empty:
        interpretation += (
            f" {len(high_cv)} variables show high variability (CV > 1.0): "
            f"{', '.join(high_cv.index[:3].tolist())}. Consider scaling or transformation."
        )

    _log_explore_step(
        "story_descriptive",
        {
            "n_numeric_cols": int(len(cols)),
            "columns": list(cols),
            "n_high_cv": int(len(high_cv)),
        },
    )

    return box("insight", "Key Distributions", evidence)


# =============================================================================
# 6. DISTRIBUTION SHAPE STORY
# =============================================================================

def story_shape(df: pd.DataFrame, cols: Optional[List[str]] = None) -> str:
    """
    Narrativa sobre forma de distribuciones (skewness y kurtosis).
    """

    if _silent():
        return ""

    if cols is None:
        types = _get_types(df)
        cols = types["numeric"]

    if not cols:
        _log_explore_step(
            "story_shape",
            {"n_cols": 0},
        )
        return ""

    skew = df[cols].skew()
    kurt = df[cols].kurtosis()

    thresholds = _get_explore_thresholds()
    skew_mod = thresholds["skew_moderate"]
    skew_severe = thresholds["skew_severe"]
    kurt_heavy = thresholds["kurtosis_heavy"]

    section("Distribution Shape Analysis", icon="📉")

    # Clasificar variables
    highly_skewed = skew[skew.abs() > skew_severe]
    moderately_skewed = skew[(skew.abs() > skew_mod) & (skew.abs() <= skew_severe)]

    heavy_tailed = kurt[kurt > kurt_heavy]

    # Construir evidencia
    evidence = "<p><strong>Skewness Summary:</strong></p>"

    if not highly_skewed.empty:
        items = "<br>".join([f"{col}: <strong>{val:.2f}</strong>" for col, val in highly_skewed.items()])
        evidence += (
            f"<p>🚨 Highly skewed (|γ₁| > {skew_severe:.1f}):<br>{items}</p>"
        )

    if not moderately_skewed.empty:
        items = "<br>".join([f"{col}: <strong>{val:.2f}</strong>" for col, val in moderately_skewed.items()])
        evidence += (
            f"<p>⚠️ Moderately skewed ({skew_mod:.1f} < |γ₁| ≤ {skew_severe:.1f}):<br>{items}</p>"
        )

    evidence += "<br><p><strong>Kurtosis Summary:</strong></p>"

    if not heavy_tailed.empty:
        items = "<br>".join([f"{col}: <strong>{val:.2f}</strong>" for col, val in heavy_tailed.items()])
        evidence += f"<p>📊 Heavy-tailed (γ₂ > {kurt_heavy:.1f}):<br>{items}</p>"

    # Insight adaptativo
    if not highly_skewed.empty:
        insight = f"{len(highly_skewed)} variables show severe asymmetry."
        interpretation = (
            "Right-skewed distributions contain few extreme high values, which can distort "
            "mean-based statistics. Left-skewed distributions show the opposite pattern."
        )
        action = "Consider log transformation (for right skew) or power transformation (Box-Cox, Yeo-Johnson)."
    else:
        insight = "Most variables show symmetric or mildly skewed distributions."
        interpretation = "Distribution shapes are suitable for parametric statistical tests."
        action = "Proceed with standard analysis. Monitor for outliers."

    risk = "Severely skewed variables can violate normality assumptions and destabilize ML models."

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    math_insight("skewness")
    math_insight("kurtosis")

    _log_explore_step(
        "story_shape",
        {
            "n_cols": int(len(cols)),
            "n_highly_skewed": int(len(highly_skewed)),
            "n_moderately_skewed": int(len(moderately_skewed)),
            "n_heavy_tailed": int(len(heavy_tailed)),
            "skew_moderate": float(skew_mod),
            "skew_severe": float(skew_severe),
            "kurtosis_heavy": float(kurt_heavy),
        },
    )

    return html


# =============================================================================
# 7. OUTLIER STORY
# =============================================================================

def story_outliers(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    method: str = "iqr",
    z_thresh: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> str:
    """
    Narrativa sobre detección de outliers.

    Args:
        df: DataFrame a analizar
        cols: columnas específicas (None = todas las numéricas)
        method: "iqr" o "zscore"
        z_thresh: umbral para z-score
        iqr_multiplier: multiplicador para IQR

    Returns:
        HTML string con narrativa
    """

    if _silent():
        return ""

    if cols is None:
        types = _get_types(df)
        cols = types["numeric"]

    if not cols:
        _log_explore_step(
            "story_outliers",
            {"n_cols": 0, "method": method},
        )
        return ""

    thresholds = _get_explore_thresholds()
    severe_pct_thr = thresholds["outlier_severe_pct"]
    moderate_pct_thr = thresholds["outlier_moderate_pct"]

    outlier_counts: Dict[str, int] = {}
    outlier_pcts: Dict[str, float] = {}

    for col in cols:
        series = df[col]
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            if pd.isna(IQR) or IQR == 0:
                outlier_counts[col] = 0
                outlier_pcts[col] = 0.0
                continue
            lower_bound = Q1 - (iqr_multiplier * IQR)
            upper_bound = Q3 + (iqr_multiplier * IQR)
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        else:  # zscore
            std = series.std()
            if std == 0 or pd.isna(std):
                outlier_counts[col] = 0
                outlier_pcts[col] = 0.0
                continue
            z = np.abs((series - series.mean()) / std)
            outliers = (z > z_thresh).sum()

        outlier_counts[col] = int(outliers)
        outlier_pcts[col] = outliers / len(df) * 100 if len(df) > 0 else 0.0

    section("Outlier Detection", icon="🚨")

    # Filtrar variables con outliers
    vars_with_outliers = {k: v for k, v in outlier_counts.items() if v > 0}

    if not vars_with_outliers:
        _log_explore_step(
            "story_outliers",
            {
                "n_cols": int(len(cols)),
                "method": method,
                "n_vars_with_outliers": 0,
            },
        )
        return box(
            "insight",
            "No Outliers Detected",
            f"Using {method.upper()} method, no extreme values were detected.",
            icon="✅",
        )

    # Clasificar por severidad
    severe = {k: v for k, v in outlier_pcts.items() if v >= severe_pct_thr}
    moderate = {k: v for k, v in outlier_pcts.items() if moderate_pct_thr <= v < severe_pct_thr}
    mild = {k: v for k, v in outlier_pcts.items() if 0 < v < moderate_pct_thr}

    # Construir evidencia
    evidence = f"<p><strong>Method:</strong> {method.upper()}</p>"

    if severe:
        items = "<br>".join(
            [f"{col}: <strong>{outlier_counts[col]}</strong> ({pct:.1f}%)" for col, pct in severe.items()]
        )
        evidence += f"<p>🚨 <strong>Severe (≥{severe_pct_thr:.1f}%):</strong><br>{items}</p>"

    if moderate:
        items = "<br>".join(
            [f"{col}: <strong>{outlier_counts[col]}</strong> ({pct:.1f}%)" for col, pct in moderate.items()]
        )
        evidence += f"<p>⚠️ <strong>Moderate ({moderate_pct_thr:.1f}%-{severe_pct_thr:.1f}%):</strong><br>{items}</p>"

    if mild:
        items = "<br>".join(
            [f"{col}: <strong>{outlier_counts[col]}</strong> ({pct:.1f}%)" for col, pct in list(mild.items())[:5]]
        )
        if len(mild) > 5:
            items += f"<br><em>... and {len(mild) - 5} more variables</em>"
        evidence += f"<p>ℹ️ <strong>Mild (<{moderate_pct_thr:.1f}%):</strong><br>{items}</p>"

    # Insight adaptativo
    if severe:
        insight = f"{len(severe)} variables show severe outlier presence (≥{severe_pct_thr:.1f}%)."
        interpretation = (
            "Extreme values may represent: (1) data entry errors, (2) genuine rare events, "
            "(3) important segments (VIP customers, fraud cases)."
        )
        action = "Investigate high-impact outliers individually. Consider capping, removal, or separate modeling."
    else:
        insight = f"Outliers detected in {len(vars_with_outliers)} variables."
        interpretation = "Mild outlier presence is normal in real-world data."
        action = "Monitor their influence on model performance. Use robust methods if needed."

    risk = "Outliers can distort mean-based models (linear regression), distance-based models (k-NN), and statistical tests."

    _log_explore_step(
        "story_outliers",
        {
            "n_cols": int(len(cols)),
            "method": method,
            "n_vars_with_outliers": int(len(vars_with_outliers)),
            "n_severe": int(len(severe)),
            "n_moderate": int(len(moderate)),
            "n_mild": int(len(mild)),
            "z_thresh": float(z_thresh),
            "iqr_multiplier": float(iqr_multiplier),
            "severe_pct": float(severe_pct_thr),
            "moderate_pct": float(moderate_pct_thr),
        },
    )

    return narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )


# =============================================================================
# 8. CORRELATION STORY
# =============================================================================

def story_correlation(df: pd.DataFrame, threshold: float = 0.5) -> str:
    """
    Narrativa sobre matriz de correlación.

    Args:
        df: DataFrame a analizar
        threshold: umbral absoluto para destacar correlaciones

    Returns:
        HTML string con narrativa
    """

    if _silent():
        return ""

    types = _get_types(df)
    numeric_cols = types["numeric"]

    if len(numeric_cols) < 2:
        _log_explore_step(
            "story_correlation",
            {"n_numeric_cols": int(len(numeric_cols)), "n_strong_pairs": 0},
        )
        return box(
            "neutral",
            "Insufficient Variables",
            "Need at least 2 numerical variables for correlation analysis.",
        )

    corr = df[numeric_cols].corr()

    thresholds = _get_explore_thresholds({"corr_strong": threshold})
    strong_thr = thresholds["corr_strong"]
    very_high_thr = thresholds["corr_very_high"]

    section("Correlation Analysis", icon="🔗")

    # Extraer pares con correlación fuerte
    strong: List[tuple] = []
    for i, a in enumerate(corr.columns):
        for j, b in enumerate(corr.columns):
            if i < j:  # Solo triángulo superior
                val = corr.loc[a, b]
                if abs(val) >= strong_thr:
                    strong.append((a, b, val))

    # Ordenar por magnitud
    strong.sort(key=lambda x: abs(x[2]), reverse=True)

    if not strong:
        _log_explore_step(
            "story_correlation",
            {"n_numeric_cols": int(len(numeric_cols)), "n_strong_pairs": 0},
        )
        return box(
            "neutral",
            "No Strong Correlations",
            f"No variable pairs exceed the ±{strong_thr} threshold. Variables appear largely independent.",
            icon="ℹ️",
        )

    # Construir evidencia
    items = []
    for a, b, v in strong[:10]:  # Top 10
        direction = "positive" if v > 0 else "negative"
        items.append(f"<strong>{a}</strong> ↔ <strong>{b}</strong>: {v:.2f} ({direction})")

    evidence = "<br>".join(items)
    if len(strong) > 10:
        evidence += f"<br><em>... and {len(strong) - 10} more pairs</em>"

    # Detectar multicolinealidad severa
    very_high = [x for x in strong if abs(x[2]) >= very_high_thr]

    if very_high:
        insight = (
            f"{len(very_high)} variable pairs show very high correlation "
            f"( |r| ≥ {very_high_thr:.2f} )."
        )
        interpretation = (
            "Very high correlations indicate potential multicollinearity. "
            "These variables may carry redundant information."
        )
        action = (
            "For predictive modeling: use dimensionality reduction (PCA) or feature selection. "
            "For interpretation: keep the most meaningful variable in each highly correlated pair."
        )
    else:
        insight = (
            f"{len(strong)} variable pairs show strong linear associations "
            f"( |r| ≥ {strong_thr:.2f} )."
        )
        interpretation = "Correlated variables may represent similar underlying dimensions or causal relationships."
        action = "Use correlation to guide feature engineering and variable selection."

    risk = "Correlation ≠ causation. Consider confounders, temporal ordering, and domain logic."

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    math_insight("correlation")

    _log_explore_step(
        "story_correlation",
        {
            "n_numeric_cols": int(len(numeric_cols)),
            "n_strong_pairs": int(len(strong)),
            "n_very_high_pairs": int(len(very_high)),
            "threshold_strong": float(strong_thr),
            "threshold_very_high": float(very_high_thr),
        },
    )

    return html


# =============================================================================
# 9. DATA QUALITY REPORT
# =============================================================================


def story_data_quality(df: pd.DataFrame) -> str:
    """
    Comprehensive narrative about data quality.

    Uses prep.get_data_quality_report() when available; otherwise falls back
    to a minimal in-function report.

    This version is defensive so that it never crashes notebooks or nbconvert.
    """

    if _silent():
        return ""

    section("Data Quality Report")

    # --- 1) Try to get the central quality report safely --------------------
    try:
        if PREP_AVAILABLE and callable(get_data_quality_report):
            report = get_data_quality_report(df)  # type: ignore[call-arg]
        else:
            raise RuntimeError("prep.get_data_quality_report not available")
    except Exception:
        # Fallback: very simple report, but guaranteed to work
        report = {
            "shape": df.shape,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "duplicate_count": int(df.duplicated().sum()),
            "missing_summary": df.isna().sum().to_dict(),
            # Minimal type summary – optional
            "type_summary": {
                "numeric": int(df.select_dtypes(include=[np.number]).shape[1]),
                "categorical": int(df.select_dtypes(include=["object", "category", "bool"]).shape[1]),
            },
            "constant_cols": [],
            "id_like_cols": [],
            "high_cardinality_cols": [],
        }

    # --- 2) Robust dimension extraction -------------------------------------
    n_rows = report.get("n_rows")
    n_cols = report.get("n_cols")
    if n_rows is None or n_cols is None:
        shape = report.get("shape", df.shape)
        try:
            n_rows, n_cols = int(shape[0]), int(shape[1])
        except Exception:
            n_rows, n_cols = df.shape

    duplicate_count = report.get("duplicate_count")
    if duplicate_count is None:
        duplicate_count = int(df.duplicated().sum())
    else:
        duplicate_count = int(duplicate_count)

    missing_summary = report.get("missing_summary")
    if missing_summary is None:
        missing_summary = df.isna().sum().to_dict()

    # --- 3) Build content HTML ----------------------------------------------
    content = (
        f"<p><strong>Dataset Dimensions:</strong> {n_rows:,} rows × {n_cols} columns</p>"
        f"<p><strong>Duplicates:</strong> {duplicate_count:,}</p>"
    )

    # Missing data summary
    missing_vars = len([v for v in missing_summary.values() if v > 0])
    if missing_vars > 0:
        content += f"<p><strong>Variables with Missing:</strong> {missing_vars}</p>"
    else:
        content += "<p><strong>Missing Data:</strong> None ✅</p>"

    # Type summary (if available)
    type_sum = report.get("type_summary") or {}
    if isinstance(type_sum, dict) and type_sum:
        content += "<br><p><strong>Variable Type Distribution:</strong></p><ul>"
        for typ, count in type_sum.items():
            try:
                count_int = int(count)
            except Exception:
                continue
            if count_int > 0:
                content += f"<li>{str(typ).capitalize()}: {count_int}</li>"
        content += "</ul>"

    # Quality warnings
    constant_cols = report.get("constant_cols") or []
    id_like_cols = report.get("id_like_cols") or []
    high_card_cols = report.get("high_cardinality_cols") or []

    warnings: List[str] = []
    if constant_cols:
        warnings.append(f"{len(constant_cols)} constant columns detected")
    if id_like_cols:
        warnings.append(f"{len(id_like_cols)} ID-like columns detected")
    if high_card_cols:
        warnings.append(f"{len(high_card_cols)} high-cardinality columns detected")

    if warnings:
        content += "<br><p><strong>⚠️ Quality Issues:</strong></p><ul>"
        for w in warnings:
            content += f"<li>{w}</li>"
        content += "</ul>"
        content += "<p><em>Consider data cleaning before analysis.</em></p>"
        overall_status = "issues_detected"
    else:
        content += "<br><p><strong>✅ Overall Quality:</strong> Good</p>"
        overall_status = "good"

    # --- 4) Safe logging -----------------------------------------------------
    try:
        _log_explore_step(
            "story_data_quality",
            {
                "n_rows": int(n_rows),
                "n_cols": int(n_cols),
                "duplicate_count": int(duplicate_count),
                "n_missing_vars": int(missing_vars),
                "n_constant_cols": int(len(constant_cols)),
                "n_id_like_cols": int(len(id_like_cols)),
                "n_high_cardinality_cols": int(len(high_card_cols)),
                "overall_status": overall_status,
            },
        )
    except Exception:
        # logging must never break story
        pass

    return box("insight", "Quality Assessment", content, icon="📋")



# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Core stories
    "story_overview",
    "story_variable_types",
    "story_missing",
    "story_duplicates",
    "story_descriptive",
    "story_shape",
    "story_outliers",
    "story_correlation",
    # Enhanced stories
    "story_data_quality",
]

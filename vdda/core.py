"""
================================================================================
INSIGHTLAB — CORE MODULE (Final Clean Version)
================================================================================
Professional and educational analytics system:
    • Unified global configuration
    • Consistent visual identity (Navy #1d085e, Teal #13d6c1)
    • UI components (section, box, divider)
    • Educational math insights (math_insight)
    • Safe HTML rendering in Jupyter
    • (New) AnalysisTrace for structured logging
    • (New) show_dataframe for premium table rendering (no display.py needed)

Author: Valesca Bravo
Version: 4.2.1 (Aligned with Blueprint + Trace-ready + Table Rendering)
License: MIT
================================================================================
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List


# =============================================================================
# 0. JUPYTER HTML RENDERING
# =============================================================================

try:
    from IPython.display import HTML, display
    JUPYTER_AVAILABLE = True
except Exception:
    HTML = None
    display = None
    JUPYTER_AVAILABLE = False


# =============================================================================
# 1. VISUAL STYLE (WCAG 2.1 AA)
# =============================================================================

class InsightLabStyle:
    """
    Professional visual identity configuration.
    Colours and typography selected with accessibility in mind.
    """

    # Primary palette
    NAVY = "#1d085e"
    TEAL = "#13d6c1"
    TEAL_DARK = "#0a9a88"
    WHITE = "#ffffff"
    GRAY_LIGHT = "#f8f9fa"
    GRAY_MED = "#6c757d"
    GRAY_DARK = "#343a40"

    # Visualisation palette (colour-blind friendly)
    VIZ_PALETTE = [
        "#1d085e",  # navy
        "#0a9a88",  # deep teal
        "#ff7043",  # orange
        "#ffc107",  # amber
        "#78909c",  # blue-grey
        "#9c27b0",  # purple
        "#4caf50",  # green
        "#e78ac3",  # soft pink
        "#8da0cb",  # periwinkle / soft blue
        "#a6d854",  # yellow-green
    ]

    FONT_FAMILY = (
        "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Roboto, 'Helvetica Neue', sans-serif"
    )

    FONT_SIZE_TITLE = "24px"
    FONT_SIZE_SUBTITLE = "18px"
    FONT_SIZE_BODY = "16px"
    FONT_SIZE_SMALL = "14px"

    @classmethod
    def get_css_string(cls) -> str:
        """Return CSS to be injected into notebook HTML outputs."""

        return f"""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        body {{
            background: {cls.WHITE};
            font-family: {cls.FONT_FAMILY};
        }}

        .il-section {{
            color: {cls.NAVY};
            font-size: {cls.FONT_SIZE_TITLE};
            font-weight: 700;
            margin: 22px 0 10px 0;
            border-left: 4px solid {cls.TEAL};
            padding-left: 14px;
        }}

        .il-subtitle {{
            color: {cls.NAVY};
            font-size: {cls.FONT_SIZE_SUBTITLE};
            font-weight: 600;
            margin: 16px 0 10px;
        }}

        .il-text {{
            font-size: {cls.FONT_SIZE_BODY};
            color: {cls.GRAY_DARK};
            line-height: 1.6;
        }}

        .il-formula {{
            font-family: 'Courier New', monospace;
            background: {cls.WHITE};
            border: 1px solid {cls.GRAY_MED};
            padding: 12px;
            border-radius: 4px;
            margin: 6px 0;
        }}

        .il-box {{
            background: {cls.WHITE};
            border: 2px solid {cls.TEAL};
            border-radius: 8px;
            padding: 18px;
            margin: 16px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}

        .il-box-insight {{
            background: {cls.GRAY_LIGHT};
            border-left: 4px solid {cls.TEAL_DARK};
            padding: 16px;
            margin: 16px 0;
            border-radius: 4px;
        }}

        .il-box-warning {{
            background: #fff8e1;
            border-left: 4px solid #ffc107;
            padding: 16px;
            margin: 16px 0;
            border-radius: 4px;
        }}

        .il-box-neutral {{
            background: {cls.GRAY_LIGHT};
            border-left: 4px solid {cls.GRAY_MED};
            padding: 16px;
            margin: 16px 0;
            border-radius: 4px;
        }}

        .il-box-business {{
            background: {cls.WHITE};
            border-left: 4px solid {cls.TEAL};
            padding: 16px;
            margin: 16px 0;
            border-radius: 4px;
        }}

        .il-divider {{
            border: none;
            height: 2px;
            background: linear-gradient(to right, {cls.TEAL}, {cls.NAVY});
            margin: 24px 0;
        }}

        /* ---------------------------------------------------------------------
           Premium table rendering (used by show_dataframe)
        --------------------------------------------------------------------- */
        .il-table-wrap {{
            overflow-x: auto;
            margin-top: 8px;
        }}

        .il-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: {cls.FONT_SIZE_SMALL};
        }}

        .il-table thead th {{
            text-align: left;
            background: {cls.GRAY_LIGHT};
            color: {cls.NAVY};
            padding: 10px 8px;
            border-bottom: 1px solid #dee2e6;
            font-weight: 700;
            position: sticky;
            top: 0;
        }}

        .il-table tbody td {{
            padding: 8px;
            border-bottom: 1px solid #eef1f4;
            color: {cls.GRAY_DARK};
            vertical-align: top;
        }}

        .il-table tbody tr:hover {{
            background: #f1f3f5;
        }}

        .il-note {{
            font-size: {cls.FONT_SIZE_SMALL};
            color: {cls.GRAY_MED};
            margin-top: 8px;
        }}
        """


# =============================================================================
# 2. MATHEMATICAL CONCEPTS (CLEANED + EXTENDED)
# =============================================================================

CONCEPTS: Dict[str, Dict[str, str]] = {
    "pvalue": {
        "formula": "P(data | H₀)",
        "stakeholder": (
            "The p-value tells us: if the null hypothesis were true, "
            "how likely is it to observe results this extreme?"
        ),
        "technical": (
            "Probability of observing data at least as extreme as the sample "
            "under H₀. Low values suggest incompatibility with the null."
        ),
        "interpretation": (
            "Small p-values (< 0.05) indicate the effect is unlikely due to chance."
        ),
    },
    "effect_size": {
        "formula": "d = (μ₁ - μ₂) / σ_pooled",
        "stakeholder": "Effect size measures how large the difference truly is.",
        "technical": (
            "Standardised mean difference. 0.2 = small, 0.5 = medium, 0.8 = large."
        ),
        "interpretation": (
            "Large effect sizes indicate practical, not only statistical, importance."
        ),
    },
    "correlation": {
        "formula": "r = cov(X,Y) / (σₓ σᵧ)",
        "stakeholder": "Correlation tells you how two variables move together.",
        "technical": (
            "Pearson correlation ranges from -1 to +1 and measures linear association."
        ),
        "interpretation": (
            "Correlation does not imply causation; confounders may exist."
        ),
    },
    "r2": {
        "formula": "R² = 1 - (SS_residual / SS_total)",
        "stakeholder": "R² tells you how much variation the model explains.",
        "technical": "Goodness of fit. Higher R² = better explanatory power.",
        "interpretation": (
            "R² = 0.7 means 70% of outcome variance is explained by the model."
        ),
    },
    # -------------------------------------------------------------------------
    # Concepts commonly used by story / ML modules
    # -------------------------------------------------------------------------
    "skewness": {
        "formula": "γ₁ = E[(X-μ)³] / σ³",
        "stakeholder": (
            "Skewness describes whether the distribution leans more to the left "
            "or to the right."
        ),
        "technical": (
            "Third standardised moment of a distribution. "
            "Positive skewness = right-tailed; negative = left-tailed."
        ),
        "interpretation": (
            "|γ₁| > 1 suggests strong asymmetry that can affect means, "
            "standard deviations and parametric tests."
        ),
    },
    "kurtosis": {
        "formula": "γ₂ = E[(X-μ)⁴] / σ⁴",
        "stakeholder": (
            "Kurtosis measures how heavy or light the tails of a distribution are "
            "compared with a normal curve."
        ),
        "technical": (
            "Fourth standardised moment. Values greater than 3 indicate heavy tails; "
            "values below 3 indicate light tails."
        ),
        "interpretation": (
            "High kurtosis implies more extreme values than expected, which can make "
            "models sensitive to outliers."
        ),
    },
    "anova": {
        "formula": "F = MS_between / MS_within",
        "stakeholder": (
            "ANOVA tests whether several group means differ beyond what we expect "
            "from random noise."
        ),
        "technical": (
            "Compares between-group variance to within-group variance using "
            "the F distribution."
        ),
        "interpretation": (
            "A significant F suggests at least one group mean differs; "
            "post-hoc tests identify which ones."
        ),
    },
    "chi_square": {
        "formula": "χ² = Σ (O - E)² / E",
        "stakeholder": (
            "The chi-square test checks whether two categorical variables "
            "are related or independent."
        ),
        "technical": (
            "Compares observed cell counts to expected counts under independence; "
            "large χ² indicates lack of fit."
        ),
        "interpretation": (
            "Significant χ² suggests an association; effect sizes like Cramér's V "
            "quantify its strength."
        ),
    },
    "pca": {
        "formula": "PCₖ = w₁X₁ + w₂X₂ + … + w_pX_p",
        "stakeholder": (
            "PCA compresses many correlated variables into a smaller set of components "
            "that still capture most variation."
        ),
        "technical": (
            "Orthogonal linear combinations of standardised variables ordered "
            "by explained variance."
        ),
        "interpretation": (
            "High cumulative variance explained means we can work in a lower "
            "dimensional space with little information loss."
        ),
    },
    "bayesian": {
        "formula": "Posterior ∝ Likelihood × Prior",
        "stakeholder": (
            "Bayesian analysis updates what we believe about a parameter "
            "after seeing the data."
        ),
        "technical": (
            "Uses Bayes' theorem to combine prior distributions with likelihoods "
            "and obtain posterior distributions."
        ),
        "interpretation": (
            "Posterior summaries and credible intervals give direct probability "
            "statements about parameters."
        ),
    },
    "decision_tree": {
        "formula": "Split criterion: Gini / Entropy / MSE",
        "stakeholder": (
            "Decision trees split the data into segments so that each leaf is as "
            "homogeneous as possible."
        ),
        "technical": (
            "Greedy recursive partitioning that chooses splits maximising impurity "
            "reduction at each node."
        ),
        "interpretation": (
            "Tree structure reveals how features interact to drive predictions; "
            "shallow trees are easier to explain."
        ),
    },
}


# =============================================================================
# 3. ANALYSIS TRACE (STRUCTURED LOGGING)
# =============================================================================

@dataclass
class AnalysisTrace:
    """
    Structured record of analytical steps.

    This is not yet used in every module, but it keeps the system ready to log:
        • stage (prep, stats, viz, ml, story, export)
        • step name
        • details (metrics, assumptions, flags)
    """

    steps: List[Dict[str, Any]] = field(default_factory=list)

    def start_run(self, name: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Mark the start of an analysis session."""
        self.steps.append({
            "stage": "run_start",
            "name": name,
            "meta": meta or {},
        })

    def log(self, stage: str, step: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log an analytical step."""
        self.steps.append({
            "stage": stage,
            "step": step,
            "details": details or {},
        })

    def to_list(self) -> List[Dict[str, Any]]:
        """Return the full trace as a list of dictionaries."""
        return list(self.steps)

    def clear(self) -> None:
        """Clear the trace."""
        self.steps.clear()


# =============================================================================
# 4. GLOBAL CONFIGURATION
# =============================================================================

@dataclass
class InsightLabConfig:
    """Global configuration for InsightLab."""

    verbosity: str = "stakeholder"   # "silent" | "stakeholder" | "technical"
    domain: str = "general"
    theme: str = "auto"
    precision: int = 3
    auto_plot: bool = True

    domain_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    loaded_modules: Dict[str, Any] = field(default_factory=dict)
    domain_defaults: Dict[str, Any] = field(default_factory=dict)
    concepts: Dict[str, Dict[str, str]] = field(default_factory=lambda: CONCEPTS)
    Style = InsightLabStyle

    # New: structured trace
    trace: AnalysisTrace = field(default_factory=AnalysisTrace)

    def set(self, verbosity=None, domain=None, precision=None, auto_plot=None, **kwargs):
        """Update global parameters."""
        if verbosity is not None:
            self.verbosity = verbosity
        if domain is not None:
            self.domain = domain
        if precision is not None:
            self.precision = int(precision)
        if auto_plot is not None:
            self.auto_plot = bool(auto_plot)
        if kwargs:
            self.domain_defaults.update(kwargs)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the current configuration."""
        return {
            "verbosity": self.verbosity,
            "domain": self.domain,
            "precision": self.precision,
            "auto_plot": self.auto_plot,
            "loaded_modules": list(self.loaded_modules.keys()),
            "presets": list(self.domain_settings.keys()),
        }

    # Convenience hooks for trace

    def start_run(self, name: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.trace.start_run(name, meta)

    def log_step(self, stage: str, step: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.trace.log(stage, step, details)

    def get_trace(self) -> List[Dict[str, Any]]:
        return self.trace.to_list()

    def reset_trace(self) -> None:
        self.trace.clear()


# Global config instance
CONFIG = InsightLabConfig()


# =============================================================================
# 5. HELPERS — VERBOSITY
# =============================================================================

def _silent() -> bool:
    return CONFIG.verbosity == "silent"


def _technical() -> bool:
    return CONFIG.verbosity == "technical"


def _stakeholder() -> bool:
    return CONFIG.verbosity == "stakeholder"


# =============================================================================
# 5b. HELPERS — VISUAL
# =============================================================================

def auto_plot_enabled() -> bool:
    return bool(getattr(CONFIG, "auto_plot", True))


# =============================================================================
# 6. DOMAIN PRESETS
# =============================================================================

def load_domain(domain: str) -> Dict[str, Any]:
    """Load a domain preset from presets/all.py."""
    module = importlib.import_module("insightlab.presets.all")

    if not hasattr(module, "PRESETS"):
        raise ValueError("presets/all.py must define PRESETS")

    presets = module.PRESETS

    if domain not in presets:
        raise ValueError(f"Domain '{domain}' not found.")

    CONFIG.domain_settings[domain] = presets[domain]
    CONFIG.domain = domain
    CONFIG.set(**presets[domain].get("defaults", {}))
    return presets[domain]


def _domain_phrase(key: str) -> str:
    """
    Retrieve a pre-defined phrase from the active domain preset.

    Used by the narrative engine (story.py) to fill risk messages and context
    footers depending on the domain (marketing, ecommerce, etc.).
    """
    domain = CONFIG.domain

    # 1) Use configuration already loaded in CONFIG.domain_settings
    info = CONFIG.domain_settings.get(domain)
    if isinstance(info, dict):
        phrases = info.get("phrases", {})
        if isinstance(phrases, dict):
            return phrases.get(key, "")

    # 2) Fallback: read directly from presets/all.py if load_domain() has not run
    try:
        module = importlib.import_module("insightlab.presets.all")
        presets = getattr(module, "PRESETS", {})
        info = presets.get(domain, {})
        phrases = info.get("phrases", {})
        if isinstance(phrases, dict):
            return phrases.get(key, "")
    except Exception:
        # Silent by design ("silent unless asked")
        pass

    return ""


# =============================================================================
# 7. HTML RENDERING
# =============================================================================

def show_html(html: str):
    """Render HTML in Jupyter (no-op outside notebooks)."""
    if _silent():
        return
    if JUPYTER_AVAILABLE:
        display(HTML(html))


def show_dataframe(
    df: Any,
    title: str = "",
    max_rows: int = 30,
) -> str:
    """
    Render a DataFrame-like object as a styled HTML table in Jupyter.

    This is a lightweight replacement for a separate display.py module and is
    intended to support "premium" portfolio outputs.

    Args:
        df: Usually a pandas DataFrame (but any object with .to_html() will work).
        title: Optional title shown above the table.
        max_rows: Maximum rows to render (prevents giant HTML outputs).

    Returns:
        The HTML string (also rendered in Jupyter when available).
    """
    if _silent():
        return ""

    note_html = ""
    df_to_show = df

    try:
        import pandas as pd  # local import to avoid hard dependency
        if isinstance(df, pd.DataFrame) and max_rows is not None and df.shape[0] > max_rows:
            df_to_show = df.head(max_rows)
            note_html = (
                f"<div class='il-note'><em>"
                f"Showing first {max_rows:,} of {df.shape[0]:,} rows."
                f"</em></div>"
            )
    except Exception:
        # If pandas is not available, we proceed with df as-is.
        pass

    header = f"<div class='il-subtitle'>{title}</div>" if title else ""

    # Use default escaping for safety (avoid injecting raw HTML from data values)
    table_html = ""
    if hasattr(df_to_show, "to_html"):
        table_html = df_to_show.to_html(index=False)
    else:
        table_html = f"<pre>{str(df_to_show)}</pre>"

    # Attach the il-table class if possible (basic replace; safe for pandas HTML)
    if "<table" in table_html and "class=" not in table_html.split("<table", 1)[1].split(">", 1)[0]:
        table_html = table_html.replace("<table", "<table class='il-table'", 1)

    html = f"""
    <div class='il-box'>
        {header}
        <div class='il-table-wrap'>
            {table_html}
        </div>
        {note_html}
    </div>
    """

    show_html(html)
    return html


# =============================================================================
# 8. STYLE LOADING
# =============================================================================

def load_style(show_banner: bool = False):
    """Inject InsightLab CSS into the notebook."""
    css = f"<style>{CONFIG.Style.get_css_string()}</style>"
    show_html(css)

    if show_banner:
        show_html("<div class='il-section'>InsightLab styles loaded</div>")


def apply_style():
    """Configure Matplotlib/Seaborn defaults for consistent visuals."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return

    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans"],
        "axes.labelcolor": CONFIG.Style.GRAY_DARK,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    sns.set_palette(CONFIG.Style.VIZ_PALETTE)
    sns.set_style("whitegrid")


# =============================================================================
# 9. UI COMPONENTS
# =============================================================================

def section(title: str, **kwargs) -> str:
    """
    Render a section heading.

    kwargs are ignored (e.g., icon=) for backwards compatibility without
    showing emojis in the design.
    """
    if _silent():
        return ""

    html = f"<div class='il-section'>{title}</div>"
    show_html(html)
    return html


def box(kind: str, title: str, content: str, **kwargs) -> str:
    """
    Render a styled box.
    kind: insight | warning | neutral | business | info

    kwargs are ignored (e.g., icon=) for silent backwards compatibility.
    """
    if _silent():
        return ""

    mapping = {
        "insight": "il-box-insight",
        "warning": "il-box-warning",
        "neutral": "il-box-neutral",
        "business": "il-box-business",
        "info": "il-box",
    }

    css_class = mapping.get(kind, "il-box")
    header = f"<div class='il-subtitle'>{title}</div>" if title else ""

    html = f"""
    <div class='{css_class}'>
        {header}
        <div class='il-text'>{content}</div>
    </div>
    """

    show_html(html)
    return html


def divider():
    """Render a divider line."""
    if _silent():
        return ""
    html = "<hr class='il-divider'>"
    show_html(html)
    return html


# =============================================================================
# 10. MATH INSIGHT
# =============================================================================

def math_insight(key: str):
    """Render an educational explanation of a mathematical concept."""
    if _silent():
        return ""

    if key not in CONFIG.concepts:
        return ""

    c = CONFIG.concepts[key]
    is_tech = _technical()

    explanation = c["technical"] if is_tech else c["stakeholder"]
    interpretation = c["interpretation"] if is_tech else ""

    html = f"""
    <div class="il-box-neutral">
        <div class="il-subtitle">Mathematical Insight</div>
        <div class="il-formula">{c['formula']}</div>
        <div class="il-text"><strong>Meaning:</strong> {explanation}</div>
    """

    if interpretation:
        html += (
            f"<div class='il-text'><strong>Why it matters:</strong> "
            f"{interpretation}</div>"
        )

    html += "</div>"

    show_html(html)
    return html


def business_insight(title: str, content: str):
    """Convenience wrapper for a business insight box."""
    return box("business", title, content)


# =============================================================================
# 11. INITIALISATION
# =============================================================================

def initialize(
    verbosity: Optional[str] = None,
    domain: Optional[str] = None,
    theme: Optional[str] = None,   # accepted for backwards compatibility (ignored)
    style: bool = True,
    reset_trace: bool = True,
) -> None:
    """
    Initialise InsightLab.

    Args:
        verbosity: "silent" | "stakeholder" | "technical"
        domain: domain preset name (marketing, ecommerce, etc.)
        theme: accepted for compatibility, but not used
        style: if True, inject CSS into the notebook
        reset_trace: if True, clear any previous trace
    """
    if verbosity:
        CONFIG.set(verbosity=verbosity)

    if domain:
        load_domain(domain)

    if style:
        load_style(show_banner=False)

    if reset_trace:
        CONFIG.reset_trace()


# =============================================================================
# 12. PUBLIC EXPORTS
# =============================================================================

__all__ = [
    "CONFIG",
    "InsightLabConfig",
    "InsightLabStyle",
    "AnalysisTrace",
    "initialize",
    "load_style",
    "apply_style",
    "load_domain",
    "_domain_phrase",
    "section",
    "box",
    "divider",
    "show_html",
    "show_dataframe",          # <-- new
    "math_insight",
    "business_insight",
    "_silent",
    "_technical",
    "_stakeholder",
]

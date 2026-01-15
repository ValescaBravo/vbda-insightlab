"""
===============================================================================
INSIGHTLAB — STATISTICAL STORYTELLING MODULE
===============================================================================
Transforms statistical test results into business-ready narratives.

This module focuses ONLY on translating results:
    • Computation lives in stats.py
    • Narratives are built with story.py
    • UI components come from core.py

Full coverage:
    • T-tests (independent, paired, one-sample)
    • ANOVA (one-way, repeated measures)
    • Chi-square / contingency tables
    • Correlation (Pearson, Spearman, Kendall)
    • Non-parametric tests (Mann–Whitney, Wilcoxon, Kruskal–Wallis)
    • Normality tests (Shapiro–Wilk, Kolmogorov–Smirnov)
    • Homogeneity tests (Levene, Bartlett)
    • Post-hoc tests (Tukey HSD, Dunn)
    • Bayesian proportion tests

Comments: English
Output: English
===============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Core: UI components and educational concepts
from insightlab.core import (
    section,
    box,
    math_insight,
    CONFIG,
    _silent,
    _stakeholder,
    _technical,
)

# Story: narrative frameworks
from insightlab.narrative import (
    narrative,
    narrative_from_dict,
    interpretation_5layers,
    star,
    explain,
    result,
)

# Stats: numeric engine
import insightlab.stats as il_stats

# Explore helpers (thresholds, typing, logging)
from insightlab.narrative.story_explore import (
    _log_explore_step,
    _get_explore_thresholds,
    _get_types,
)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _format_pvalue(p: float, alpha: float = 0.05) -> str:
    """
    Format a p-value with a contextual interpretation.
    """
    if p < 0.001:
        return "p < 0.001 (highly significant)"
    elif p < alpha:
        return f"p = {p:.4f} (significant at α = {alpha})"
    else:
        return f"p = {p:.4f} (not significant at α = {alpha})"


def _interpret_significance(p: float, alpha: float = 0.05) -> str:
    """
    Return a verbal interpretation of statistical significance.
    """
    return "significant" if p < alpha else "not significant"


def _effect_size_interpretation(magnitude: str) -> str:
    """
    Return a contextual interpretation of effect size magnitude.
    """
    interpretations = {
        "negligible": "The effect is negligible — practically zero impact.",
        "small": "The effect is small but may matter in large samples or cumulative contexts.",
        "medium": "The effect is medium — meaningful in most practical scenarios.",
        "large": "The effect is large — clear practical significance.",
        "very large": "The effect is very large — substantial real-world impact.",
    }
    return interpretations.get(magnitude.lower(), "Effect size noted.")


# =============================================================================
# 1. T-TESTS
# =============================================================================

def story_ttest_independent(
    group1: Union[pd.Series, np.ndarray],
    group2: Union[pd.Series, np.ndarray],
    name1: str = "Group A",
    name2: str = "Group B",
    alpha: float = 0.05,
    stats_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Narrative for an independent-samples t-test.

    Args:
        group1: First group data.
        group2: Second group data.
        name1: Name of the first group.
        name2: Name of the second group.
        alpha: Significance level.
        stats_result: Optional precomputed result.

    Returns:
        HTML string with the full narrative.
    """
    if _silent():
        return ""

    # 1) Compute statistics
    res = stats_result or il_stats.ttest_independent(group1, group2)

    # 2) Extract key values
    p = float(res.get("p_value", np.nan))
    t = float(res.get("statistic", np.nan))
    d = float(res.get("cohen_d", np.nan))
    magnitude = res.get("magnitude", "unknown")

    mean1 = float(res.get("mean1", np.mean(group1)))
    mean2 = float(res.get("mean2", np.mean(group2)))
    diff = mean1 - mean2

    sig_status = _interpret_significance(p, alpha)

    # 3) UI section
    section(f"Independent t-test: {name1} vs {name2}")

    # 4) 5-layer interpretation
    descriptive = (
        f"{name1}: mean ≈ {mean1:.3f}. "
        f"{name2}: mean ≈ {mean2:.3f}. "
        f"Observed difference: {diff:+.3f}."
    )

    statistical = (
        f"t-statistic = {t:.3f}, {_format_pvalue(p, alpha)}. "
        f"Cohen's d = {d:.3f} ({magnitude}). "
        f"The difference is {sig_status}."
    )

    behavioural = (
        f"Individuals in {name1} "
        f"{'score higher on average' if diff > 0 else 'score lower on average'} than {name2}."
        if p < alpha
        else f"No systematic difference detected between {name1} and {name2} groups."
    )

    strategic = (
        "If these groups represent customer segments or experimental conditions, "
        "investigate what drives the difference and whether it is actionable."
        if p < alpha
        else "The observed variation appears compatible with random sampling fluctuation."
    )

    operational = (
        "Validate assumptions: check normality (Shapiro–Wilk) and equal variances (Levene test). "
        "If assumptions fail, consider the Mann–Whitney U test as a non-parametric alternative."
    )

    html = interpretation_5layers(
        descriptive=descriptive,
        statistical=statistical,
        behavioural=behavioural,
        strategic=strategic,
        operational=operational,
    )

    # 5) Educational concepts
    math_insight("pvalue")
    math_insight("effect_size")

    return html


def story_ttest_paired(
    before: Union[pd.Series, np.ndarray],
    after: Union[pd.Series, np.ndarray],
    name_before: str = "Before",
    name_after: str = "After",
    alpha: float = 0.05,
    stats_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Narrative for a paired-samples t-test.
    """
    if _silent():
        return ""

    # 1) Compute statistics
    res = stats_result or il_stats.ttest_paired(before, after)

    # 2) Extract key values
    p = float(res.get("p_value", np.nan))
    t = float(res.get("statistic", np.nan))
    d = float(res.get("cohen_d", np.nan))
    magnitude = res.get("magnitude", "unknown")

    mean_before = float(np.mean(before))
    mean_after = float(np.mean(after))
    mean_diff = mean_after - mean_before

    sig_status = _interpret_significance(p, alpha)

    # 3) UI
    section(f"Paired t-test: {name_before} vs {name_after}")

    # 4) Narrative
    descriptive = (
        f"{name_before}: mean ≈ {mean_before:.3f}. "
        f"{name_after}: mean ≈ {mean_after:.3f}. "
        f"Mean change: {mean_diff:+.3f}."
    )

    statistical = (
        f"t-statistic = {t:.3f}, {_format_pvalue(p, alpha)}. "
        f"Cohen's d = {d:.3f} ({magnitude}). "
        f"The change is {sig_status}."
    )

    behavioural = (
        f"Participants show a {'positive' if mean_diff > 0 else 'negative'} shift "
        f"from {name_before} to {name_after}."
        if p < alpha
        else "No systematic change detected between time points."
    )

    strategic = (
        "If this represents a treatment or intervention effect, the observed change suggests meaningful impact."
        if p < alpha
        else "The intervention appears to have no detectable effect beyond random variation."
    )

    operational = (
        "Check for normality of differences (Shapiro–Wilk on diff = after − before). "
        "If violated, use the Wilcoxon signed-rank test instead."
    )

    html = interpretation_5layers(
        descriptive=descriptive,
        statistical=statistical,
        behavioural=behavioural,
        strategic=strategic,
        operational=operational,
    )

    math_insight("pvalue")
    math_insight("effect_size")

    return html


def story_ttest_one_sample(
    data: Union[pd.Series, np.ndarray],
    population_mean: float,
    name: str = "Sample",
    alpha: float = 0.05,
    stats_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Narrative for a one-sample t-test.
    """
    if _silent():
        return ""

    # 1) Compute statistics
    res = stats_result or il_stats.ttest_one_sample(data, population_mean)

    # 2) Extract key values
    p = float(res.get("p_value", np.nan))
    t = float(res.get("statistic", np.nan))
    sample_mean = float(np.mean(data))

    sig_status = _interpret_significance(p, alpha)

    # 3) UI
    section(f"One-Sample t-test: {name} vs Population Mean")

    # 4) Narrative
    descriptive = (
        f"Sample mean: {sample_mean:.3f}. "
        f"Population mean: {population_mean:.3f}. "
        f"Difference: {sample_mean - population_mean:+.3f}."
    )

    statistical = (
        f"t-statistic = {t:.3f}, {_format_pvalue(p, alpha)}. "
        f"The sample is {sig_status}ly different from the population."
    )

    behavioural = (
        f"The {name} group {'exceeds' if sample_mean > population_mean else 'falls below'} "
        f"the expected population benchmark."
        if p < alpha
        else f"The {name} group is consistent with the population mean."
    )

    strategic = (
        "This suggests the sample comes from a different underlying distribution."
        if p < alpha
        else "No evidence of systematic deviation from the population."
    )

    operational = (
        "Validate the normality assumption and confirm that the population mean is a valid benchmark."
    )

    html = interpretation_5layers(
        descriptive=descriptive,
        statistical=statistical,
        behavioural=behavioural,
        strategic=strategic,
        operational=operational,
    )

    math_insight("pvalue")

    return html


# =============================================================================
# 2. ANOVA
# =============================================================================

def story_anova_oneway(
    *groups,
    group_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    stats_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Narrative for a one-way ANOVA.
    """
    if _silent():
        return ""

    # 1) Compute statistics
    res = stats_result or il_stats.anova_oneway(*groups)

    # 2) Extract key values
    p = float(res.get("p_value", np.nan))
    f = float(res.get("statistic", np.nan))
    eta2 = float(res.get("eta2", np.nan))
    magnitude = res.get("magnitude", "unknown")

    df_between = res.get("df_between", len(groups) - 1)
    df_within = res.get("df_within", sum(len(g) for g in groups) - len(groups))

    sig_status = _interpret_significance(p, alpha)

    # 3) Group names
    if group_names is None:
        group_names = [f"Group {i+1}" for i in range(len(groups))]

    # 4) UI
    section(f"One-Way ANOVA: {', '.join(group_names)}")

    # 5) Narrative
    means_str = ", ".join(
        [f"{name}: {np.mean(g):.2f}" for name, g in zip(group_names, groups)]
    )

    descriptive = f"Comparing {len(groups)} groups: {means_str}."

    statistical = (
        f"F({df_between}, {df_within}) = {f:.3f}, {_format_pvalue(p, alpha)}. "
        f"η² = {eta2:.3f} ({magnitude}). "
        f"Group means are {sig_status}ly different."
    )

    behavioural = (
        "At least one group shows systematically different behaviour from the others."
        if p < alpha
        else "No meaningful differences detected across groups."
    )

    strategic = (
        "Follow up with post-hoc tests (Tukey HSD) to identify which specific pairs differ."
        if p < alpha
        else "Groups appear homogeneous — no further pairwise testing needed."
    )

    operational = (
        "Validate assumptions: normality (Shapiro–Wilk per group) and equal variances (Levene test). "
        "If violated, consider the Kruskal–Wallis test."
    )

    html = interpretation_5layers(
        descriptive=descriptive,
        statistical=statistical,
        behavioural=behavioural,
        strategic=strategic,
        operational=operational,
    )

    math_insight("anova")

    return html


# =============================================================================
# 3. CHI-SQUARE / CONTINGENCY TABLES
# =============================================================================

def story_chi_square(
    contingency_table: pd.DataFrame,
    alpha: float = 0.05,
    stats_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Narrative for a chi-square test of independence.

    Args:
        contingency_table: Contingency table (rows = categories A, columns = categories B).
        alpha: Significance level (default 0.05).
        stats_result: Optional dict with results already computed by il_stats.chi_square_test().

    Returns:
        HTML string with a full narrative (5 layers).
    """
    if _silent():
        return ""

    # 1) Compute statistics
    res = stats_result or il_stats.chi_square_test(contingency_table)

    # 2) Extract key values (aligned with stats.py keys)
    p = float(res.get("p_value", np.nan))
    chi2 = float(res.get("statistic", np.nan))
    cramers = float(res.get("cramers_v", np.nan))
    w = float(res.get("cohens_w", np.nan))
    magnitude = res.get("magnitude", "unknown")
    df = res.get("dof", np.nan)  # aligned key: dof

    sig_status = _interpret_significance(p, alpha)

    # 3) UI
    section("Chi-Square Test of Independence")

    # 4) Narrative: 5 layers
    descriptive = (
        f"Contingency table: {contingency_table.shape[0]} rows × "
        f"{contingency_table.shape[1]} columns. "
        "Testing independence between two categorical variables."
    )

    statistical = (
        f"χ²({df}) = {chi2:.3f}, {_format_pvalue(p, alpha)}. "
        f"Cramér's V = {cramers:.3f}. "
        f"Cohen's w = {w:.3f} ({magnitude}). "
        f"Variables are {sig_status}ly associated."
    )

    behavioural = (
        "The categorical variables show systematic association — knowing one helps predict the other."
        if p < alpha
        else "The variables appear independent — no predictable relationship."
    )

    strategic = (
        "Use this association to inform segmentation, targeting, or feature engineering."
        if p < alpha
        else "You can treat these variables as independent in most modelling scenarios."
    )

    operational = (
        "Inspect standardised residuals to identify which category combinations drive the association. "
        "Ensure expected cell counts ≥ 5 to satisfy chi-square assumptions; otherwise consider Fisher's exact test."
    )

    html = interpretation_5layers(
        descriptive=descriptive,
        statistical=statistical,
        behavioural=behavioural,
        strategic=strategic,
        operational=operational,
    )

    math_insight("chi_square")

    return html


# =============================================================================
# 4. CORRELATION
# =============================================================================

def story_correlation(
    x: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    method: str = "pearson",
    alpha: float = 0.05,
    threshold: float = 0.5,
    stats_result: Optional[Dict[str, Any]] = None,
    var_name_x: Optional[str] = None,
    var_name_y: Optional[str] = None,
) -> str:
    """
    Two-in-one correlation narrative helper.

    • If x is a DataFrame and y is None  → matrix correlation story (strong pairs, multicollinearity).
    • Otherwise                          → pairwise correlation story between x and y.
    """
    if _silent():
        return ""

    # ------------------------------------------------------------------
    # BRANCH 1 — MATRIX MODE
    # ------------------------------------------------------------------
    if isinstance(x, pd.DataFrame) and y is None:
        df = x

        types = _get_types(df)
        numeric_cols = types["numeric"]

        if len(numeric_cols) < 2:
            _log_explore_step(
                "story_correlation_matrix",
                {"n_numeric_cols": int(len(numeric_cols)), "n_strong_pairs": 0},
            )
            return box(
                "neutral",
                "Insufficient Variables",
                "Need at least 2 numerical variables for correlation analysis.",
            )

        corr = df[numeric_cols].corr(method=method)

        thresholds = _get_explore_thresholds({"corr_strong": threshold})
        strong_thr = thresholds["corr_strong"]
        very_high_thr = thresholds["corr_very_high"]

        section(f"Correlation Analysis ({method.capitalize()})")

        # Extract strong correlation pairs (upper triangle only)
        strong: List[tuple] = []
        for i, a in enumerate(corr.columns):
            for j, b in enumerate(corr.columns):
                if i < j:
                    val = corr.loc[a, b]
                    if abs(val) >= strong_thr:
                        strong.append((a, b, val))

        # Sort by magnitude
        strong.sort(key=lambda x: abs(x[2]), reverse=True)

        if not strong:
            _log_explore_step(
                "story_correlation_matrix",
                {"n_numeric_cols": int(len(numeric_cols)), "n_strong_pairs": 0},
            )
            return box(
                "neutral",
                "No Strong Correlations Detected",
                f"No variable pairs exceed the ±{strong_thr:.2f} threshold. Variables appear largely independent.",
            )

        # Strength interpretation helper
        def _interpret_strength(r: float) -> str:
            abs_r = abs(r)
            if abs_r >= 0.9:
                return "VERY STRONG"
            elif abs_r >= 0.7:
                return "STRONG"
            elif abs_r >= 0.5:
                return "MODERATE"
            elif abs_r >= 0.3:
                return "WEAK"
            else:
                return "VERY WEAK"

        items = []
        for a, b, v in strong[:10]:
            direction = "POSITIVE" if v > 0 else "NEGATIVE"
            strength = _interpret_strength(v)
            items.append(
                f"<strong>{a}</strong> ↔ <strong>{b}</strong>: "
                f"<code>{v:.3f}</code> | {strength} | {direction}"
            )

        evidence = "<br>".join(items)
        if len(strong) > 10:
            evidence += f"<br><em>... and {len(strong) - 10} more pairs above threshold</em>"

        # Detect severe multicollinearity
        very_high = [t for t in strong if abs(t[2]) >= very_high_thr]

        if very_high:
            insight = (
                f"{len(very_high)} variable pairs show very high correlation "
                f"(|r| ≥ {very_high_thr:.2f}), indicating potential <strong>multicollinearity</strong>."
            )
            interpretation = (
                "<strong>What This Means:</strong><br>"
                "• Variables carry <strong>redundant information</strong><br>"
                "• In predictive models, this can cause unstable coefficients and inflated standard errors<br>"
                "• One variable may be 'stealing' predictive power from another<br><br>"
                "<strong>Why It Matters:</strong><br>"
                "Very high correlations (|r| > 0.85) suggest the variables measure nearly the same underlying dimension."
            )
            action = (
                "<strong>Recommended Actions:</strong><br>"
                "1) <strong>For Predictive Modelling:</strong> Use dimensionality reduction (PCA) or "
                "feature selection (VIF, LASSO)<br>"
                "2) <strong>For Interpretation:</strong> Keep only the most meaningful variable from each "
                "highly correlated pair<br>"
                "3) <strong>For Segmentation:</strong> Create composite scores combining related variables"
            )
            risk = (
                "<strong>Critical reminder:</strong> <strong>Correlation ≠ causation</strong>.<br>"
                "Variables moving together does not mean one causes the other. "
                "Confirm temporal order, domain logic, and rule out confounders before claiming causality."
            )
        else:
            insight = (
                f"{len(strong)} variable pairs show <strong>strong linear associations</strong> "
                f"(|r| ≥ {strong_thr:.2f})."
            )
            interpretation = (
                "<strong>What This Means:</strong><br>"
                "• These variables tend to move together in a consistent pattern<br>"
                "• The relationship could reflect causal links, shared drivers, or measurement overlap<br><br>"
                "<strong>Strength guide:</strong><br>"
                "• <strong>0.9–1.0:</strong> Almost perfect relationship<br>"
                "• <strong>0.7–0.9:</strong> Strong relationship<br>"
                "• <strong>0.5–0.7:</strong> Moderate relationship (current threshold)<br>"
                "• <strong>0.3–0.5:</strong> Weak but notable relationship"
            )
            action = (
                "<strong>How to use this:</strong><br>"
                "• <strong>Feature engineering:</strong> Create interaction terms for strongly correlated pairs<br>"
                "• <strong>Variable selection:</strong> Remove redundant features before modelling<br>"
                "• <strong>Hypothesis generation:</strong> Investigate <em>why</em> these variables relate<br>"
                "• <strong>Business strategy:</strong> Use relationships to inform targeting or bundling decisions"
            )
            risk = (
                "<strong>Correlation ≠ causation</strong>.<br>"
                "Strong correlation shows variables are related, not that one causes the other."
            )

        html = narrative(
            insight=insight,
            evidence=evidence,
            interpretation=interpretation,
            action=action,
            risk=risk,
        )

        math_insight("correlation")

        _log_explore_step(
            "story_correlation_matrix",
            {
                "n_numeric_cols": int(len(numeric_cols)),
                "n_strong_pairs": int(len(strong)),
                "n_very_high_pairs": int(len(very_high)),
                "threshold_strong": float(strong_thr),
                "threshold_very_high": float(very_high_thr),
                "method": method,
            },
        )

        return html

    # ------------------------------------------------------------------
    # BRANCH 2 — PAIR MODE (x vs y)
    # ------------------------------------------------------------------
    if y is None:
        raise ValueError("story_correlation: when x is not a DataFrame, y cannot be None.")

    # Numeric calculation (stats core)
    res = stats_result or il_stats.correlation(x, y, method=method)
    r = float(res.get("correlation", np.nan))
    p = float(res.get("p_value", np.nan))
    magnitude = str(res.get("magnitude", "unknown")).lower()
    method_name = str(res.get("method", method)).title()

    if var_name_x is None:
        var_name_x = getattr(x, "name", None) or "X"
    if var_name_y is None:
        var_name_y = getattr(y, "name", None) or "Y"

    direction = "positive" if r > 0 else "negative"
    sig_status = _interpret_significance(p, alpha)

    descriptive = (
        f"{var_name_x} and {var_name_y} show a {direction} relationship "
        f"(r = {r:.3f}, strength: {magnitude})."
    )

    statistical = (
        f"{method_name} r = {r:.3f}, {_format_pvalue(p, alpha)}. "
        f"The relationship is statistically {sig_status} at α = {alpha:.2f}."
    )

    behavioural = (
        f"When {var_name_x} increases, {var_name_y} tends to "
        f"{'increase' if r > 0 else 'decrease'}. "
        "This pattern may be driven by specific customer subgroups "
        "(e.g., heavy spenders or highly digital users)."
    )

    strategic = (
        "Use this relationship to prioritise which metrics to monitor together and "
        "where to focus hypotheses, A/B tests, or segmentation refinements."
    )

    operational = (
        "Correlation does not imply causation. Before making strong decisions, "
        "review campaign timing, business rules, and potential confounders that could explain the link."
    )

    html = interpretation_5layers(
        descriptive=descriptive,
        statistical=statistical,
        behavioural=behavioural,
        strategic=strategic,
        operational=operational,
    )

    math_insight("correlation")

    return html


# =============================================================================
# 5. NON-PARAMETRIC TESTS
# =============================================================================

def story_mannwhitney(
    group1: Union[pd.Series, np.ndarray],
    group2: Union[pd.Series, np.ndarray],
    name1: str = "Group A",
    name2: str = "Group B",
    alpha: float = 0.05,
    stats_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Narrative for the Mann–Whitney U test.
    """
    if _silent():
        return ""

    # 1) Compute statistics
    res = stats_result or il_stats.mannwhitney_test(group1, group2)

    # 2) Extract values
    p = float(res.get("p_value", np.nan))
    u = float(res.get("statistic", np.nan))
    r_bis = float(res.get("rank_biserial_r", np.nan))
    magnitude = res.get("magnitude", "unknown")

    sig_status = _interpret_significance(p, alpha)

    median1 = float(np.median(group1))
    median2 = float(np.median(group2))

    # 3) UI
    section(f"Mann–Whitney U Test: {name1} vs {name2}")

    # 4) Narrative
    descriptive = (
        f"{name1}: median ≈ {median1:.3f}. "
        f"{name2}: median ≈ {median2:.3f}. "
        "Non-parametric comparison of distributions."
    )

    statistical = (
        f"U-statistic = {u:.1f}, {_format_pvalue(p, alpha)}. "
        f"Rank-biserial r = {r_bis:.3f} ({magnitude}). "
        f"Distributions are {sig_status}ly different."
    )

    behavioural = (
        f"{name1} shows {'higher' if median1 > median2 else 'lower'} central tendency than {name2}."
        if p < alpha
        else f"No systematic difference in distributions between {name1} and {name2}."
    )

    strategic = "This test is robust to outliers and non-normality — use it when t-test assumptions fail."

    operational = (
        "Mann–Whitney tests whether one distribution is stochastically larger than the other. "
        "It does not assume normality or equal variances."
    )

    html = interpretation_5layers(
        descriptive=descriptive,
        statistical=statistical,
        behavioural=behavioural,
        strategic=strategic,
        operational=operational,
    )

    return html


def story_wilcoxon(
    before: Union[pd.Series, np.ndarray],
    after: Union[pd.Series, np.ndarray],
    name_before: str = "Before",
    name_after: str = "After",
    alpha: float = 0.05,
    stats_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Narrative for the Wilcoxon signed-rank test (paired, non-parametric).
    """
    if _silent():
        return ""

    # 1) Compute statistics
    res = stats_result or il_stats.wilcoxon_signed_test(before, after)

    # 2) Extract values
    p = float(res.get("p_value", np.nan))
    w = float(res.get("statistic", np.nan))
    r_bis = float(res.get("rank_biserial_r", np.nan))
    magnitude = res.get("magnitude", "unknown")

    sig_status = _interpret_significance(p, alpha)

    median_before = float(np.median(before))
    median_after = float(np.median(after))

    # 3) UI
    section(f"Wilcoxon Signed-Rank Test: {name_before} vs {name_after}")

    # 4) Narrative
    descriptive = (
        f"{name_before}: median ≈ {median_before:.3f}. "
        f"{name_after}: median ≈ {median_after:.3f}. "
        "Non-parametric paired comparison."
    )

    statistical = (
        f"W-statistic = {w:.1f}, {_format_pvalue(p, alpha)}. "
        f"Rank-biserial r = {r_bis:.3f} ({magnitude}). "
        f"The change is {sig_status}."
    )

    behavioural = (
        f"Participants show a {'positive' if median_after > median_before else 'negative'} shift."
        if p < alpha
        else "No systematic change detected between time points."
    )

    strategic = (
        "Wilcoxon is robust to outliers and non-normal differences — "
        "use it when paired t-test assumptions fail."
    )

    operational = (
        "This test evaluates whether the median of differences is zero. "
        "It does not assume normality of differences."
    )

    html = interpretation_5layers(
        descriptive=descriptive,
        statistical=statistical,
        behavioural=behavioural,
        strategic=strategic,
        operational=operational,
    )

    return html


def story_kruskal_wallis(
    *groups,
    group_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    stats_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Narrative for the Kruskal–Wallis test (non-parametric ANOVA alternative).
    """
    if _silent():
        return ""

    # 1) Compute statistics
    res = stats_result or il_stats.kruskal_test(*groups)

    # 2) Extract values
    p = float(res.get("p_value", np.nan))
    h = float(res.get("statistic", np.nan))
    epsilon2 = float(res.get("epsilon_squared", np.nan))
    magnitude = res.get("magnitude", "unknown")

    sig_status = _interpret_significance(p, alpha)

    # 3) Group names
    if group_names is None:
        group_names = [f"Group {i+1}" for i in range(len(groups))]

    # 4) UI
    section(f"Kruskal–Wallis Test: {', '.join(group_names)}")

    # 5) Narrative
    medians_str = ", ".join([f"{name}: {np.median(g):.2f}" for name, g in zip(group_names, groups)])

    descriptive = f"Comparing {len(groups)} groups (medians): {medians_str}."

    statistical = (
        f"H-statistic = {h:.3f}, {_format_pvalue(p, alpha)}. "
        f"ε² = {epsilon2:.3f} ({magnitude}). "
        f"Distributions are {sig_status}ly different."
    )

    behavioural = (
        "At least one group shows a systematically different distribution."
        if p < alpha
        else "No meaningful differences detected across groups."
    )

    strategic = (
        "Follow up with Dunn's post-hoc test to identify which pairs differ."
        if p < alpha
        else "Groups appear homogeneous."
    )

    operational = (
        "Kruskal–Wallis is the non-parametric alternative to ANOVA. "
        "Use it when normality or equal-variance assumptions fail."
    )

    html = interpretation_5layers(
        descriptive=descriptive,
        statistical=statistical,
        behavioural=behavioural,
        strategic=strategic,
        operational=operational,
    )

    return html


# =============================================================================
# 6. ASSUMPTION TESTS
# =============================================================================

def story_normality_test(
    data: Union[pd.Series, np.ndarray],
    name: str = "Variable",
    method: str = "shapiro",
    alpha: float = 0.05,
    stats_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Narrative for a normality test.
    """
    if _silent():
        return ""

    # 1) Compute statistics
    if method == "shapiro":
        res = stats_result or il_stats.normality_shapiro(data)
        test_name = "Shapiro–Wilk"
    else:
        res = stats_result or il_stats.normality_ks(data)
        test_name = "Kolmogorov–Smirnov"

    # 2) Extract values
    p = float(res.get("p_value", np.nan))
    stat = float(res.get("statistic", np.nan))

    # 3) Interpretation (p >= alpha = consistent with normality)
    is_normal = p >= alpha

    # 4) UI
    section(f"{test_name} Normality Test: {name}")

    # 5) Simplified narrative
    content = f"""
    <p><strong>Test:</strong> {test_name}</p>
    <p><strong>Statistic:</strong> {stat:.4f}</p>
    <p><strong>P-value:</strong> {p:.4f}</p>
    <p><strong>Result:</strong> {'Data appears normally distributed' if is_normal else 'Data is NOT normally distributed'}</p>
    <p><strong>Interpretation:</strong>
    {'Parametric tests (t-test, ANOVA) are appropriate.' if is_normal else
     'Consider non-parametric alternatives (Mann–Whitney, Kruskal–Wallis) or data transformation (log, Box–Cox).'}
    </p>
    """

    html = box("neutral", "Normality Check", content)

    if not is_normal:
        math_insight("skewness")

    return html


def story_levene_test(
    *groups,
    group_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    stats_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Narrative for Levene's test (homogeneity of variances).
    """
    if _silent():
        return ""

    # 1) Compute statistics
    res = stats_result or il_stats.levene_test(*groups)

    # 2) Extract values
    p = float(res.get("p_value", np.nan))
    stat = float(res.get("statistic", np.nan))

    # 3) Interpretation (p >= alpha = variances can be treated as equal)
    equal_var = p >= alpha

    # 4) Group names
    if group_names is None:
        group_names = [f"Group {i+1}" for i in range(len(groups))]

    # 5) UI
    section(f"Levene's Test: {', '.join(group_names)}")

    # 6) Narrative
    content = f"""
    <p><strong>Test:</strong> Levene's Test for Homogeneity of Variance</p>
    <p><strong>Statistic:</strong> {stat:.4f}</p>
    <p><strong>P-value:</strong> {p:.4f}</p>
    <p><strong>Result:</strong> {'Variances are equal' if equal_var else 'Variances are NOT equal'}</p>
    <p><strong>Recommendation:</strong>
    {'Standard ANOVA and t-tests are appropriate.' if equal_var else
     'Use Welch t-test or Welch ANOVA for unequal variances.'}
    </p>
    """

    html = box("neutral", "Variance Homogeneity Check", content)

    return html


# =============================================================================
# 7. BAYESIAN TESTS
# =============================================================================

def story_bayesian_proportion(
    successes: int,
    trials: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    stats_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Narrative for a Bayesian proportion test.
    """
    if _silent():
        return ""

    # 1) Compute statistics
    res = stats_result or il_stats.bayesian_proportion(successes, trials, prior_alpha, prior_beta)

    # 2) Extract values
    posterior_mean = float(res.get("posterior_mean", np.nan))
    credible_interval = res.get("credible_interval", (np.nan, np.nan))
    prob_gt_half = float(res.get("prob_gt_half", np.nan))

    # 3) UI
    section("Bayesian Proportion Test")

    # 4) Narrative
    descriptive = (
        f"Observed {successes} successes in {trials} trials. "
        f"Sample proportion: {successes/trials:.3f}."
    )

    statistical = (
        f"Posterior mean: {posterior_mean:.3f}. "
        f"95% credible interval: [{credible_interval[0]:.3f}, {credible_interval[1]:.3f}]. "
        f"Probability that the true proportion > 0.5: {prob_gt_half:.3f}."
    )

    behavioural = (
        f"There is a {prob_gt_half*100:.1f}% probability that the true success rate exceeds 50%."
        if prob_gt_half > 0.95
        else f"The success rate is likely around {posterior_mean*100:.1f}%."
    )

    strategic = (
        "Bayesian credible intervals provide direct probability statements about parameters — "
        "often more intuitive than frequentist confidence intervals."
    )

    operational = (
        "Use Bayesian methods when you have prior information or need probability statements. "
        "Update the posterior as new data arrives."
    )

    html = interpretation_5layers(
        descriptive=descriptive,
        statistical=statistical,
        behavioural=behavioural,
        strategic=strategic,
        operational=operational,
    )

    math_insight("bayesian")

    return html


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # T-tests
    "story_ttest_independent",
    "story_ttest_paired",
    "story_ttest_one_sample",
    # ANOVA
    "story_anova_oneway",
    # Chi-square
    "story_chi_square",
    # Correlation
    "story_correlation",
    # Non-parametric
    "story_mannwhitney",
    "story_wilcoxon",
    "story_kruskal_wallis",
    # Assumption tests
    "story_normality_test",
    "story_levene_test",
    # Bayesian
    "story_bayesian_proportion",
]

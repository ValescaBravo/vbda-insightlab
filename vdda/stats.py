"""
===============================================================================
INSIGHTLAB — STATISTICAL CORE MODULE (FIXED & ENHANCED)
===============================================================================
Numerical/statistical engine for InsightLab.

This module focuses ONLY on calculations:
    • Parametric tests (t-tests, ANOVA)
    • Non-parametric tests (Mann–Whitney, Kruskal–Wallis, Wilcoxon)
    • Normality tests (Shapiro–Wilk, Kolmogorov–Smirnov)
    • Assumption tests (Levene, Bartlett) ✨ NEW
    • Categorical association (Chi-square, Fisher, Cramér's V, Odds Ratio) ✨ ENHANCED
    • Correlation helpers (Pearson, Spearman, Kendall, pairwise matrix)
    • Effect sizes (Cohen's d, Hedges' g, Glass Δ, η² / partial η², Cohen's w) ✨ NEW
    • Bayesian proportion comparison (Beta–Binomial)
    • Multiple testing corrections (Bonferroni, Holm, FDR/BH)
    • Post-hoc tests (Tukey HSD, Dunn) ✨ NEW

CHANGES FROM ORIGINAL:
    ✅ Standardized all dict keys: "statistic" and "p_value" (not "stat", "pvalue")
    ✅ Added "magnitude" label to all effect size returns
    ✅ Added sample size warnings (n < 30 for parametric tests)
    ✅ Fixed pairwise_correlation to return DataFrame with standard columns
    ✅ Added assumption tests (Levene, Bartlett)
    ✅ Added post-hoc tests (Tukey HSD, Dunn)
    ✅ Added more effect sizes (odds ratio, risk ratio, Cohen's w)
    ✅ Improved docstrings with examples
    ✅ Better input validation

No HTML, no storytelling here.
Story translation is handled by story_stats.py & core.py.

Version: 4.1 (Fixed)
===============================================================================
"""

from __future__ import annotations

import warnings
from math import sqrt
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import (
    ttest_ind,
    ttest_rel,
    ttest_1samp,
    f_oneway,
    pearsonr,
    spearmanr,
    kendalltau,
    chi2_contingency,
    fisher_exact,
    beta,
    shapiro,
    kstest,
    mannwhitneyu,
    kruskal,
    wilcoxon,
    norm,
    levene,  # ✨ NEW
    bartlett,  # ✨ NEW
)

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


# =============================================================================
# 0. INTERNAL HELPERS
# =============================================================================

def _to_array(x: ArrayLike) -> np.ndarray:
    """Convert input to 1D float array and drop NaNs."""
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float)
    else:
        arr = np.asarray(x, dtype=float)
    return arr[~np.isnan(arr)]


def _check_two_groups(a: ArrayLike, b: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Convert two groups to clean arrays."""
    g1 = _to_array(a)
    g2 = _to_array(b)
    if len(g1) == 0 or len(g2) == 0:
        raise ValueError("Both groups must contain at least one non-NaN value.")
    return g1, g2


def _warn_small_sample(n: int, threshold: int = 30, test_name: str = "test"):
    """✅ NEW: Advertir si el tamaño de muestra es pequeño."""
    if n < threshold:
        warnings.warn(
            f"[{test_name}] Small sample size (n={n}). "
            f"Consider non-parametric alternatives or interpret with caution.",
            UserWarning
        )


# =============================================================================
# 1. EFFECT SIZE HELPERS
# =============================================================================

def cohens_d(group1: ArrayLike, group2: ArrayLike, paired: bool = False) -> float:
    """
    Cohen's d effect size.

    Independent design:
        d = (mean1 - mean2) / pooled_sd

    Paired design:
        d_z = mean(diff) / sd(diff)
    """
    g1, g2 = _check_two_groups(group1, group2)

    if paired:
        if len(g1) != len(g2):
            raise ValueError("Paired design requires both groups to have same length.")
        diff = g1 - g2
        return float(diff.mean() / diff.std(ddof=1))

    n1, n2 = len(g1), len(g2)
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    pooled = sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((g1.mean() - g2.mean()) / pooled)


def hedges_g(group1: ArrayLike, group2: ArrayLike, paired: bool = False) -> float:
    """
    Hedges' g (small-sample corrected Cohen's d).
    """
    d = cohens_d(group1, group2, paired=paired)
    g1, g2 = _check_two_groups(group1, group2)
    n = len(g1) + len(g2)
    # J correction factor
    J = 1.0 - (3.0 / (4.0 * n - 9.0))
    return float(d * J)


def glass_delta(group_treated: ArrayLike, group_control: ArrayLike) -> float:
    """
    Glass Δ: difference in means divided by control SD.
    """
    treated, control = _check_two_groups(group_treated, group_control)
    s_control = control.std(ddof=1)
    return float((treated.mean() - control.mean()) / s_control)


def eta_squared_from_anova(F: float, df_between: int, df_within: int) -> float:
    """
    Eta-squared for one-way ANOVA using F and degrees of freedom:
        η² = (F * df_between) / (F * df_between + df_within)
    """
    return float((F * df_between) / (F * df_between + df_within))


def partial_eta_squared(F: float, df_effect: int, df_error: int) -> float:
    """
    Partial eta-squared:
        η²_partial = (F * df_effect) / (F * df_effect + df_error)
    For one-way ANOVA, this equals η².
    """
    return float((F * df_effect) / (F * df_effect + df_error))


def effect_size_label(d: float) -> str:
    """
    Map |effect size| to qualitative label.
    Cohen's conventional thresholds:
        0.0–0.2   → negligible
        0.2–0.5   → small
        0.5–0.8   → medium
        >0.8      → large
    """
    a = abs(float(d))
    if a < 0.2:
        return "negligible"
    elif a < 0.5:
        return "small"
    elif a < 0.8:
        return "medium"
    else:
        return "large"


def cramers_v(table: pd.DataFrame) -> float:
    """Cramér's V effect size for categorical association."""
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    r, c = table.shape
    return float(sqrt(chi2 / (n * (min(r, c) - 1))))


def cohens_w(table: pd.DataFrame) -> float:
    """
    ✨ NEW: Cohen's w effect size for chi-square test.
    
    w = sqrt(χ² / N)
    
    Thresholds:
        0.1 = small
        0.3 = medium
        0.5 = large
    """
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    return float(sqrt(chi2 / n))


def odds_ratio(table_2x2: pd.DataFrame) -> float:
    """
    ✨ NEW: Odds ratio for 2x2 contingency table.
    
    OR = (a*d) / (b*c)
    
    where table is:
        [[a, b],
         [c, d]]
    """
    arr = table_2x2.to_numpy()
    if arr.shape != (2, 2):
        raise ValueError("Odds ratio requires a 2x2 table.")
    
    a, b = arr[0, :]
    c, d = arr[1, :]
    
    # Avoid division by zero
    if b == 0 or c == 0:
        return np.inf if (a * d) > 0 else 0.0
    
    return float((a * d) / (b * c))


def risk_ratio(table_2x2: pd.DataFrame) -> float:
    """
    ✨ NEW: Risk ratio (relative risk) for 2x2 contingency table.
    
    RR = (a/(a+b)) / (c/(c+d))
    """
    arr = table_2x2.to_numpy()
    if arr.shape != (2, 2):
        raise ValueError("Risk ratio requires a 2x2 table.")
    
    a, b = arr[0, :]
    c, d = arr[1, :]
    
    p1 = a / (a + b)
    p2 = c / (c + d)
    
    if p2 == 0:
        return np.inf if p1 > 0 else 1.0
    
    return float(p1 / p2)


# =============================================================================
# 2. NORMALITY TESTS
# =============================================================================

def normality_shapiro(x: ArrayLike) -> Dict[str, Any]:
    """
    Shapiro–Wilk normality test.

    Returns:
        {
            'test': 'shapiro',
            'n': n,
            'statistic': W,  # ✅ FIX: was 'W'
            'p_value': p,
        }
    """
    arr = _to_array(x)
    stat, p = shapiro(arr)
    return {
        "test": "shapiro",
        "n": int(len(arr)),
        "statistic": float(stat),  # ✅ FIX: standardized key name
        "p_value": float(p),
    }


def normality_ks(x: ArrayLike) -> Dict[str, Any]:
    """
    Kolmogorov–Smirnov test against normal with sample mean & std.

    Returns:
        {
            'test': 'ks',
            'n': n,
            'statistic': D,  # ✅ FIX: was 'D'
            'p_value': p,
        }
    """
    arr = _to_array(x)
    if len(arr) < 2:
        raise ValueError("KS test requires at least 2 observations.")

    mu = arr.mean()
    sigma = arr.std(ddof=1)
    if sigma == 0:
        raise ValueError("KS test not defined for zero variance sample.")

    # Standardize and test against standard normal
    z = (arr - mu) / sigma
    stat, p = kstest(z, "norm")

    return {
        "test": "ks",
        "n": int(len(arr)),
        "statistic": float(stat),  # ✅ FIX: standardized
        "p_value": float(p),
    }


# =============================================================================
# 3. ASSUMPTION TESTS (✨ NEW)
# =============================================================================

def levene_test(*groups: ArrayLike, center: str = "median") -> Dict[str, Any]:
    """
    ✨ NEW: Levene test for homogeneity of variances.
    
    H0: All groups have equal variances.
    
    Args:
        *groups: Variable number of groups
        center: 'median' (default, more robust) or 'mean'
    
    Returns:
        {
            'test': 'levene',
            'k': number of groups,
            'ns': sample sizes,
            'statistic': W,
            'p_value': p,
        }
    
    Example:
        >>> result = levene_test(group1, group2, group3)
        >>> if result['p_value'] < 0.05:
        ...     print("Variances differ significantly")
    """
    clean = [_to_array(g) for g in groups]
    
    stat, p = levene(*clean, center=center)
    
    return {
        "test": "levene",
        "k": len(clean),
        "ns": [len(g) for g in clean],
        "statistic": float(stat),
        "p_value": float(p),
    }


def bartlett_test(*groups: ArrayLike) -> Dict[str, Any]:
    """
    ✨ NEW: Bartlett test for homogeneity of variances.
    
    More sensitive to departures from normality than Levene.
    Use Levene if you suspect non-normality.
    
    H0: All groups have equal variances.
    
    Returns:
        {
            'test': 'bartlett',
            'k': number of groups,
            'ns': sample sizes,
            'statistic': T,
            'p_value': p,
        }
    """
    clean = [_to_array(g) for g in groups]
    
    stat, p = bartlett(*clean)
    
    return {
        "test": "bartlett",
        "k": len(clean),
        "ns": [len(g) for g in clean],
        "statistic": float(stat),
        "p_value": float(p),
    }


# =============================================================================
# 4. PARAMETRIC TESTS (T-TESTS, ANOVA)
# =============================================================================

def ttest_one_sample(x: ArrayLike, popmean: float = 0.0) -> Dict[str, Any]:
    """
    One-sample t-test: is the mean of x different from popmean?

    Returns dict with:
        test, type, n, mean, popmean, statistic, p_value, df,
        cohen_d, magnitude  # ✅ FIX: added magnitude
    """
    arr = _to_array(x)
    n = len(arr)
    
    # ✅ NEW: Warn if small sample
    _warn_small_sample(n, test_name="t-test (one-sample)")
    
    stat, p = ttest_1samp(arr, popmean)
    df = n - 1
    
    # ✅ FIX: Calculate effect size
    d = float((arr.mean() - popmean) / arr.std(ddof=1))

    return {
        "test": "t-test",
        "type": "one-sample",
        "n": int(n),
        "mean": float(arr.mean()),
        "popmean": float(popmean),
        "statistic": float(stat),  # ✅ FIX: was inconsistent
        "p_value": float(p),
        "df": int(df),
        "cohen_d": float(d),
        "magnitude": effect_size_label(d),  # ✅ FIX: added
    }


def ttest_independent(
    group1: ArrayLike,
    group2: ArrayLike,
    equal_var: bool = True,
) -> Dict[str, Any]:
    """
    Two-sample independent t-test.

    Returns dict with:
        test, type, n1, n2, mean1, mean2, statistic, p_value, df,
        cohen_d, hedges_g, magnitude  # ✅ FIX: standardized
    """
    g1, g2 = _check_two_groups(group1, group2)
    n1, n2 = len(g1), len(g2)
    
    # ✅ NEW: Warn if either sample is small
    if n1 < 30:
        _warn_small_sample(n1, test_name="t-test (group 1)")
    if n2 < 30:
        _warn_small_sample(n2, test_name="t-test (group 2)")
    
    stat, p = ttest_ind(g1, g2, equal_var=equal_var)
    df = n1 + n2 - 2

    d = cohens_d(g1, g2, paired=False)
    g = hedges_g(g1, g2, paired=False)

    return {
        "test": "t-test",
        "type": "independent",
        "n1": int(n1),
        "n2": int(n2),
        "mean1": float(g1.mean()),
        "mean2": float(g2.mean()),
        "statistic": float(stat),  # ✅ FIX: was "statistic"
        "p_value": float(p),
        "df": int(df),
        "cohen_d": float(d),
        "hedges_g": float(g),
        "magnitude": effect_size_label(d),  # ✅ FIX: was "effect_label"
    }


def ttest_paired(
    group1: ArrayLike,
    group2: ArrayLike,
) -> Dict[str, Any]:
    """
    Paired-samples t-test.

    Returns dict with:
        test, type, n, mean_diff, statistic, p_value, df,
        cohen_d_paired, magnitude
    """
    g1, g2 = _check_two_groups(group1, group2)
    if len(g1) != len(g2):
        raise ValueError("Paired t-test requires equal-length groups.")

    n = len(g1)
    
    # ✅ NEW: Warn if small sample
    _warn_small_sample(n, test_name="t-test (paired)")
    
    stat, p = ttest_rel(g1, g2)
    diff = g1 - g2
    d_paired = cohens_d(g1, g2, paired=True)

    return {
        "test": "t-test",
        "type": "paired",
        "n": int(n),
        "mean_diff": float(diff.mean()),
        "statistic": float(stat),
        "p_value": float(p),
        "df": int(n - 1),
        "cohen_d_paired": float(d_paired),
        "magnitude": effect_size_label(d_paired),  # ✅ FIX: was "effect_label"
    }


def anova_oneway(*groups: ArrayLike) -> Dict[str, Any]:
    """
    One-way ANOVA across k groups.

    Returns dict with:
        test, k, ns, statistic, p_value, df_between, df_within,
        eta2, partial_eta2, magnitude  # ✅ FIX: added magnitude
    """
    clean_groups = [_to_array(g) for g in groups]
    if len(clean_groups) < 2:
        raise ValueError("ANOVA requires at least two groups.")

    stat, p = f_oneway(*clean_groups)

    ns = [len(g) for g in clean_groups]
    k = len(clean_groups)
    N = sum(ns)
    df_between = k - 1
    df_within = N - k

    eta2 = eta_squared_from_anova(stat, df_between, df_within)
    p_eta2 = partial_eta_squared(stat, df_between, df_within)

    return {
        "test": "anova",
        "k": int(k),
        "ns": ns,
        "statistic": float(stat),  # ✅ FIX: was "F"
        "p_value": float(p),
        "df_between": int(df_between),
        "df_within": int(df_within),
        "eta2": float(eta2),
        "partial_eta2": float(p_eta2),
        "magnitude": effect_size_label(sqrt(eta2)),  # ✅ FIX: added
    }


# Backwards-compatible alias for lazy-loading
def ttest(*args, **kwargs) -> Dict[str, Any]:
    """Alias for independent two-sample t-test."""
    return ttest_independent(*args, **kwargs)


def anova(*groups: ArrayLike) -> Dict[str, Any]:
    """Alias for anova_oneway."""
    return anova_oneway(*groups)

# =============================================================================
# 4. REPEATED-MEASURES ANOVA (✨ NEW)
# =============================================================================

def anova_repeated(
    df: pd.DataFrame,
    subject: str,
    within: Union[str, List[str]],
    dv: str,
) -> Dict[str, Any]:
    """
    Repeated-measures ANOVA using statsmodels.AnovaRM.
    
    Args:
        df: DataFrame en formato largo (una fila por observación).
        subject: nombre de la columna con el identificador de sujeto.
        within: nombre (str) o lista de nombres de factores intra-sujeto.
        dv: nombre de la variable dependiente.
    
    Returns:
        dict con:
            - test: "anova_repeated"
            - n_subjects: número de sujetos únicos
            - effects: lista de dicts por efecto con:
                {effect, F, df_effect, df_error, p_value, partial_eta2, magnitude}
            - table: DataFrame crudo de AnovaRM (por si se quiere inspeccionar)
    
    Notas:
        - No imprime nada (silent by design).
        - Requiere statsmodels:
            pip install statsmodels
    """
    try:
        from statsmodels.stats.anova import AnovaRM
    except ImportError:
        raise ImportError(
            "Repeated-measures ANOVA requires statsmodels. "
            "Install with: pip install statsmodels"
        )
    
    # Normalizar within a lista
    if isinstance(within, str):
        within_factors = [within]
    else:
        within_factors = list(within)
    
    # Columnas necesarias
    needed_cols = [subject, dv] + within_factors
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # Limpiar filas con NaN en variables relevantes
    df_clean = df[needed_cols].dropna()
    
    if df_clean.empty:
        raise ValueError("No rows left after dropping NaNs for repeated-measures ANOVA.")
    
    # Ajustar modelo de medidas repetidas
    aov = AnovaRM(
        df_clean,
        depvar=dv,
        subject=subject,
        within=within_factors
    ).fit()
    
    # Tabla de resultados de statsmodels
    table = aov.anova_table.reset_index().rename(
        columns={
            "index": "effect",
            "F Value": "F",
            "Pr > F": "p_value",
        }
    )
    
    effects = []
    for _, row in table.iterrows():
        effect = str(row.get("effect", ""))
        
        # Suele haber fila "Residual" que no nos interesa como efecto principal
        if effect.lower() == "residual":
            continue
        
        F = float(row.get("F", np.nan))
        df_effect = int(row.get("Num DF", 0))
        df_error = int(row.get("Den DF", 0))
        p = float(row.get("p_value", np.nan))
        
        # Partial eta-squared usando helper ya definido en este módulo
        eta2_p = partial_eta_squared(F, df_effect, df_error) if df_error > 0 else np.nan
        mag = effect_size_label(sqrt(max(0.0, eta2_p))) if not np.isnan(eta2_p) else "unknown"
        
        effects.append(
            {
                "effect": effect,
                "F": F,
                "df_effect": df_effect,
                "df_error": df_error,
                "p_value": p,
                "partial_eta2": float(eta2_p) if not np.isnan(eta2_p) else np.nan,
                "magnitude": mag,
            }
        )
    
    n_subjects = df_clean[subject].nunique()
    
    return {
        "test": "anova_repeated",
        "n_subjects": int(n_subjects),
        "effects": effects,
        "table": table,
    }

# =============================================================================
# 5. POST-HOC TESTS (✨ NEW)
# =============================================================================

def tukey_hsd(*groups: ArrayLike, alpha: float = 0.05) -> pd.DataFrame:
    """
    ✨ NEW: Tukey HSD post-hoc test for ANOVA.
    
    Identifies which pairs of groups differ significantly.
    
    Args:
        *groups: Variable number of groups
        alpha: Significance level
    
    Returns:
        DataFrame with columns:
            ['group1', 'group2', 'mean_diff', 'statistic', 'p_value', 'reject']
    
    Example:
        >>> result = tukey_hsd(group_a, group_b, group_c)
        >>> print(result[result['reject']])
    """
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError:
        raise ImportError(
            "Tukey HSD requires statsmodels. "
            "Install with: pip install statsmodels"
        )
    
    # Prepare data in long format
    clean_groups = [_to_array(g) for g in groups]
    
    data = []
    labels = []
    for i, g in enumerate(clean_groups):
        data.extend(g)
        labels.extend([f"Group_{i+1}"] * len(g))
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Run Tukey HSD
    tukey_result = pairwise_tukeyhsd(data, labels, alpha=alpha)
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'group1': tukey_result.groupsunique[tukey_result._results_table.data[1:, 0]],
        'group2': tukey_result.groupsunique[tukey_result._results_table.data[1:, 1]],
        'mean_diff': tukey_result._results_table.data[1:, 2],
        'p_value': tukey_result._results_table.data[1:, 3],
        'lower_ci': tukey_result._results_table.data[1:, 4],
        'upper_ci': tukey_result._results_table.data[1:, 5],
        'reject': tukey_result._results_table.data[1:, 6],
    })
    
    return df


def dunn_test(*groups: ArrayLike, alpha: float = 0.05) -> pd.DataFrame:
    """
    ✨ NEW: Dunn's test (post-hoc for Kruskal-Wallis).
    
    Non-parametric post-hoc test that compares pairs of groups
    using rank-based methods.
    
    Args:
        *groups: Variable number of groups
        alpha: Significance level
    
    Returns:
        DataFrame with columns:
            ['group1', 'group2', 'statistic', 'p_value', 'p_value_adjusted', 'reject']
    
    Example:
        >>> result = dunn_test(group_a, group_b, group_c)
        >>> print(result[result['reject']])
    """
    try:
        from scikit_posthocs import posthoc_dunn
    except ImportError:
        raise ImportError(
            "Dunn test requires scikit-posthocs. "
            "Install with: pip install scikit-posthocs"
        )
    
    # Prepare data in long format
    clean_groups = [_to_array(g) for g in groups]
    
    data = []
    labels = []
    for i, g in enumerate(clean_groups):
        data.extend(g)
        labels.extend([f"Group_{i+1}"] * len(g))
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Create DataFrame
    df_input = pd.DataFrame({'data': data, 'group': labels})
    
    # Run Dunn test
    result_matrix = posthoc_dunn(df_input, val_col='data', group_col='group', p_adjust='bonferroni')
    
    # Convert matrix to long format
    pairs = []
    for i in range(len(result_matrix)):
        for j in range(i + 1, len(result_matrix)):
            pairs.append({
                'group1': result_matrix.index[i],
                'group2': result_matrix.columns[j],
                'p_value_adjusted': result_matrix.iloc[i, j],
                'reject': result_matrix.iloc[i, j] < alpha
            })
    
    return pd.DataFrame(pairs)


# =============================================================================
# 6. NON-PARAMETRIC TESTS
# =============================================================================

def mannwhitney_test(
    group1: ArrayLike,
    group2: ArrayLike,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """
    Mann–Whitney U test (independent samples, non-parametric).

    Effect size:
        Rank-biserial correlation r_rb = 1 - 2U / (n1*n2)
        
    Returns:
        ✅ FIX: standardized keys with magnitude label
    """
    g1, g2 = _check_two_groups(group1, group2)
    n1, n2 = len(g1), len(g2)

    stat, p = mannwhitneyu(g1, g2, alternative=alternative)

    r_rb = 1.0 - (2.0 * stat) / (n1 * n2)

    return {
        "test": "mannwhitney",
        "alternative": alternative,
        "n1": int(n1),
        "n2": int(n2),
        "statistic": float(stat),  # ✅ FIX: was "U"
        "p_value": float(p),
        "rank_biserial_r": float(r_rb),
        "magnitude": effect_size_label(r_rb),  # ✅ FIX: added
    }


def wilcoxon_signed_test(
    before: ArrayLike,
    after: ArrayLike,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test (paired non-parametric).

    Returns:
        test, alternative, n, statistic, p_value
        ✅ FIX: standardized keys
    """
    b, a = _check_two_groups(before, after)
    if len(b) != len(a):
        raise ValueError("Wilcoxon signed-rank test requires equal-length pairs.")

    stat, p = wilcoxon(b, a, alternative=alternative)

    return {
        "test": "wilcoxon",
        "alternative": alternative,
        "n": int(len(b)),
        "statistic": float(stat),  # ✅ FIX: was "W"
        "p_value": float(p),
    }


def kruskal_test(*groups: ArrayLike) -> Dict[str, Any]:
    """
    Kruskal–Wallis H-test (non-parametric ANOVA).

    Effect size approximation:
        epsilon² = (H - k + 1) / (N - k)
        
    Returns:
        ✅ FIX: standardized keys with magnitude
    """
    clean = [_to_array(g) for g in groups]
    stat, p = kruskal(*clean)

    ns = [len(g) for g in clean]
    k = len(clean)
    N = sum(ns)

    epsilon2 = (stat - (k - 1)) / (N - k) if N > k else np.nan

    return {
        "test": "kruskal",
        "k": int(k),
        "ns": ns,
        "statistic": float(stat),  # ✅ FIX: was "H"
        "p_value": float(p),
        "epsilon2": float(epsilon2),
        "magnitude": effect_size_label(sqrt(abs(epsilon2))) if not np.isnan(epsilon2) else "unknown",  # ✅ FIX: added
    }


# =============================================================================
# 7. CATEGORICAL ASSOCIATION
# =============================================================================

def chi_square_test(table: pd.DataFrame) -> Dict[str, Any]:
    """
    Chi-square test of independence for contingency table.

    Returns:
        test, statistic, p_value, dof, expected, 
        cramers_v, cohens_w, magnitude  # ✅ FIX: added cohens_w and magnitude
    """
    chi2, p, dof, expected = chi2_contingency(table)
    v = cramers_v(table)
    w = cohens_w(table)  # ✨ NEW

    return {
        "test": "chi-square",
        "statistic": float(chi2),  # ✅ FIX: was "chi2"
        "p_value": float(p),
        "dof": int(dof),
        "expected": expected,
        "cramers_v": float(v),
        "cohens_w": float(w),  # ✨ NEW
        "magnitude": effect_size_label(w),  # ✅ FIX: added
    }


def fisher_test(table_2x2: pd.DataFrame) -> Dict[str, Any]:
    """
    Fisher's exact test for 2x2 contingency table.

    Returns:
        test, oddsratio, p_value
        ✅ FIX: standardized key names
    """
    arr = table_2x2.to_numpy()
    if arr.shape != (2, 2):
        raise ValueError("Fisher's exact test requires a 2x2 table.")

    odds, p = fisher_exact(arr)

    return {
        "test": "fisher",
        "statistic": float(odds),  # ✅ FIX: was "oddsratio"
        "p_value": float(p),
    }


# Backwards-compatible alias
def chi_square(table: pd.DataFrame) -> Dict[str, Any]:
    """Alias for chi_square_test."""
    return chi_square_test(table)


# =============================================================================
# 8. CORRELATION HELPERS
# =============================================================================

def correlation(
    x: ArrayLike,
    y: ArrayLike,
    method: str = "pearson",
) -> Dict[str, Any]:
    """
    Compute correlation between two variables.

    method ∈ {'pearson', 'spearman', 'kendall'}
    
    Returns:
        ✅ FIX: added magnitude label
    """
    x_arr = _to_array(x)
    y_arr = _to_array(y)
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length.")

    method = method.lower()
    if method == "pearson":
        r, p = pearsonr(x_arr, y_arr)
    elif method == "spearman":
        r, p = spearmanr(x_arr, y_arr)
    elif method == "kendall":
        r, p = kendalltau(x_arr, y_arr)
    else:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'.")

    return {
        "test": "correlation",
        "method": method,
        "n": int(len(x_arr)),
        "statistic": float(r),  # ✅ FIX: was "r"
        "p_value": float(p),
        "magnitude": effect_size_label(r),  # ✅ FIX: added
    }


def pairwise_correlation(
    df: pd.DataFrame,
    method: str = "pearson",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    ✅ FIXED: Now returns DataFrame with standard columns.
    
    Compute pairwise correlation matrix and return as long-format DataFrame.

    Args:
        df: Input DataFrame
        method: 'pearson', 'spearman', or 'kendall'
        threshold: If provided, only return pairs with |r| >= threshold
    
    Returns:
        DataFrame with columns:
            ['var1', 'var2', 'coef', 'p_value', 'magnitude']
    
    Example:
        >>> result = pairwise_correlation(df, method='pearson', threshold=0.5)
        >>> print(result)
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for pairwise correlation")
    
    cols = numeric_df.columns.tolist()
    
    results = []
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            var1, var2 = cols[i], cols[j]
            
            # Compute correlation
            corr_result = correlation(
                numeric_df[var1].dropna(),
                numeric_df[var2].dropna(),
                method=method
            )
            
            coef = corr_result['statistic']
            pval = corr_result['p_value']
            mag = corr_result['magnitude']
            
            # Apply threshold if specified
            if threshold is None or abs(coef) >= threshold:
                results.append({
                    'var1': var1,
                    'var2': var2,
                    'coef': coef,
                    'p_value': pval,
                    'magnitude': mag
                })
    
    # ✅ FIX: Always return DataFrame with correct columns, even if empty
    df_result = pd.DataFrame(results)
    
    # If empty, return empty DataFrame with correct structure
    if df_result.empty:
        df_result = pd.DataFrame(columns=['var1', 'var2', 'coef', 'p_value', 'magnitude'])
    
    return df_result


# =============================================================================
# 9. BAYESIAN PROPORTION COMPARISON
# =============================================================================

def bayesian_proportion(
    success_a: int,
    total_a: int,
    success_b: int,
    total_b: int,
    prior_a: Tuple[float, float] = (1.0, 1.0),
    prior_b: Tuple[float, float] = (1.0, 1.0),
    credible_interval: float = 0.95,
    n_samples: int = 50000,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare two proportions A and B using a Beta–Binomial Bayesian model.

    Posterior:
        p_A ~ Beta(alpha_a + success_a, beta_a + total_a - success_a)
        p_B ~ Beta(alpha_b + success_b, beta_b + total_b - success_b)

    Returns:
        {
            'model': 'bayesian_proportion',
            'posterior_mean_a',
            'posterior_mean_b',
            'mean_diff',
            'ci_low',
            'ci_high',
            'prob_diff_gt_0',  # P(p_A - p_B > 0)
            'samples': n_samples,
        }
    """
    if total_a <= 0 or total_b <= 0:
        raise ValueError("Totals must be positive for Bayesian proportion test.")

    rng = np.random.default_rng(seed=random_state)

    alpha_a = prior_a[0] + success_a
    beta_a_ = prior_a[1] + (total_a - success_a)

    alpha_b = prior_b[0] + success_b
    beta_b_ = prior_b[1] + (total_b - success_b)

    samples_a = rng.beta(alpha_a, beta_a_, size=n_samples)
    samples_b = rng.beta(alpha_b, beta_b_, size=n_samples)

    diff = samples_a - samples_b

    mean_a = float(samples_a.mean())
    mean_b = float(samples_b.mean())
    mean_diff = float(diff.mean())
    ci_low, ci_high = np.quantile(diff, [(1 - credible_interval) / 2, 1 - (1 - credible_interval) / 2])
    prob_diff_gt_0 = float(np.mean(diff > 0))

    return {
        "model": "bayesian_proportion",
        "posterior_mean_a": mean_a,
        "posterior_mean_b": mean_b,
        "mean_diff": mean_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "prob_diff_gt_0": prob_diff_gt_0,
        "samples": int(n_samples),
    }


# Backwards-compatible alias
def bayesian(*args, **kwargs) -> Dict[str, Any]:
    """Alias to bayesian_proportion for lazy-loading."""
    return bayesian_proportion(*args, **kwargs)


# =============================================================================
# 10. MULTIPLE TESTING CORRECTIONS
# =============================================================================

def adjust_pvalues(
    p_values: Sequence[float],
    method: str = "fdr_bh",
) -> Dict[str, Any]:
    """
    Adjust p-values for multiple testing.

    Supported methods:
        • 'bonferroni'  — Bonferroni correction
        • 'holm'        — Holm–Bonferroni step-down
        • 'fdr_bh'      — Benjamini–Hochberg FDR

    Returns:
        {
            'method': method,
            'original': np.ndarray,
            'adjusted': np.ndarray,
        }
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be a 1D sequence of floats.")

    m = float(len(p))
    if m == 0:
        return {"method": method, "original": p, "adjusted": p}

    method = method.lower()

    if method == "bonferroni":
        padj = np.minimum(p * m, 1.0)

    elif method == "holm":
        order = np.argsort(p)
        ranked = p[order]
        adj = np.empty_like(ranked)
        # Step-down
        for i, pi in enumerate(ranked):
            adj[i] = (m - i) * pi
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        adj = np.minimum(adj, 1.0)
        padj = np.empty_like(adj)
        padj[order] = adj

    elif method == "fdr_bh":
        order = np.argsort(p)
        ranked = p[order]
        m_int = len(ranked)
        adj = np.empty_like(ranked)
        # Benjamini–Hochberg
        for i in range(m_int, 0, -1):
            adj[i - 1] = min((m * ranked[i - 1]) / i, 1.0)
            if i < m_int:
                adj[i - 1] = min(adj[i - 1], adj[i])
        padj = np.empty_like(adj)
        padj[order] = adj

    else:
        raise ValueError("method must be 'bonferroni', 'holm', or 'fdr_bh'.")

    return {
        "method": method,
        "original": p,
        "adjusted": padj,
    }


# =============================================================================
# 11. EFFECT SIZE FUNCTION (UNIFIED INTERFACE)
# =============================================================================

def effect_size(
    data1: ArrayLike,
    data2: Optional[ArrayLike] = None,
    method: str = "cohen",
    **kwargs
) -> Dict[str, Any]:
    """
    ✨ NEW: Unified effect size calculator.
    
    Automatically selects and computes appropriate effect size based on data.
    
    Args:
        data1: First group/variable
        data2: Second group/variable (optional for some methods)
        method: 'cohen', 'hedges', 'glass', 'cramers_v', 'cohens_w'
        **kwargs: Additional arguments (e.g., paired=True)
    
    Returns:
        {
            'method': method name,
            'value': effect size value,
            'magnitude': qualitative label,
        }
    
    Example:
        >>> es = effect_size(group1, group2, method='hedges')
        >>> print(f"Effect: {es['value']:.3f} ({es['magnitude']})")
    """
    method = method.lower()
    
    if method == 'cohen':
        if data2 is None:
            raise ValueError("Cohen's d requires two groups")
        paired = kwargs.get('paired', False)
        d = cohens_d(data1, data2, paired=paired)
        return {
            'method': "Cohen's d",
            'value': float(d),
            'magnitude': effect_size_label(d),
        }
    
    elif method == 'hedges':
        if data2 is None:
            raise ValueError("Hedges' g requires two groups")
        paired = kwargs.get('paired', False)
        g = hedges_g(data1, data2, paired=paired)
        return {
            'method': "Hedges' g",
            'value': float(g),
            'magnitude': effect_size_label(g),
        }
    
    elif method == 'glass':
        if data2 is None:
            raise ValueError("Glass' delta requires two groups (treated, control)")
        delta = glass_delta(data1, data2)
        return {
            'method': "Glass' Δ",
            'value': float(delta),
            'magnitude': effect_size_label(delta),
        }
    
    elif method in ['cramers_v', 'cramers']:
        if not isinstance(data1, pd.DataFrame):
            raise ValueError("Cramér's V requires a contingency table (DataFrame)")
        v = cramers_v(data1)
        return {
            'method': "Cramér's V",
            'value': float(v),
            'magnitude': effect_size_label(v),
        }
    
    elif method in ['cohens_w', 'cohens w']:
        if not isinstance(data1, pd.DataFrame):
            raise ValueError("Cohen's w requires a contingency table (DataFrame)")
        w = cohens_w(data1)
        return {
            'method': "Cohen's w",
            'value': float(w),
            'magnitude': effect_size_label(w),
        }
    
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            "Available: 'cohen', 'hedges', 'glass', 'cramers_v', 'cohens_w'"
        )


# =============================================================================
# 12. PUBLIC EXPORT LIST
# =============================================================================

__all__ = [
    # Effect sizes
    "cohens_d",
    "hedges_g",
    "glass_delta",
    "eta_squared_from_anova",
    "partial_eta_squared",
    "effect_size_label",
    "cramers_v",
    "cohens_w",  # ✨ NEW
    "odds_ratio",  # ✨ NEW
    "risk_ratio",  # ✨ NEW
    "effect_size",  # ✨ NEW

    # Normality
    "normality_shapiro",
    "normality_ks",

    # Assumption tests ✨ NEW
    "levene_test",
    "bartlett_test",

    # Parametric tests
    "ttest_one_sample",
    "ttest_independent",
    "ttest_paired",
    "anova_oneway",
    "anova_repeated",  # ✨ NEW
    "ttest",
    "anova",


    # Post-hoc tests ✨ NEW
    "tukey_hsd",
    "dunn_test",

    # Non-parametric tests
    "mannwhitney_test",
    "wilcoxon_signed_test",
    "kruskal_test",

    # Categorical association
    "chi_square_test",
    "fisher_test",
    "chi_square",

    # Correlation
    "correlation",
    "pairwise_correlation",

    # Bayesian proportion
    "bayesian_proportion",
    "bayesian",

    # Multiple testing
    "adjust_pvalues",
]
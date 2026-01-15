"""
===============================================================================
INSIGHTLAB â€” PUBLIC API
===============================================================================
Sistema analÃ­tico profesional y educativo integrado.

Lightweight analytical engine with:
     Global configuration (CONFIG)
     Lazy-loaded submodules (prep, stats, viz, ml, export, narrative)
     Visual identity (navy #1d085e, teal #13d6c1, white)
     Statistical storytelling framework
     HTML export with responsive design

Comentarios: EspaÃ±ol
Output: English

...
===============================================================================
"""

__version__ = "4.0.0"
__author__ = "Valesca Bravo"
__description__ = "Professional analytical and storytelling engine for data science"
__license__ = "MIT"

from typing import Any, Dict, List, Optional

# =============================================================================
# CORE IMPORTS (SIEMPRE DISPONIBLES)
# =============================================================================

from .core import (
    # Configuración global
    CONFIG,
    InsightLabConfig,
    InsightLabStyle,
    initialize,
    
    # Estilos
    load_style,
    apply_style,
    
    # Dominio
    load_domain,
    
    # UI Components
    section,
    box,
    divider,
    show_html,
    
    # Math Insights
    math_insight,
    CONCEPTS,
)

from .narrative.story import (
    # Frameworks narrativos
    narrative,
    narrative_from_dict,
    interpretation_5layers,
    layers,  # CORRECTO: alias de interpretation_5layers
    star,
    
    # Helpers
    explain,
    result,
)

# =============================================================================
# LAZY LOADING MAP (MóDULOS PESADOS)
# =============================================================================
# NOTA IMPORTANTE:
# sistema de lazy loading.
# Elimina cualquier __getattr__ duplicado en core.py 

import importlib as _importlib

_LAZY_MAP = {
        # -------------------------------------------------------------------------
    # PREP.PY (Data Preparation)
    # -------------------------------------------------------------------------
    "_log_prep_step": "prep",
    "detect_types": "prep",
    "get_numeric_cols": "prep",
    "get_categorical_cols": "prep",
    "get_datetime_cols": "prep",
    "validate_columns": "prep",
    "normalize_columns": "prep",
   
    # Duplicates
    "drop_duplicates": "prep",
    "flag_duplicates": "prep",
    # Missing
    "fill_missing": "prep",
    "flag_missing": "prep",
    # Outliers
    "outliers": "prep",
    "detect_outliers": "prep",
    "cap_outliers_iqr": "prep",
    "cap_outliers_percentile": "prep",
    "cap_outliers_zscore": "prep",
    "flag_outliers": "prep",
    # Scaling & transforms
    "scale_numeric": "prep",
    "log_transform": "prep",
    "power_transform": "prep",
    # Encoding
    "encode_onehot": "prep",
    "encode_ordinal": "prep",
    "encode_target": "prep",
    "encode_categoricals": "prep",
    # Date & discretisation
    "extract_date_features": "prep",
    "discretize_equal_width": "prep",
    "discretize_equal_freq": "prep",
    # Quality & filters
    "remove_constant_columns": "prep",
    "remove_high_missing": "prep",
    "data_quality": "prep",
    "get_data_quality_report": "prep",
    # High-level helpers
    "split_train_test": "prep",
    "prep_summary": "prep",
    "Cleaner": "prep",
    "PrepResult": "prep",

    
    # -------------------------------------------------------------------------
    # STATS.PY (Statistical Tests)
    # -------------------------------------------------------------------------
    # Descriptive
    "describe_numeric": "stats",
    "describe_categorical": "stats",
    
    # T-tests
    "ttest_independent": "stats",
    "ttest_paired": "stats",
    
    # ANOVA
    "anova_oneway": "stats",
    "anova_two_way": "stats",
    
    # Correlation
    "correlation_matrix": "stats",
    "correlation": "stats",
    "pairwise_correlation": "stats",
    
    # Chi-square
    "chi_square_test": "stats",
    
    # Non-parametric
    "mannwhitney_test": "stats",
    "wilcoxon_signed_test": "stats",
    "kruskal_test": "stats",
    
    # Normality
    "normality_shapiro": "stats",
    "normality_ks": "stats",
    
    # Homogeneity
    "levene_test": "stats",
    "bartlett_test": "stats",
    
    # Effect sizes
    "cohens_d": "stats",
    "hedges_g": "stats",
    "glass_delta": "stats",
    "cohens_w": "stats",
    "cramers_v": "stats",
    "phi_coefficient": "stats",
    "odds_ratio": "stats",
    "risk_ratio": "stats",
    "rank_biserial_r": "stats",
    "epsilon_squared": "stats",
    "eta_squared": "stats",
    "omega_squared": "stats",
    "effect_size": "stats",
    
    # Post-hoc
    "tukey_hsd": "stats",
    "dunn_test": "stats",
    
    # Bayesian
    "bayesian_proportion": "stats",
    
    # Backwards compatibility aliases
    "ttest": "stats",
    "anova": "stats",
    "chi_square": "stats",
    
    # -------------------------------------------------------------------------
    # VIZ.PY (Visualizations)
    # -------------------------------------------------------------------------
    "plot_distribution": "viz",
    "plot_box": "viz",
    "plot_scatter": "viz",
    "plot_correlation": "viz",
    "plot_time_series": "viz",
    "plot_categories": "viz",
    "plot_cluster": "viz",
    "plot_bar" : "viz",
    
    # -------------------------------------------------------------------------
    # ML.PY (Machine Learning)
    # -------------------------------------------------------------------------
    "analyse_pca": "ml",
    "plot_pca_variance_explained": "ml",
    "plot_pca_components": "ml",
    "train_tree_classifier": "ml",
    "train_random_forest": "ml",
    "train_logistic_regression": "ml",
    "train_linear_regression": "ml",
    "train_kmeans": "ml",
    "train_model": "ml",
    "evaluate_classification": "ml",
    "evaluate_regression": "ml",
    "evaluate_clustering": "ml",
    "plot_cluster_sizes": "ml",
    
    # -------------------------------------------------------------------------
    # EXPORT.PY (Export functions)
    # -------------------------------------------------------------------------
    "export_html": "export",
    "export_notebook": "export",
    "export_html_with_images": "export",
    
    # -------------------------------------------------------------------------
    # STORY_EXPLORE.PY (EDA Storytelling)
    # -------------------------------------------------------------------------
    "story_overview": "narrative.story_explore",
    "story_variable_types": "narrative.story_explore",
    "story_missing": "narrative.story_explore",
    "story_duplicates": "narrative.story_explore",
    "story_descriptive": "narrative.story_explore",
    "story_shape": "narrative.story_explore",
    "story_outliers": "narrative.story_explore",
    "story_correlation": "narrative.story_explore",  # Note: duplicate with story_stats
    "story_data_quality": "narrative.story_explore",

    # -------------------------------------------------------------------------
    # NARRATIVE/STORY_PREP.PY (Prep Storytelling)
    # -------------------------------------------------------------------------
    "story_prep": "narrative.story_prep",
    "prep_summary": "narrative.story_prep",

    # -------------------------------------------------------------------------
    # NARRATIVE/STORY_STATS.PY (Statistical Test Storytelling)
    # -------------------------------------------------------------------------
    "story_ttest_independent": "narrative.story_stats",
    "story_ttest_paired": "narrative.story_stats",
    "story_ttest_one_sample": "narrative.story_stats",
    "story_anova_oneway": "narrative.story_stats",
    "story_correlation": "narrative.story_stats",
    "story_chi_square": "narrative.story_stats",
    "story_mannwhitney": "narrative.story_stats",
    "story_wilcoxon": "narrative.story_stats",
    "story_kruskal_wallis": "narrative.story_stats",
    "story_normality_test": "narrative.story_stats",
    "story_levene_test": "narrative.story_stats",
    "story_bayesian_proportion": "narrative.story_stats",
    
    # -------------------------------------------------------------------------
    # NARRATIVE/STORY_ML.PY (ML Storytelling)
    # -------------------------------------------------------------------------
    "story_pca": "narrative.story_ml",
    "story_clusters": "narrative.story_ml",
    "story_feature_importance": "narrative.story_ml",
    "story_classification": "narrative.story_ml",
    "story_regression": "narrative.story_ml",
    "story_model_comparison": "narrative.story_ml",
    "story_forecast": "narrative.story_ml",
    "story_tree_rules": "narrative.story_ml",
    "story_confusion_matrix": "narrative.story_ml",
    "story_cross_validation": "narrative.story_ml",
    "story_learning_curve": "narrative.story_ml",
    "story_residual_analysis": "narrative.story_ml",
}


def __getattr__(name: str):
    """
    Lazy loading de funciones desde submÃ³dulos.
    
    Permite:
        import insightlab as il
        il.ttest_independent(...)  # Auto-importa desde stats.py
    
    Sin necesidad de importar manualmente cada submÃ³dulo.
    """
    if name in _LAZY_MAP:
        module_name = _LAZY_MAP[name]
        
        # Importar mÃ³dulo bajo demanda
        try:
            module = _importlib.import_module(f".{module_name}", __name__)
        except ImportError as exc:  # pragma: no cover
            raise AttributeError(
                f"[InsightLab] Failed to lazy-load module '{module_name}' "
                f"for attribute '{name}': {exc}"
            ) from exc
        
        # Obtener atributo del mÃ³dulo
        try:
            attr = getattr(module, name)
        except AttributeError as exc:  # pragma: no cover
            raise AttributeError(
                f"[InsightLab] Module '{module_name}' "
                f"does not define attribute '{name}'."
            ) from exc
        
        return attr
    
    raise AttributeError(
        f"[InsightLab] No attribute '{name}' in insightlab namespace. "
        f"Check if the function exists in the module documentation."
    )


# =============================================================================
# PUBLIC API SUMMARY
# =============================================================================


# =============================================================================
# HIGH-LEVEL FACADE FUNCTIONS
# =============================================================================

def quick_explore(
    df,
    *,
    include_descriptive: bool = True,
    include_shape: bool = True,
    include_outliers: bool = True,
    include_correlation: bool = True,
    include_quality: bool = True,
) -> Dict[str, str]:
    """
    High-level EDA helper.

    Ejecuta un EDA narrativo estÃ¡ndar usando story_explore.py:
        - Overview (shape, missing, duplicates)
        - Variable types
        - Missing values
        - Duplicates
        - (Opcional) Descriptive statistics
        - (Opcional) Distribution shape (skewness / kurtosis)
        - (Opcional) Outliers
        - (Opcional) Correlations
        - (Opcional) Data quality summary

    Respeta CONFIG.verbosity:
        - "silent": no renderiza nada y retorna dict vacÃ­o.
        - "stakeholder" / "technical": muestra cajas HTML en Jupyter.

    Args:
        df: DataFrame a analizar.
        include_descriptive: incluir historia de estadÃ­sticas descriptivas.
        include_shape: incluir historia de skewness/kurtosis.
        include_outliers: incluir historia de outliers.
        include_correlation: incluir historia de correlaciones numÃ©ricas.
        include_quality: incluir historia de calidad de datos.

    Returns:
        dict con HTMLs de cada secciÃ³n para uso programÃ¡tico.
    """
    if CONFIG.verbosity == "silent":
        return {}

    # Importar solo cuando se usa (mantener paquete ligero)
    from .narrative import story_explore as _story_explore

    stories: Dict[str, str] = {}

    stories["overview"] = _story_explore.story_overview(df)
    stories["variable_types"] = _story_explore.story_variable_types(df)
    stories["missing"] = _story_explore.story_missing(df)
    stories["duplicates"] = _story_explore.story_duplicates(df)

    if include_descriptive:
        stories["descriptive"] = _story_explore.story_descriptive(df)
    if include_shape:
        stories["shape"] = _story_explore.story_shape(df)
    if include_outliers:
        stories["outliers"] = _story_explore.story_outliers(df)
    if include_correlation:
        stories["correlation"] = _story_explore.story_correlation(df)
    if include_quality:
        stories["data_quality"] = _story_explore.story_data_quality(df)

    return stories


def quick_compare_groups(
    df,
    value_col: str,
    group_col: str,
    *,
    test: Optional[str] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Quick group comparison with narrative output.

    Flujo:
        1. Construye grupos a partir de df[group_col] para value_col.
        2. Si test es None, selecciona automÃ¡ticamente:
            - 2 grupos: t-test independiente o Mannâ€“Whitney.
            - â‰¥3 grupos: ANOVA de una vÃ­a o Kruskalâ€“Wallis.
           La decisiÃ³n se basa en normalidad (Shapiro) y homogeneidad
           de varianza (Levene).
        3. Llama a stats.py para cÃ¡lculos y a story_stats.py para narrativa.

    Args:
        df: DataFrame con los datos.
        value_col: columna numÃ©rica a comparar.
        group_col: columna categÃ³rica que define los grupos.
        test: forzar test concreto ("ttest", "mannwhitney", "anova", "kruskal").
        alpha: nivel de significancia.

    Returns:
        dict con:
            - "test": nombre del test usado
            - "stats_result": dict numÃ©rico del test
            - "normality": lista de resultados por grupo (si calculado)
            - "levene": resultado de Levene (si calculado)
            - "group_names": nombres de grupos utilizados
            - "story_html": narrativa HTML generada
    """
    # Validar columnas
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame.")
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame.")

    # Construir grupos
    subset = df[[group_col, value_col]].dropna(subset=[value_col])
    grouped = subset.groupby(group_col)

    groups = []
    group_names: List[str] = []
    for name, g in grouped:
        vals = g[value_col].dropna().to_numpy()
        if len(vals) >= 2:
            groups.append(vals)
            group_names.append(str(name))

    n_groups = len(groups)
    if n_groups < 2:
        raise ValueError(
            "quick_compare_groups requires at least two groups with â‰¥2 observations each."
        )

    # Importar módulos pesados bajo demanda
    from . import stats as _stats
    from .narrative import story_stats as _story_stats

    test_used: Optional[str] = None
    normality_results: Optional[List[Dict[str, Any]]] = None
    levene_result: Optional[Dict[str, Any]] = None

    test_normalized = (test or "").lower() or None

    def _check_assumptions():
        """Evaluar normalidad (Shapiro) y homogeneidad (Levene)."""
        nonlocal normality_results, levene_result

        normality_results = []
        for g in groups:
            try:
                res = _stats.normality_shapiro(g)
            except Exception as e:
                res = {
                    "test": "shapiro",
                    "n": len(g),
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "error": str(e),
                }
            normality_results.append(res)

        try:
            levene_result = _stats.levene_test(*groups)
        except Exception as e:
            levene_result = {
                "test": "levene",
                "statistic": float("nan"),
                "p_value": float("nan"),
                "error": str(e),
            }

    def _is_normal(res: Dict[str, Any]) -> bool:
        try:
            return float(res.get("p_value", 0.0)) >= alpha
        except Exception:
            return False

    def _has_equal_var(res: Optional[Dict[str, Any]]) -> bool:
        if res is None:
            return False
        try:
            return float(res.get("p_value", 0.0)) >= alpha
        except Exception:
            return False

    # Selección de test según número de grupos
    if n_groups == 2:
        if test_normalized in ("ttest", "student", "parametric"):
            test_used = "ttest_independent"
        elif test_normalized in ("mannwhitney", "mw", "nonparametric"):
            test_used = "mannwhitney_test"
        else:
            _check_assumptions()
            all_normal = all(_is_normal(res) for res in normality_results or [])
            equal_var = _has_equal_var(levene_result)
            if all_normal and equal_var:
                test_used = "ttest_independent"
            else:
                test_used = "mannwhitney_test"

        if test_used == "ttest_independent":
            stats_result = _stats.ttest_independent(groups[0], groups[1])
            story = _story_stats.story_ttest_independent(
                groups[0],
                groups[1],
                name1=group_names[0],
                name2=group_names[1],
                alpha=alpha,
                stats_result=stats_result,
            )
        else:
            stats_result = _stats.mannwhitney_test(groups[0], groups[1])
            story = _story_stats.story_mannwhitney(
                groups[0],
                groups[1],
                name1=group_names[0],
                name2=group_names[1],
                alpha=alpha,
                stats_result=stats_result,
            )

    else:
        if test_normalized in ("anova", "parametric"):
            test_used = "anova_oneway"
        elif test_normalized in ("kruskal", "nonparametric"):
            test_used = "kruskal_test"
        else:
            _check_assumptions()
            all_normal = all(_is_normal(res) for res in normality_results or [])
            equal_var = _has_equal_var(levene_result)
            if all_normal and equal_var:
                test_used = "anova_oneway"
            else:
                test_used = "kruskal_test"

        if test_used == "anova_oneway":
            stats_result = _stats.anova_oneway(*groups)
            story = _story_stats.story_anova_oneway(
                *groups,
                group_names=group_names,
                alpha=alpha,
                stats_result=stats_result,
            )
        else:
            stats_result = _stats.kruskal_test(*groups)
            story = _story_stats.story_kruskal_wallis(
                *groups,
                group_names=group_names,
                alpha=alpha,
                stats_result=stats_result,
            )

    return {
        "test": test_used,
        "stats_result": stats_result,
        "normality": normality_results,
        "levene": levene_result,
        "group_names": group_names,
        "story_html": story,
    }


__all__ = list({
    # Core
    "CONFIG",
    "InsightLabConfig",
    "InsightLabStyle",
    "initialize",
    "load_style",
    "apply_style",
    "load_domain",
    "section",
    "box",
    "divider",
    "show_html",
    "math_insight",
    "CONCEPTS",

    # Narrative
    "narrative",
    "narrative_from_dict",
    "interpretation_5layers",
    "layers",
    "star",
    "explain",
    "result",

    # High-level helpers
    "quick_explore",
    "quick_compare_groups",

    # Metadata
    "__version__",
    "__author__",
    "__description__",
    "__license__",
} | set(_LAZY_MAP.keys()))



# =============================================================================
# INITIALIZATION MESSAGE (OPTIONAL)
# =============================================================================

def _print_welcome():
    """Mensaje de bienvenida opcional (silenciado por defecto)."""
    if CONFIG.verbosity != "silent":
        print(f"InsightLab v{__version__} loaded")
        print("Initialize with: il.initialize(verbosity='stakeholder', style=True)")


# No llamar _print_welcome() automÃ¡ticamente
# El usuario debe hacerlo explÃ­citamente o usar il.initialize()

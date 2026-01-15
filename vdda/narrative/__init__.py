"""
===============================================================================
INSIGHTLAB — NARRATIVE SUBPACKAGE
===============================================================================
Storytelling engine para análisis de datos.

Incluye:
    • story.py — Frameworks narrativos (5-layers, STAR, narrative)
    • story_explore.py — Narrativas de EDA
    • story_prep.py — Narrativas de preparación de datos
    • story_stats.py — Narrativas de tests estadísticos
    • story_ml.py — Narrativas de ML y modelos predictivos

Comentarios: Español
Output: English
===============================================================================
"""

# Core narrative frameworks
from .story import (
    interpretation_5layers,
    layers,  # Alias de interpretation_5layers
    star,
    narrative,
    narrative_from_dict,
    explain,
    result,
)

# Exploration narratives
from .story_explore import (
    story_overview,
    story_variable_types,
    story_missing,
    story_duplicates,
    story_descriptive,
    story_shape,
    story_outliers,
    story_correlation,
    story_data_quality,
)

# Prep narratives
from .story_prep import (
    story_prep,
    #prep_summary,
)

# Stats narratives
from .story_stats import (
    story_ttest_independent,
    story_ttest_paired,
    story_ttest_one_sample,
    story_anova_oneway,
    #story_correlation,
    story_chi_square,
    story_mannwhitney,
    story_wilcoxon,
    story_kruskal_wallis,
    story_normality_test,
    story_levene_test,
    story_bayesian_proportion,
)

# ML narratives
from .story_ml import (
    story_pca,
    story_clusters,
    story_feature_importance,
    story_classification,
    story_regression,
    story_model_comparison,
    story_forecast,
    story_tree_rules,
    story_confusion_matrix,
)


__all__ = [
    # Core frameworks
    "interpretation_5layers",
    "layers",
    "star",
    "narrative",
    "narrative_from_dict",
    "explain",
    "result",
    
    # Explore
    "story_overview",
    "story_variable_types",
    "story_missing",
    "story_duplicates",
    "story_descriptive",
    "story_shape",
    "story_outliers",
    "story_correlation",
    "story_data_quality",
    
    # Prep
    "story_prep",
    "prep_summary",
    
    # Stats
    "story_ttest_independent",
    "story_ttest_paired",
    "story_ttest_one_sample",
    "story_anova_oneway",
    #"story_correlation",
    "story_chi_square",
    "story_mannwhitney",
    "story_wilcoxon",
    "story_kruskal_wallis",
    "story_normality_test",
    "story_levene_test",
    "story_bayesian_proportion",
    
    # ML
    "story_pca",
    "story_clusters",
    "story_feature_importance",
    "story_classification",
    "story_regression",
    "story_model_comparison",
    "story_forecast",
    "story_tree_rules",
    "story_confusion_matrix",
]

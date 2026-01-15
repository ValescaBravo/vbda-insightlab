"""
===============================================================================
INSIGHTLAB — MACHINE LEARNING STORYTELLING MODULE (v2.1)
===============================================================================
Transforms ML model outputs into business narratives using the InsightLab
storytelling engine (STAR, 5-layers, math_insight).

This module is intentionally "math-light":
    • Receives raw arrays (y_true, y_pred, etc.) or simple metric values
    • Returns HTML-ready narrative blocks (no prints)
    • Respects global CONFIG.verbosity (silent / stakeholder / technical)
    • Pairs with insightlab.ml, which does the heavy numeric lifting

Comments: Spanish
Output: English
===============================================================================
"""

from __future__ import annotations

import warnings
from typing import Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    roc_auc_score,
    silhouette_score,
)

# Story engine
from insightlab.narrative.story import (
    narrative,
    narrative_from_dict,
    interpretation_5layers,
    star,
    section,
    box,
)

# Core imports
from insightlab.core import (
    CONFIG,
    math_insight,
    _silent,
    _stakeholder,
    _technical,
)


# =============================================================================
# HELPERS
# =============================================================================

def _validate_arrays(y_true, y_pred, name: str = "predictions") -> None:
    """Validar que arrays tengan la misma longitud (no rompe silenciosamente)."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"[story_ml] {name}: y_true ({len(y_true)}) and y_pred ({len(y_pred)}) "
            f"must have the same length."
        )


def _format_metric(value: float, precision: Optional[int] = None) -> str:
    """
    Formatea una métrica según CONFIG.precision cuando exista.

    precision:
        - None → usa CONFIG.precision si existe, si no 3
        - int  → usa ese valor
    """
    try:
        prec = int(precision if precision is not None else getattr(CONFIG, "precision", 3))
    except Exception:
        prec = 3
    return f"{float(value):.{prec}f}"


def _get_domain_phrase(key: str) -> str:
    """Obtener frase del preset de dominio activo (si existe)."""
    try:
        preset = CONFIG.domain_settings.get(CONFIG.domain, {})  # type: ignore[attr-defined]
        phrases = preset.get("phrases", {})
        return phrases.get(key, "")
    except Exception:
        return ""

        
def _log_story(step: str, details: Dict[str, Any]) -> None:
    trace = getattr(CONFIG, "trace", None)
    if trace is None:
        return
    try:
        trace.log(stage="story_ml", step=step, details=details)
    except Exception:
        pass



def _log_payload(extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construye un payload estándar para el trace de story_ml.
    Siempre intenta añadir dominio y verbosity.
    """
    payload: Dict[str, Any] = {}
    try:
        payload["domain"] = getattr(CONFIG, "domain", None)
        payload["verbosity"] = getattr(CONFIG, "verbosity", None)
    except Exception:
        pass
    payload.update(extra)
    return payload



def _audience() -> str:
    """
    Devuelve el tipo de audiencia según CONFIG.verbosity.

    Returns:
        "stakeholder" | "technical" | "mixed"
    """
    try:
        if _stakeholder():
            return "stakeholder"
        if _technical():
            return "technical"
    except Exception:
        # Si por cualquier motivo CONFIG no está listo, usar modo mixto
        return "mixed"
    return "mixed"



def _business_impact(metric_name: str, value: float, domain: Optional[str] = None) -> str:
    """
    Traducir métrica técnica a impacto de negocio según dominio.

    Args:
        metric_name: "accuracy", "precision", "recall", "r2", "mae"
        value: valor de la métrica
        domain: dominio específico (None = usar CONFIG.domain)
    """
    try:
        domain = domain or CONFIG.domain
    except Exception:
        domain = domain or "generic"

    impact_map = {
        "marketing": {
            "accuracy": f"Campaign targeting effectiveness: {value*100:.1f}% of predictions correct.",
            "precision": f"Ad spend efficiency: {value*100:.1f}% of flagged users are true converters.",
            "recall": f"Opportunity capture: {value*100:.1f}% of potential converters identified.",
            "r2": f"Revenue predictability: Model explains {value*100:.1f}% of sales variance.",
            "mae": f"Forecast error: Predictions off by ${value:,.0f} on average.",
        },
        "churn": {
            "accuracy": f"Churn prediction accuracy: {value*100:.1f}% of forecasts correct.",
            "precision": f"Retention investment efficiency: {value*100:.1f}% of flagged users actually churn.",
            "recall": f"Churn detection: {value*100:.1f}% of at-risk customers identified.",
        },
        "ecommerce": {
            "accuracy": f"Demand forecast accuracy: {value*100:.1f}% of predictions correct.",
            "r2": f"Revenue predictability: Model explains {value*100:.1f}% of sales variance.",
            "mae": f"Forecast error: Predictions off by {value:,.0f} units on average.",
        },
        "healthcare": {
            "accuracy": f"Diagnostic accuracy: {value*100:.1f}% of predictions correct.",
            "precision": f"Treatment precision: {value*100:.1f}% of flagged cases are true positives.",
            "recall": f"Risk detection: {value*100:.1f}% of at-risk patients identified.",
        },
    }

    generic = {
        "accuracy": f"{value*100:.1f}% of predictions are correct.",
        "precision": f"{value*100:.1f}% of positive predictions are true positives.",
        "recall": f"{value*100:.1f}% of actual positives are correctly identified.",
        "r2": f"Model explains {value*100:.1f}% of outcome variance.",
        "mae": f"Predictions are off by {value:.3f} on average.",
    }

    if domain not in impact_map:
        return generic.get(metric_name, "")

    return impact_map[domain].get(metric_name, generic.get(metric_name, ""))


# =============================================================================
# 1. PCA STORY
# =============================================================================

def story_pca(
    pca_model,
    feature_names: Optional[List[str]] = None,
    explained_threshold: float = 0.95,
    domain: Optional[str] = None,
) -> str:
    """
    Narrativa de análisis PCA con contexto de negocio.

    Args:
        pca_model: modelo PCA fitted (sklearn)
        feature_names: nombres de features originales
        explained_threshold: umbral de varianza acumulada para "buena reducción"
        domain: dominio específico (None = usar CONFIG.domain)

    Returns:
        HTML string con narrativa
    """
    if _silent():
        return ""

    if not hasattr(pca_model, "explained_variance_ratio_"):
        return box(
            "warning",
            "Invalid PCA Model",
            "The provided model is not a fitted PCA object or lacks explained_variance_ratio_.",
        )

    explained = np.asarray(pca_model.explained_variance_ratio_)
    cumulative = np.cumsum(explained)
    n_components = len(explained)

    # Identificar top features por componente
    top_components: List[str] = []
    if feature_names is not None and hasattr(pca_model, "components_"):
        for i in range(min(n_components, 3)):  # Top 3 componentes
            idx = np.abs(pca_model.components_[i]).argsort()[::-1][:3]
            feats = ", ".join(
                [
                    feature_names[j]
                    for j in idx
                    if j < len(feature_names)
                ]
            )
            top_components.append(f"PC{i+1}: {feats}")

    section("Principal Component Analysis", icon="🎯")

    original_dim = len(feature_names) if feature_names is not None else "the data"
    insight = (
        f"PCA reduced {original_dim} dimensions into "
        f"{n_components} components capturing {cumulative[-1]*100:.1f}% of total variance."
    )

    evidence_parts = [
        f"PC{i+1}: {explained[i]*100:.1f}% variance (cumulative: {cumulative[i]*100:.1f}%)"
        for i in range(min(n_components, 5))
    ]
    evidence = "<br>".join(evidence_parts)

    if top_components:
        evidence += "<br><br><strong>Key Component Drivers:</strong><br>" + "<br>".join(top_components)

    # Interpretación adaptativa
    if cumulative[-1] >= explained_threshold:
        interpretation = (
            "These components capture most of the data structure. "
            "Dimensionality reduction is highly effective."
        )
    elif cumulative[-1] >= 0.7:
        interpretation = (
            "Components capture substantial variance but some information is lost. "
            "Consider increasing n_components if downstream performance suffers."
        )
    else:
        interpretation = (
            "Components capture limited variance. Data may be highly complex or noisy. "
            "PCA may not be the optimal dimensionality reduction technique."
        )

    # Dominio-específico
    domain = domain or getattr(CONFIG, "domain", None) or "generic"
    if domain == "marketing":
        action = (
            "Use PCs for customer segmentation, lookalike modelling, or reducing "
            "noisy campaign features."
        )
    elif domain == "healthcare":
        action = (
            "Use PCs to identify patient clusters or reduce high-dimensional biomarker data."
        )
    else:
        action = (
            "Use PCs for: (1) visualisation (2D/3D plots), "
            "(2) noise reduction, (3) feature engineering for ML models."
        )

    risk = (
        "PCA components are linear combinations and may be harder to interpret. "
        "Always validate downstream model performance after dimensionality reduction."
    )

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    math_insight("pca")
    
    # 🔹 LOG
    _log_story(
        "story_pca",
        _log_payload({
            "n_components": int(n_components),
            "total_variance": float(cumulative[-1]),
            "explained_threshold": float(explained_threshold),
            "top_components_logged": len(top_components),
        }),
    )


    return html


# =============================================================================
# 2. CLUSTERING STORY
# =============================================================================

def story_clusters(
    labels: np.ndarray,
    X: Optional[np.ndarray] = None,
    cluster_names: Optional[List[str]] = None,
    domain: Optional[str] = None,
    k: Optional[int] = None,
) -> str:
    """
    Narrativa de segmentación de clusters con métricas de calidad.

    Args:
        labels: array de etiquetas de cluster
        X: datos originales (opcional, para silhouette)
        cluster_names: nombres personalizados de clusters
        domain: dominio específico
        k: número esperado de clusters (opcional; si None se infiere de labels)

    Returns:
        HTML string con narrativa
    """
    if _silent():
        return ""

    if len(labels) == 0:
        return box("warning", "Empty Data", "No cluster labels provided.")

    counts = pd.Series(labels).value_counts().sort_index()
    n_detected = len(counts)
    n_clusters = int(k) if k is not None else n_detected

    # Silhouette si X está disponible
    silhouette = None
    if X is not None and n_detected > 1:
        try:
            silhouette = silhouette_score(X, labels)
        except Exception as e:
            warnings.warn(f"[story_clusters] Could not calculate silhouette: {e}")

    section(f"Cluster Analysis: {n_clusters} Segments", icon="🎯")

    evidence = "<strong>Cluster Distribution:</strong><br>"
    for idx, count in counts.items():
        cluster_name = (
            cluster_names[idx]
            if cluster_names and idx < len(cluster_names)
            else f"Cluster {idx}"
        )
        pct = count / len(labels) * 100
        evidence += f"{cluster_name}: <strong>{count}</strong> ({pct:.1f}%)<br>"

    # Si k no coincide con lo detectado, dejamos una nota suave (útil para debugging)
    if k is not None and k != n_detected:
        evidence += (
            f"<br><em>Note: k was set to {k}, "
            f"but {n_detected} distinct clusters were found in labels.</em>"
        )

    if silhouette is not None:
        evidence += f"<br><strong>Silhouette Score:</strong> {silhouette:.3f}"
        if silhouette > 0.5:
            evidence += " (well-separated clusters)"
        elif silhouette > 0.25:
            evidence += " (moderate separation)"
        else:
            evidence += " (weak separation – consider fewer clusters)"

    insight = f"Identified {n_clusters} segments with distinct behavioural patterns."

    if n_clusters <= 3:
        interpretation = (
            f"{n_clusters} segments suggest a simple structure. "
            "Easy to operationalise but may miss nuanced sub-groups."
        )
    elif n_clusters <= 6:
        interpretation = (
            f"{n_clusters} segments balance granularity with interpretability. "
            "Suitable for targeted strategies."
        )
    else:
        interpretation = (
            f"{n_clusters} segments may be too granular for practical use. "
            "Consider hierarchical clustering or dimensionality reduction."
        )

    domain = domain or getattr(CONFIG, "domain", None) or "generic"
    if domain == "marketing":
        action = (
            "Profile each segment: demographics, behaviour, value. "
            "Create personalised campaigns and messaging per cluster."
        )
    elif domain == "ecommerce":
        action = (
            "Identify high-value clusters. Optimise product recommendations "
            "and promotions per segment."
        )
    else:
        action = (
            "Profile each segment to understand what drives membership. "
            "Use clusters as features in downstream models or for targeted interventions."
        )

    risk = (
        "Cluster stability depends on: (1) feature scaling, (2) algorithm choice, "
        "(3) hyperparameters. Validate segments on holdout data."
    )

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    _log_story(
        "story_clusters",
        _log_payload({
            "n_clusters": int(n_clusters),
            "n_clusters_detected": int(n_detected),
            "n_samples": int(len(labels)),
            "silhouette": float(silhouette) if silhouette is not None else None,
        }),
    )

    return html



# =============================================================================
# 3. FEATURE IMPORTANCE STORY
# =============================================================================

def story_feature_importance(
    feature_df: pd.DataFrame,
    top_n: int = 5,
    importance_col: str = "importance",
    feature_col: str = "feature",
    domain: Optional[str] = None,
) -> str:
    """
    Narrativa de importancia de features con validación.
    """
    if _silent():
        return ""

    required = [feature_col, importance_col]
    missing = [c for c in required if c not in feature_df.columns]
    if missing:
        return box(
            "warning",
            "Invalid DataFrame",
            f"Feature importance DataFrame missing required columns: {', '.join(missing)}",
        )

    if len(feature_df) == 0:
        return box("warning", "Empty Data", "No features to analyse.")

    feature_df = feature_df.sort_values(importance_col, ascending=False)
    top = feature_df.head(top_n)

    total_importance = feature_df[importance_col].sum()
    top_importance = top[importance_col].sum()
    concentration = top_importance / total_importance if total_importance > 0 else 0

    section("Feature Importance Analysis", icon="🎯")

    evidence = "<strong>Top Drivers:</strong><br>"
    for _, row in top.iterrows():
        feat = row[feature_col]
        imp = row[importance_col]
        pct = (imp / total_importance * 100) if total_importance > 0 else 0
        evidence += f"{feat}: <strong>{imp:.3f}</strong> ({pct:.1f}%)<br>"

    evidence += f"<br><strong>Top {top_n} Features:</strong> {concentration*100:.1f}% of total importance"

    if concentration > 0.8:
        insight = f"Model is dominated by {top_n} features ({concentration*100:.1f}% of importance)."
        interpretation = (
            "High feature concentration suggests a few strong predictors. "
            "Model is interpretable but may be vulnerable if these features degrade."
        )
    elif concentration > 0.5:
        insight = f"Top {top_n} features drive {concentration*100:.1f}% of predictions."
        interpretation = (
            "Moderate feature concentration. Model balances between key drivers "
            "and supporting features."
        )
    else:
        insight = "Predictive power is distributed across many features."
        interpretation = (
            "Low concentration suggests the model captures complex interactions. "
            "More robust but harder to interpret."
        )

    domain = domain or getattr(CONFIG, "domain", None) or "generic"
    if domain == "marketing":
        action = (
            "Focus dashboards and creative optimisation on top drivers. "
            "Monitor these features for drift over time."
        )
    else:
        action = (
            "Prioritise data quality and engineering for top features. "
            "Consider removing low-importance features to simplify the model."
        )

    risk = (
        "Feature importance varies by model type: "
        "tree-based models (Gini importance), linear models (coefficients), "
        "permutation importance. Always validate with multiple methods."
    )

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    _log_story(
        "story_feature_importance",
        _log_payload({
            "n_features": int(len(feature_df)),
            "top_n": int(top_n),
            "top_concentration": float(concentration),
        }),
    )

    return html


# =============================================================================
# 4. CLASSIFICATION STORY
# =============================================================================

def story_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    class_names: Optional[List[str]] = None,
    domain: Optional[str] = None,
    y_pred_proba: Optional[np.ndarray] = None,
) -> str:
    """
    Narrativa completa de clasificación con business impact,
    adaptada a stakeholder vs technical.
    """
    if _silent():
        return ""

    _validate_arrays(y_true, y_pred, "classification")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    n = len(y_true)

    # AUC si hay probabilidades
    auc = None
    if y_pred_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_pred_proba, average="weighted", multi_class="ovr")
        except Exception:
            auc = None

    section(f"{model_name} Performance", icon="📊")

    aud = _audience()
    domain = domain or getattr(CONFIG, "domain", None) or "generic"

    # Bloque de métricas base
    metrics_block_full = (
        f"<strong>Accuracy:</strong> {_format_metric(acc)}<br>"
        f"<strong>Precision (weighted):</strong> {_format_metric(prec)}<br>"
        f"<strong>Recall (weighted):</strong> {_format_metric(rec)}<br>"
        f"<strong>F1-Score (weighted):</strong> {_format_metric(f1)}"
    )
    if auc is not None:
        metrics_block_full += f"<br><strong>AUC-ROC:</strong> {_format_metric(auc)}"
    metrics_block_full += f"<br><small>N = {n}</small>"

    # Versión resumida para stakeholder
    metrics_block_short = (
        f"<strong>Accuracy:</strong> {_format_metric(acc)}<br>"
        f"<strong>F1-Score:</strong> {_format_metric(f1)}"
    )
    if auc is not None:
        metrics_block_short += f"<br><strong>AUC-ROC:</strong> {_format_metric(auc)}"

    # Impacto de negocio
    business_acc = _business_impact("accuracy", acc, domain)
    business_prec = _business_impact("precision", prec, domain)
    business_rec = _business_impact("recall", rec, domain)

    business_block = ""
    if business_acc:
        business_block = "<br><br><strong>Business Impact:</strong><br>" + business_acc
        if business_prec:
            business_block += "<br>" + business_prec
        if business_rec:
            business_block += "<br>" + business_rec

    # Construir evidence según audiencia
    if aud == "stakeholder":
        evidence = metrics_block_short + business_block
    elif aud == "technical":
        evidence = metrics_block_full + business_block
    else:  # mixed / default
        evidence = metrics_block_full
        if business_block:
            evidence += business_block

    # Insight adaptativo
    if acc >= 0.9:
        insight = f"{model_name} achieves excellent performance ({acc*100:.1f}% accuracy)."
        interpretation = "Model is production-ready with strong predictive power."
    elif acc >= 0.75:
        insight = f"{model_name} achieves good performance ({acc*100:.1f}% accuracy)."
        interpretation = "Model performs well but monitor edge cases and class imbalance."
    else:
        insight = f"{model_name} achieves moderate performance ({acc*100:.1f}% accuracy)."
        interpretation = (
            "Model struggles to distinguish classes. Consider: "
            "(1) more data, (2) feature engineering, (3) a different algorithm."
        )

    # Balance precision/recall (más interesante para técnicos, pero útil para ambos)
    if abs(prec - rec) > 0.2:
        interpretation += (
            f" Precision–recall imbalance detected ({prec:.2f} vs {rec:.2f}). "
            "Adjust decision threshold or class weights."
        )

    action = "Examine the confusion matrix to identify systematic misclassification patterns."

    risk = (
        "Accuracy can be misleading with imbalanced classes. "
        "Always review precision, recall, and F1-score per class."
    )

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    # 🔹 LOG
    _log_story(
        "story_classification",
        _log_payload({
            "model_name": model_name,
            "n": int(len(y_true)),
            "accuracy_weighted": float(acc),
            "precision_weighted": float(prec),
            "recall_weighted": float(rec),
            "f1_weighted": float(f1),
            "auc_weighted": float(auc) if auc is not None else None,
        }),
    )

    return html



# =============================================================================
# 5. REGRESSION STORY
# =============================================================================

def story_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    domain: Optional[str] = None,
    units: str = "",
) -> str:
    """
    Narrativa de regresión con business impact,
    adaptada a stakeholder vs technical.
    """
    if _silent():
        return ""

    _validate_arrays(y_true, y_pred, "regression")

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true_arr = np.asarray(y_true, dtype=float)
    mean_actual = float(np.mean(y_true_arr))
    if mean_actual != 0:
        mape = float(np.mean(np.abs((y_true_arr - np.asarray(y_pred)) / y_true_arr)) * 100)
    else:
        mape = 0.0

    section(f"{model_name} Performance", icon="📈")

    aud = _audience()
    domain = domain or getattr(CONFIG, "domain", None) or "generic"
    u = f" {units}" if units else ""

    # Bloques de métricas
    metrics_full = (
        f"<strong>R²:</strong> {_format_metric(r2)}<br>"
        f"<strong>MAE:</strong> {_format_metric(mae)}{u}<br>"
        f"<strong>RMSE:</strong> {_format_metric(rmse)}{u}<br>"
        f"<strong>MAPE:</strong> {_format_metric(mape, precision=1)}%"
    )
    metrics_short = (
        f"<strong>R²:</strong> {_format_metric(r2)}<br>"
        f"<strong>MAPE:</strong> {_format_metric(mape, precision=1)}%"
    )

    business_r2 = _business_impact("r2", r2, domain)
    business_mae = _business_impact("mae", mae, domain)

    business_block = ""
    if business_r2:
        business_block = "<br><br><strong>Business Impact:</strong><br>" + business_r2
        if business_mae:
            business_block += "<br>" + business_mae

    if aud == "stakeholder":
        evidence = metrics_short + business_block
    elif aud == "technical":
        evidence = metrics_full + business_block
    else:
        evidence = metrics_full
        if business_block:
            evidence += business_block

    # Insight adaptativo
    if r2 >= 0.8:
        insight = f"{model_name} explains {r2*100:.1f}% of outcome variance (excellent fit)."
        interpretation = "Model captures the main trends and is suitable for forecasting."
    elif r2 >= 0.5:
        insight = f"{model_name} explains {r2*100:.1f}% of outcome variance (good fit)."
        interpretation = (
            "Model captures substantial variance but unexplained factors remain. "
            "Consider additional features or non-linear relationships."
        )
    else:
        insight = f"{model_name} explains {r2*100:.1f}% of outcome variance (weak fit)."
        interpretation = (
            "Model struggles to predict the outcome. "
            "Key drivers may be missing or relationships may be non-linear."
        )

    # Error interpretation
    if mape < 10:
        interpretation += f" Prediction errors are low (MAPE: {mape:.1f}%)."
    elif mape < 20:
        interpretation += f" Prediction errors are moderate (MAPE: {mape:.1f}%)."
    else:
        interpretation += (
            f" Prediction errors are high (MAPE: {mape:.1f}%). "
            "Review outliers and feature quality."
        )

    action = (
        "Inspect residual plots to diagnose: "
        "(1) heteroscedasticity, (2) non-linearity, (3) outliers, (4) omitted variables."
    )

    risk = (
        "R² alone does not guarantee good predictions. "
        "Always validate on holdout data and monitor MAE/RMSE in production."
    )

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    math_insight("r2")

    # 🔹 LOG
    _log_story(
        "story_regression",
        _log_payload({
            "model_name": model_name,
            "n": int(len(y_true)),
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "units": units,
        }),
    )

    return html



# =============================================================================
# 6. MODEL COMPARISON STORY
# =============================================================================
def story_model_comparison(scores_dict: Dict[str, float]) -> str:
    """
    Historia de comparación de modelos.

    Args:
        scores_dict: {model_name: score}
    """
    if _silent():
        return ""

    if not scores_dict:
        return box("warning", "No Models", "No model scores provided for comparison.")

    sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    best, best_score = sorted_scores[0]

    section("Model Comparison", icon="🏆")

    insight = f"The best performing model is {best} with {best_score*100:.1f}% score."

    evidence_lines = [
        f"{m}: <strong>{s*100:.1f}%</strong>"
        for m, s in sorted_scores
    ]
    evidence = "<br>".join(evidence_lines)

    interpretation = (
        "Performance varies across algorithms, indicating differences in "
        "how they capture feature interactions."
    )

    action = "Validate the best model with cross-validation or out-of-time testing before deployment."

    risk = "Scores may not generalise if the test set is small, noisy, or non-representative."

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    _log_story(
        "story_model_comparison",
        _log_payload({
            "n_models": int(len(scores_dict)),
            "best_model": best,
            "best_score": float(best_score),
        }),
    )

    return html


# =============================================================================
# 7. FORECASTING STORY
# =============================================================================

def story_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Forecast Model",
    posterior_prob: Optional[float] = None,
) -> str:
    """
    Historia de forecasting con MAE y opción bayesiana.
    """
    if _silent():
        return ""

    _validate_arrays(y_true, y_pred, "forecasting")

    mae = mean_absolute_error(y_true, y_pred)

    section(f"{model_name} Performance", icon="📅")

    insight = f"{model_name} predicts future values with an average error of {_format_metric(mae)}."

    evidence = f"<strong>MAE:</strong> {_format_metric(mae)}"

    interpretation = (
        "The model is suitable for short-term forecasting and captures the main trend."
    )

    action = "Monitor drift regularly and update the model as new data arrives."

    risk = "Forecast accuracy degrades when behaviour shifts abruptly."

    if posterior_prob is not None:
        evidence += (
            f"<br><strong>Posterior probability of increase:</strong> "
            f"{posterior_prob*100:.1f}%"
        )
        interpretation += (
            f" The Bayesian posterior suggests {posterior_prob*100:.1f}% probability of an upward trend."
        )

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    _log_story(
        "story_forecast",
        _log_payload({
            "model_name": model_name,
            "n": int(len(y_true)),
            "mae": float(mae),
            "posterior_prob_increase": float(posterior_prob) if posterior_prob is not None else None,
        }),
    )

    return html


# =============================================================================
# 8. DECISION TREE TEXT SUMMARY
# =============================================================================

def story_tree_rules(rules: List[str], model_name: str = "Decision Tree") -> str:
    """
    Resumen textual de reglas de árbol de decisión.

    Args:
        rules: lista de reglas en formato texto
        model_name: nombre del modelo

    Returns:
        HTML string con caja de reglas
    """
    if _silent():
        return ""

    content = "<br>".join(rules)
    return box("business", f"{model_name} — Decision Logic", content)


# =============================================================================
# 9. CONFUSION MATRIX STORY
# =============================================================================

def story_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
) -> str:
    """
    Narrativa detallada de matriz de confusión.

    Args:
        y_true: etiquetas verdaderas
        y_pred: predicciones
        class_names: nombres de clases
        normalize: normalizar por fila (True) o usar conteos absolutos (False)

    Returns:
        HTML string con narrativa
    """
    if _silent():
        return ""

    _validate_arrays(y_true, y_pred, "confusion_matrix")

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    section("Confusion Matrix Analysis", icon="🎯")

    misclass: List[tuple] = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                misclass.append(
                    (class_names[i], class_names[j], float(cm[i, j]))
                )

    misclass.sort(key=lambda x: x[2], reverse=True)

    evidence = "<strong>Confusion Matrix:</strong><br>"
    for i in range(n_classes):
        correct = cm[i, i]
        if normalize:
            evidence += f"{class_names[i]}: {correct*100:.1f}% correctly classified<br>"
        else:
            evidence += f"{class_names[i]}: {int(correct)} correctly classified<br>"

    if misclass:
        evidence += "<br><strong>Top Misclassifications:</strong><br>"
        for true_class, pred_class, val in misclass[:5]:
            if normalize:
                evidence += f"{true_class} → {pred_class}: {val*100:.1f}%<br>"
            else:
                evidence += f"{true_class} → {pred_class}: {int(val)} cases<br>"

    if not misclass:
        insight = "Perfect classification — all predictions correct."
        interpretation = "Model separates classes with 100% accuracy."
    elif len(misclass) <= 2:
        insight = f"Model shows limited confusion between {len(misclass)} class pairs."
        interpretation = "Most classes are well-separated. Focus on the confused pairs."
    else:
        insight = f"Model shows {len(misclass)} distinct misclassification patterns."
        interpretation = "Several classes are confused. Consider feature engineering or rebalancing."

    if misclass:
        top_confusion = misclass[0]
        interpretation += (
            f" Primary confusion: {top_confusion[0]} misclassified as {top_confusion[1]}. "
            "Investigate distinguishing features."
        )

    action = (
        "Use the confusion matrix to: "
        "(1) identify weak class boundaries, "
        "(2) adjust decision thresholds, "
        "(3) collect more data for confused classes."
    )

    risk = "Class imbalance can inflate accuracy while hiding poor minority class performance."

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    math_insight("chi_square")

    # 🔹 LOG
    overall_acc = float(np.trace(cm) / np.sum(cm)) if cm.sum() > 0 else None
    _log_story(
        "story_confusion_matrix",
        _log_payload({
            "n_classes": int(n_classes),
            "normalised": bool(normalize),
            "overall_accuracy_from_cm": overall_acc,
        }),
    )

    return html


# =============================================================================
# 10. CROSS-VALIDATION STORY
# =============================================================================

def story_cross_validation(
    cv_scores: Union[np.ndarray, List[float]],
    metric_name: str = "Accuracy",
    n_folds: Optional[int] = None,
) -> str:
    """
    Narrativa de resultados de cross-validation, adaptada por audiencia.
    """
    if _silent():
        return ""

    scores = np.asarray(cv_scores, dtype=float)
    if scores.size == 0:
        return box("warning", "Empty Scores", "No cross-validation scores provided.")

    n_folds = n_folds or len(scores)

    mean_score = float(scores.mean())
    std_score = float(scores.std())
    min_score = float(scores.min())
    max_score = float(scores.max())

    section(f"Cross-Validation Results ({n_folds}-Fold)", icon="🔄")

    aud = _audience()

    metrics_full = (
        f"<strong>Mean {metric_name}:</strong> {_format_metric(mean_score)}<br>"
        f"<strong>Std Dev:</strong> {_format_metric(std_score)}<br>"
        f"<strong>Min:</strong> {_format_metric(min_score)}<br>"
        f"<strong>Max:</strong> {_format_metric(max_score)}<br>"
        f"<strong>Range:</strong> {_format_metric(max_score - min_score)}"
    )
    metrics_short = (
        f"<strong>Mean {metric_name}:</strong> {_format_metric(mean_score)}<br>"
        f"<strong>Std Dev:</strong> {_format_metric(std_score)}"
    )

    evidence = metrics_short if aud == "stakeholder" else metrics_full

    cv_stability = std_score / mean_score if mean_score > 0 else 0.0

    if cv_stability < 0.05:
        insight = f"Model shows excellent stability (CV std: {std_score:.3f})."
        interpretation = "Performance is consistent across folds. Model generalises well."
    elif cv_stability < 0.15:
        insight = f"Model shows good stability (CV std: {std_score:.3f})."
        interpretation = "Performance varies slightly but remains acceptable."
    else:
        insight = f"Model shows high variance (CV std: {std_score:.3f})."
        interpretation = (
            "Performance fluctuates significantly across folds. "
            "Model may be sensitive to training data composition."
        )

    if mean_score < 0.6:
        interpretation += " Low mean score suggests underfitting. Add features or increase model complexity."

    action = (
        f"Use {mean_score:.3f} ± {std_score:.3f} as expected performance on unseen data. "
        "If variance is high, consider: (1) more data, (2) regularisation, (3) ensemble methods."
    )

    risk = "CV estimates generalisation but cannot detect data leakage or temporal drift."

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    #🔹 LOG
    _log_story(
        "story_cross_validation",
        _log_payload({
            "metric_name": metric_name,
            "n_folds": int(n_folds),
            "mean_score": float(mean_score),
            "std_score": float(std_score),
            "min_score": float(min_score),
            "max_score": float(max_score),
        }),
    )

    return html


# =============================================================================
# 11. LEARNING CURVE STORY
# =============================================================================

def story_learning_curve(
    train_scores: Union[np.ndarray, List[float]],
    val_scores: Union[np.ndarray, List[float]],
    train_sizes: Union[np.ndarray, List[int]],
) -> str:
    """
    Narrativa de learning curve para diagnosticar over/underfitting,
    con evidencia ajustada por audiencia.
    """
    if _silent():
        return ""

    train_scores_arr = np.asarray(train_scores, dtype=float)
    val_scores_arr = np.asarray(val_scores, dtype=float)
    train_sizes_arr = np.asarray(train_sizes)

    if not (
        len(train_scores_arr) == len(val_scores_arr) == len(train_sizes_arr)
    ):
        return box(
            "warning",
            "Invalid Data",
            "train_scores, val_scores, and train_sizes must have the same length.",
        )

    gap_arr = train_scores_arr - val_scores_arr
    final_gap = float(gap_arr[-1])
    final_val = float(val_scores_arr[-1])

    if len(val_scores_arr) > 1:
        val_improvement = float(val_scores_arr[-1] - val_scores_arr[0])
    else:
        val_improvement = 0.0

    section("Learning Curve Analysis", icon="📊")

    aud = _audience()

    metrics_full = (
        f"<strong>Final Training Score:</strong> {_format_metric(train_scores_arr[-1])}<br>"
        f"<strong>Final Validation Score:</strong> {_format_metric(final_val)}<br>"
        f"<strong>Train–Val Gap:</strong> {_format_metric(final_gap)}<br>"
        f"<strong>Validation Improvement:</strong> {_format_metric(val_improvement)}"
    )
    metrics_short = (
        f"<strong>Final Training Score:</strong> {_format_metric(train_scores_arr[-1])}<br>"
        f"<strong>Final Validation Score:</strong> {_format_metric(final_val)}"
    )

    evidence = metrics_short if aud == "stakeholder" else metrics_full

    # Diagnóstico
    if final_gap > 0.15:
        diagnosis = "Overfitting"
        insight = f"Model shows overfitting (train–val gap: {final_gap:.3f})."
        interpretation = (
            "Training score significantly exceeds validation score. "
            "Model memorises training data rather than learning generalisable patterns."
        )
        action = (
            "Reduce overfitting: "
            "(1) increase regularisation, "
            "(2) reduce model complexity, "
            "(3) add more training data, "
            "(4) use dropout or early stopping."
        )
    elif final_val < 0.6:
        diagnosis = "Underfitting"
        insight = f"Model shows underfitting (val score: {final_val:.3f})."
        interpretation = (
            "Both training and validation scores are low. "
            "Model is too simple to capture data patterns."
        )
        action = (
            "Reduce underfitting: "
            "(1) increase model complexity, "
            "(2) add more features, "
            "(3) reduce regularisation, "
            "(4) train longer."
        )
    elif val_improvement < 0.02 and len(val_scores_arr) > 3:
        diagnosis = "Plateau"
        insight = "Validation score has plateaued — more data will not help significantly."
        interpretation = (
            "Model has reached its learning capacity. "
            "Further improvements require better features or a different architecture."
        )
        action = "Focus on feature engineering rather than collecting more data."
    else:
        diagnosis = "Good Fit"
        insight = f"Model shows good generalisation (train–val gap: {final_gap:.3f})."
        interpretation = (
            "Training and validation scores converge. "
            "Model generalises well to unseen data."
        )
        action = "Model is ready for deployment. Monitor performance on a holdout test set."

    risk = (
        "Learning curves diagnose the bias–variance trade-off but do not detect "
        "data leakage or distribution shift."
    )

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    diagnosis_content = (
        f"<strong>Diagnosis:</strong> {diagnosis}<br>"
        f"<strong>Recommendation:</strong> {action}"
    )

    html += box("insight", "Model Diagnosis", diagnosis_content)

     # 🔹 LOG
    _log_story(
        "story_learning_curve",
        _log_payload({
            "final_train_score": float(train_scores_arr[-1]),
            "final_val_score": float(final_val),
            "final_gap": float(final_gap),
            "val_improvement": float(val_improvement),
            "diagnosis": diagnosis,
        }),
    )

    return html



# =============================================================================
# 12. RESIDUAL ANALYSIS STORY
# =============================================================================

def story_residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot: bool = False,
) -> str:
    """
    Narrativa de análisis de residuos para regresión,
    con detalle ajustado a stakeholder vs technical.
    """
    if _silent():
        return ""

    _validate_arrays(y_true, y_pred, "residuals")

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    residuals = y_true_arr - y_pred_arr

    mean_resid = float(np.mean(residuals))
    std_resid = float(np.std(residuals))
    skew_resid = float(pd.Series(residuals).skew())

    abs_resid = np.abs(residuals)
    heteroscedasticity = float(np.corrcoef(abs_resid, y_pred_arr)[0, 1])

    section("Residual Analysis", icon="🔬")

    aud = _audience()

    metrics_full = (
        f"<strong>Mean Residual:</strong> {_format_metric(mean_resid)}<br>"
        f"<strong>Std Residual:</strong> {_format_metric(std_resid)}<br>"
        f"<strong>Skewness:</strong> {_format_metric(skew_resid)}<br>"
        f"<strong>Heteroscedasticity (r):</strong> {_format_metric(heteroscedasticity)}"
    )

    # Para stakeholder damos menos números y más mensaje
    if aud == "stakeholder":
        # Evaluamos issues primero
        issues_short: List[str] = []
        if std_resid > 0 and abs(mean_resid) > 0.1 * std_resid:
            issues_short.append("small systematic bias")
        if abs(skew_resid) > 1.0:
            issues_short.append("asymmetry in errors")
        if abs(heteroscedasticity) > 0.3:
            issues_short.append("non-constant error variance")

        if not issues_short:
            evidence = "Residuals look well behaved (no strong bias or pattern detected)."
        else:
            evidence = (
                "Residuals reveal: "
                + ", ".join(issues_short)
                + "."
            )
    else:
        evidence = metrics_full

    # Diagnóstico técnico (reutilizamos la lógica anterior)
    issues: List[str] = []

    if std_resid > 0 and abs(mean_resid) > 0.1 * std_resid:
        issues.append("Systematic bias detected (mean ≠ 0).")

    if abs(skew_resid) > 1.0:
        issues.append(f"Residuals are skewed (skewness ≈ {skew_resid:.2f}).")

    if abs(heteroscedasticity) > 0.3:
        issues.append("Heteroscedasticity detected (non-constant variance).")

    if not issues:
        insight = "Residuals show healthy properties: centred, symmetric, homoscedastic."
        interpretation = (
            "Model assumptions are broadly satisfied. Predictions are unbiased and reliable."
        )
        action = "Model is ready for deployment."
    else:
        insight = f"Residual analysis reveals {len(issues)} diagnostic issue(s)."
        # En stakeholder podemos dejar la lista, sigue siendo útil
        interpretation = "<br>".join(f"• {issue}" for issue in issues)
        action = (
            "Address these issues: "
            "(1) add missing features to reduce bias, "
            "(2) transform the outcome variable to reduce skew, "
            "(3) use robust standard errors if heteroscedasticity persists."
        )

    if plot:
        action += "<br><br>Recommended plots: (1) residuals vs fitted, (2) Q–Q plot, (3) scale–location plot."

    risk = (
        "Violated assumptions affect confidence intervals and hypothesis tests, "
        "but predictions may still be useful in practice."
    )

    html = narrative(
        insight=insight,
        evidence=evidence,
        interpretation=interpretation,
        action=action,
        risk=risk,
    )

    # 🔹 LOG
    _log_story(
        "story_residual_analysis",
        _log_payload({
            "mean_residual": float(mean_resid),
            "std_residual": float(std_resid),
            "skew_residual": float(skew_resid),
            "heteroscedasticity_r": float(heteroscedasticity),
            "n_issues": int(len(issues)),
        }),
    )
    

    return html


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Core ML stories
    "story_pca",
    "story_clusters",
    "story_feature_importance",
    "story_classification",
    "story_regression",
    # Comparison & forecasting
    "story_model_comparison",
    "story_forecast",
    "story_tree_rules",
    # Diagnostics
    "story_confusion_matrix",
    "story_cross_validation",
    "story_learning_curve",
    "story_residual_analysis",
]

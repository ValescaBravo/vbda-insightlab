"""
===============================================================================
INSIGHTLAB MACHINE LEARNING MODULE (Final – Lightweight + Narrative Metadata)
===============================================================================
Clean, modern, explainable ML utilities with InsightLab branding.

Rol de este módulo:
    • Entrenar modelos (wrappers ligeros)
    • Hacer diagnósticos numéricos de ML (PCA, clasificación, regresión, clusters)
    • Generar METADATA estructurada para story_ml.py (no narrativa aquí)
    • Opcionalmente registrar pasos en CONFIG.trace (stage="ml")
    • Respetar CONFIG.Style y CONFIG.auto_plot para visualizaciones

Comentarios: Español
Salidas técnicas: Inglés
No emojis.
===============================================================================
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

from scipy import stats  # Para QQ-plot

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics import silhouette_score

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


# =============================================================================
# =============================================================================
# CORE: CONFIG + ESTILO + TRACE (opcionales)
# =============================================================================

try:  # En un paquete real: insightlab.core existe siempre
    from .core import CONFIG, apply_style  # type: ignore
except Exception:  # pragma: no cover
    CONFIG = None  # type: ignore

    def apply_style() -> None:  # type: ignore
        """Fallback vacío si core no está disponible."""
        pass


def _auto_plot_enabled() -> bool:
    """
    Helper para respetar CONFIG.auto_plot si existe.
    """
    if CONFIG is None:
        return True
    return bool(getattr(CONFIG, "auto_plot", True))


def _log_ml_step(step: str, details: Dict[str, Any]) -> None:
    """
    Registrar paso de ML en el trace global si existe.

    Nunca debe romper el flujo analítico.
    """
    if CONFIG is None:
        return
    trace = getattr(CONFIG, "trace", None)
    if trace is None:
        return
    try:
        trace.log(stage="ml", step=step, details=details)
    except Exception:
        # El trace nunca debe romper el análisis.
        pass


# =============================================================================
# NARRATIVE HELPER (optional)
# =============================================================================
try:
    # Cuando trabajas dentro del paquete insightlab local
    from .story import narrative_from_dict  # type: ignore
except Exception:
    try:
        # Cuando insightlab está instalado como paquete
        from insightlab.narrative.story import narrative_from_dict  # type: ignore
    except Exception:
        # Fallback silencioso: no rompe si el motor narrativo no está disponible
        def narrative_from_dict(info: Dict[str, Any]) -> str:  # type: ignore
            return ""

#=========================================================     
# 1. PCA ANALYSIS
#=========================================================
def plot_pca_components(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    *,
    scale: bool = True,
    add_centroids: bool = True,
    add_ellipses: bool = False,
    show_cluster_sizes: bool = True,
    cluster_labels: Optional[Dict[Any, str]] = None,
) -> plt.Figure:
    """
    PCA 2D projection (PC1 vs PC2) with optional group details.

    Enhancements vs previous version:
      - Axis labels and title include explained variance.
      - Legend can show n and % of each cluster.
      - Optional centroids (big X) per cluster.
      - Optional 95% confidence ellipses per cluster.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] < 2:
        raise ValueError("PCA requires at least two numeric columns.")

    # Fit PCA on numeric features
    if scale:
        X = StandardScaler().fit_transform(numeric_df.values)
    else:
        X = numeric_df.values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    total_var = var_pc1 + var_pc2

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    if label_col is not None and label_col in df.columns:
        labels = df.loc[numeric_df.index, label_col]
        unique = sorted(labels.dropna().unique())
        palette = getattr(CONFIG.Style, "VIZ_PALETTE", None) if CONFIG else None

        for i, lab in enumerate(unique):
            mask = labels == lab
            X_group = X_pca[mask]
            n = int(mask.sum())
            pct = (n / len(labels) * 100.0) if len(labels) > 0 else 0.0

            # Optional human-readable cluster names
            name = (
                cluster_labels.get(lab, str(lab))
                if cluster_labels is not None
                else str(lab)
            )
            legend_label = (
                f"{name} (n={n}, {pct:.1f}%)"
                if show_cluster_sizes
                else name
            )

            color = (
                palette[i % len(palette)]
                if palette is not None
                else None
            )

            # Points
            ax.scatter(
                X_group[:, 0],
                X_group[:, 1],
                alpha=0.8,
                s=25,
                label=legend_label,
                color=color,
            )

            # Centroid marker
            if add_centroids and n > 0:
                cx, cy = X_group.mean(axis=0)
                ax.scatter(
                    cx,
                    cy,
                    marker="X",
                    s=90,
                    color=color,
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=4,
                )

            # 95% ellipse
            if add_ellipses and n > 2:
                _add_confidence_ellipse(
                    ax,
                    X_group[:, 0],
                    X_group[:, 1],
                    n_std=2.0,
                    edgecolor=color or "grey",
                    linewidth=1.0,
                    alpha=0.5,
                )

        ax.legend(title=label_col or "Group")
    else:
        # No labels: simple scatter
        ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            alpha=0.8,
            s=25,
            color=getattr(CONFIG.Style, "TEAL", "#13d6c1") if CONFIG else "#13d6c1",
        )

    ax.set_xlabel(f"PC1 ({var_pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}%)")
    ax.set_title(f"PCA 2D Projection (PC1+PC2 = {total_var:.1f}% variance)")

    # Factor-map style axes
    ax.axhline(0, linewidth=0.5, color="grey", alpha=0.4)
    ax.axvline(0, linewidth=0.5, color="grey", alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")

    if _auto_plot_enabled():
        plt.show()

    _log_ml_step(
        "plot_pca_components",
        {
            "n_samples": int(numeric_df.shape[0]),
            "n_features": int(numeric_df.shape[1]),
        },
    )

    return fig

def plot_pca_biplot(
    df: pd.DataFrame,
    *,
    scale: bool = True,
    max_features: int = 8,
) -> plt.Figure:
    """
    PCA biplot: individuals (scores) + variable loadings (arrows).

    This is the classic PCA graphic used in a lot of textbooks and in
    tools like FactoMineR / factoextra.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] < 2:
        raise ValueError("PCA requires at least two numeric columns.")

    if scale:
        X = StandardScaler().fit_transform(numeric_df.values)
    else:
        X = numeric_df.values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    feature_names = list(numeric_df.columns)

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    total_var = var_pc1 + var_pc2

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Points (individuals)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, s=15)

    # Scale loadings so arrows fit in the same space
    x_span = X_pca[:, 0].max() - X_pca[:, 0].min()
    y_span = X_pca[:, 1].max() - X_pca[:, 1].min()
    arrow_scale = 0.5 * max(x_span, y_span)

    loadings = pca.components_.T  # shape (n_features, 2)

    for i in range(min(max_features, loadings.shape[0])):
        vx, vy = loadings[i, 0] * arrow_scale, loadings[i, 1] * arrow_scale
        ax.arrow(
            0,
            0,
            vx,
            vy,
            linewidth=1.0,
            head_width=0.03 * arrow_scale,
            head_length=0.05 * arrow_scale,
            length_includes_head=True,
            color=getattr(CONFIG.Style, "NAVY", "#1d085e") if CONFIG else "#1d085e",
        )
        ax.text(
            vx * 1.05,
            vy * 1.05,
            feature_names[i],
            fontsize=8,
            ha="center",
            va="center",
        )

    ax.set_xlabel(f"PC1 ({var_pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}%)")
    ax.set_title(f"PCA Biplot (PC1+PC2 = {total_var:.1f}% variance)")
    ax.axhline(0, linewidth=0.5, color="grey", alpha=0.4)
    ax.axvline(0, linewidth=0.5, color="grey", alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")

    if _auto_plot_enabled():
        plt.show()

    _log_ml_step(
        "plot_pca_biplot",
        {
            "n_samples": int(numeric_df.shape[0]),
            "n_features": int(numeric_df.shape[1]),
        },
    )

    return fig

# =============================================================================
# 1. PCA ANALYSIS
# =============================================================================
def _add_confidence_ellipse(ax, x, y, n_std: float = 2.0, **kwargs) -> None:
    """
    Draw a covariance-based confidence ellipse of x and y on ax.

    n_std ~ number of standard deviations (2.0 ≈ 95% for Gaussian data).

    This is inspired by common PCA/ordination plots where ellipses
    summarise group spread in the factor map.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size <= 1:
        return

    cov = np.cov(x, y)
    # If covariance matrix is singular, skip.
    if np.linalg.det(cov) == 0:
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellip = Ellipse(
        (np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        fill=False,
        **kwargs,
    )
    ax.add_patch(ellip)

#=====================================================================================
def analyse_pca(
    df: pd.DataFrame,
    n_components: int = 2,
    *,
    scale: bool = True,
    dropna: bool = True,
    plot: bool = False,
) -> Dict[str, Any]:
    """
    Ejecuta PCA sobre columnas numéricas y retorna metadata rica.

    Args:
        df: DataFrame de entrada.
        n_components: número de componentes a extraer.
        scale: si True, aplica StandardScaler antes de PCA.
        dropna: si True, elimina filas con NA en columnas numéricas.
        plot: si True, genera gráficos (varianza + proyección 2D).

    Returns:
        dict con:
            - task: "pca"
            - pca_model: objeto PCA fitted
            - components: matriz de scores (n_samples x n_components)
            - scaled_data: datos escalados usados en PCA
            - feature_names: lista de columnas numéricas usadas
            - metadata: {
                  "n_components",
                  "explained_variance",
                  "cumulative_variance",
                  "components_df",
                  "top_features"
              }
            - figures: {"variance_plot": fig, "projection_2d": fig?}
    """
    # Seleccionar columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    if dropna:
        numeric_df = numeric_df.dropna()

    if numeric_df.shape[1] < 2:
        raise ValueError("PCA requires at least two numeric columns.")

    feature_names = list(numeric_df.columns)

    # Escalado
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df.values)
    else:
        X_scaled = numeric_df.values

    # PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    components_df = pd.DataFrame(
        pca.components_,
        index=[f"PC{i+1}" for i in range(n_components)],
        columns=feature_names,
    )

    top_features: Dict[str, List[str]] = {
        f"PC{i+1}": components_df.iloc[i].abs().nlargest(3).index.tolist()
        for i in range(n_components)
    }

    metadata = {
        "n_components": n_components,
        "explained_variance": explained,
        "cumulative_variance": cumulative,
        "components_df": components_df,
        "top_features": top_features,
    }

    figs: Dict[str, plt.Figure] = {}

    if plot:
        apply_style()

        # Varianza explicada
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(
            range(1, n_components + 1),
            explained * 100,
            color=getattr(CONFIG.Style, "TEAL", "#13d6c1") if CONFIG else "#13d6c1",
        )
        ax1.plot(
            range(1, n_components + 1),
            cumulative * 100,
            marker="o",
            color=getattr(CONFIG.Style, "NAVY", "#1d085e") if CONFIG else "#1d085e",
        )
        ax1.set_title("PCA: Variance Explained")
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Explained variance (%)")
        ax1.grid(True, alpha=0.3)
        figs["variance_plot"] = fig1
        if _auto_plot_enabled():
            plt.show()

        # Proyección 2D
        if n_components >= 2:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.scatter(
                components[:, 0],
                components[:, 1],
                alpha=0.7,
                color=getattr(CONFIG.Style, "TEAL", "#13d6c1") if CONFIG else "#13d6c1",
            )
            ax2.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
            ax2.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
            ax2.set_title("PCA 2D Projection")
            ax2.grid(True, alpha=0.3)
            figs["projection_2d"] = fig2
            if _auto_plot_enabled():
                plt.show()

    result = {
        "task": "pca",
        "pca_model": pca,
        "components": components,
        "scaled_data": X_scaled,
        "feature_names": feature_names,
        "metadata": metadata,
        "figures": figs,
    }

    _log_ml_step(
        "analyse_pca",
        {
            "n_samples": int(numeric_df.shape[0]),
            "n_features": int(numeric_df.shape[1]),
            "n_components": int(n_components),
            "explained_total": float(cumulative[-1]),
        },
    )

    return result


def plot_pca_variance_explained(
    df: pd.DataFrame,
    *,
    max_components: Optional[int] = None,
    scale: bool = True,
) -> plt.Figure:
    """
    Gráfico simple de varianza explicada por PCA.

    Devuelve la figura para uso programático.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] < 2:
        raise ValueError("PCA requires at least two numeric columns.")

    if scale:
        X = StandardScaler().fit_transform(numeric_df.values)
    else:
        X = numeric_df.values

    n_features = X.shape[1]
    n_comp = max_components or n_features
    pca = PCA(n_components=n_comp)
    pca.fit(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(1, len(explained) + 1),
        explained * 100,
        alpha=0.5,
        color=getattr(CONFIG.Style, "TEAL", "#13d6c1") if CONFIG else "#13d6c1",
    )
    ax.plot(
        range(1, len(explained) + 1),
        cumulative * 100,
        marker="o",
        linewidth=2,
        color=getattr(CONFIG.Style, "NAVY", "#1d085e") if CONFIG else "#1d085e",
    )
    ax.set_title("PCA: Cumulative Variance Explained")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance (%)")
    ax.grid(True, alpha=0.3)

    if _auto_plot_enabled():
        plt.show()

    _log_ml_step(
        "plot_pca_variance_explained",
        {
            "n_samples": int(numeric_df.shape[0]),
            "n_features": int(numeric_df.shape[1]),
            "n_components": int(len(explained)),
        },
    )

    return fig



# Alias por compatibilidad con código antiguo
def plot_pca_2d_projection(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
) -> plt.Figure:
    """Alias para plot_pca_components (compat)."""
    return plot_pca_components(df, label_col=label_col)


# =============================================================================
# 2. FEATURE IMPORTANCE
# =============================================================================

def analyse_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Calcula importancia de características y retorna metadata.

    Args:
        model: modelo fitted con atributo feature_importances_.
        feature_names: lista de nombres de variables.
        top_n: número de features top a destacar.

    Returns:
        dict con:
            - task: "feature_importance"
            - importance_df: DataFrame ordenado desc.
            - top: DataFrame con top_n
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model has no feature_importances_ attribute.")

    importances = np.asarray(model.feature_importances_)
    if len(importances) != len(feature_names):
        raise ValueError("feature_importances_ length does not match feature_names length.")

    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    top = df_imp.head(top_n)

    result = {
        "task": "feature_importance",
        "importance_df": df_imp,
        "top": top,
        "top_n": int(top_n),
    }

    _log_ml_step(
        "analyse_feature_importance",
        {
            "n_features": int(len(feature_names)),
            "top_n": int(top_n),
        },
    )

    return result


# =============================================================================
# 3. TRAINING WRAPPERS
# =============================================================================

def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    **fit_kwargs: Any,
) -> Dict[str, Any]:
    """
    Entrena un modelo sklearn-like y retorna metadata básica.

    Args:
        model: estimador sklearn (con método fit).
        X_train: features de entrenamiento.
        y_train: target (None para modelos no supervisados).
        fit_kwargs: argumentos extra para fit.

    Returns:
        dict con:
            - task: "training"
            - model: objeto fitted
            - model_type: str
            - n_samples, n_features
            - params: dict de hiperparámetros
    """
    if y_train is not None:
        fitted = model.fit(X_train, y_train, **fit_kwargs)
    else:
        fitted = model.fit(X_train, **fit_kwargs)

    n_samples, n_features = X_train.shape

    result = {
        "task": "training",
        "model": fitted,
        "model_type": type(fitted).__name__,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "params": fitted.get_params() if hasattr(fitted, "get_params") else {},
    }

    _log_ml_step(
        "train_model",
        {
            "model_type": result["model_type"],
            "n_samples": result["n_samples"],
            "n_features": result["n_features"],
        },
    )

    return result


def train_tree_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Wrapper para DecisionTreeClassifier."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, **kwargs)
    res = train_model(model, X_train, y_train)
    res["task"] = "training_classification"
    return res


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Wrapper para RandomForestClassifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )
    res = train_model(model, X_train, y_train)
    res["task"] = "training_classification"
    return res


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    penalty: str = "l2",
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Wrapper para LogisticRegression."""
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )
    res = train_model(model, X_train, y_train)
    res["task"] = "training_classification"
    return res


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Wrapper para LinearRegression."""
    model = LinearRegression(**kwargs)
    res = train_model(model, X_train, y_train)
    res["task"] = "training_regression"
    return res


def train_kmeans(
    X: pd.DataFrame,
    *,
    n_clusters: int = 3,
    random_state: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Wrapper para KMeans (no supervisado)."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    res = train_model(model, X, y_train=None)
    res["task"] = "training_clustering"
    res["n_clusters"] = int(n_clusters)
    return res


# =============================================================================
# 4. CLASSIFICATION DIAGNOSTICS
# =============================================================================

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average: str = "weighted",
    plot: bool = False,
    model_name: Optional[str] = None,   
) -> Dict[str, Any]:

    """
    Evalúa un modelo de clasificación de forma numérica y opcionalmente visual.

    Args:
        y_true: etiquetas verdaderas.
        y_pred: etiquetas predichas.
        y_proba: probabilidades predichas (opcional).
        class_names: nombres de clases (opcional).
        average: esquema de averaging para métricas multiclase.
        plot: si True, genera matriz de confusión.

    Returns:
        dict con:
            - task: "classification"
            - metrics: {accuracy, precision, recall, f1}
            - confusion_matrix: np.ndarray
            - class_names: lista de nombres
            - y_true, y_pred, y_proba
            - figures: {"confusion_matrix": fig?}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [str(c) for c in np.unique(y_true)]

    figs: Dict[str, plt.Figure] = {}
    
    if plot and class_names is not None:
        apply_style()
        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = sns.light_palette(
            getattr(CONFIG.Style, "TEAL", "#13d6c1") if CONFIG else "#13d6c1",
            as_cmap=True,
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        title = "Confusion Matrix"
        if model_name:
            title += f" — {model_name}"   
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        figs["confusion_matrix"] = fig
        if _auto_plot_enabled():
            plt.show()


    result = {
        "task": "classification",
        "model_name": model_name,  
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        },
        "confusion_matrix": cm,
        "class_names": class_names,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "figures": figs,
    }

    _log_ml_step(
        "evaluate_classification",
        {    
            "model_name": model_name,   
            "n_samples": int(len(y_true)),
            "n_classes": int(len(class_names)),
            "accuracy": float(acc),
        },
    )

    return result


# Alias por compatibilidad
#diagnose_classification = evaluate_classification


# =============================================================================
# 5. REGRESSION DIAGNOSTICS
# =============================================================================

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    plot: bool = False,
) -> Dict[str, Any]:
    """
    Evalúa un modelo de regresión con métricas y diagnósticos de residuos.

    Args:
        y_true: valores verdaderos.
        y_pred: predicciones.
        plot: si True, genera un panel de diagnósticos (4 subplots).

    Returns:
        dict con:
            - task: "regression"
            - metrics: {mae, mse, rmse, r2}
            - residuals: np.ndarray
            - figures: {"diagnostics": fig?}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    residuals = y_true - y_pred

    figs: Dict[str, plt.Figure] = {}
    if plot:
        apply_style()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax1, ax2, ax3, ax4 = axes.flatten()

        # Actual vs Predicted
        ax1.scatter(y_true, y_pred)
        ax1.set_title("Actual vs Predicted")
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.grid(True, alpha=0.3)

        # Residuals vs Predicted
        ax2.scatter(y_pred, residuals)
        ax2.axhline(0, color="gray", linestyle="--")
        ax2.set_title("Residuals vs Predicted")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residuals")
        ax2.grid(True, alpha=0.3)

        # Histogram of residuals
        ax3.hist(residuals, bins=30)
        ax3.set_title("Residual Distribution")
        ax3.set_xlabel("Residual")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)

        # QQ plot
        stats.probplot(residuals, plot=ax4)
        ax4.set_title("QQ Plot (Residuals)")

        plt.tight_layout()
        figs["diagnostics"] = fig
        if _auto_plot_enabled():
            plt.show()

    result = {
        "task": "regression",
        "metrics": {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
        },
        "residuals": residuals,
        "figures": figs,
    }

    _log_ml_step(
        "evaluate_regression",
        {
            "n_samples": int(len(y_true)),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        },
    )

    return result


# Alias por compatibilidad
diagnose_regression = evaluate_regression


# =============================================================================
# 6. CLUSTERING DIAGNOSTICS
# =============================================================================

def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "silhouette",
    plot: bool = False,
    model_name: Optional[str] = None,   # 👈 NUEVO: nombre del algoritmo
) -> Dict[str, Any]:
    """
    Evalúa calidad de clustering con métricas sencillas.

    Args:
        X: matriz de features (n_samples x n_features).
        labels: etiquetas de cluster por observación.
        metric: actualmente soporta "silhouette".
        plot: si True, genera gráfico simple de tamaño de clusters.
        model_name: nombre del algoritmo de clustering (ej. "KMeans (k=4)").
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of rows.")

    sizes = pd.Series(labels).value_counts().sort_index()
    n_clusters = int(sizes.shape[0])

    silhouette_val: Optional[float] = None
    if metric == "silhouette" and n_clusters > 1:
        try:
            silhouette_val = float(silhouette_score(X, labels))
        except Exception:
            silhouette_val = None

    figs: Dict[str, plt.Figure] = {}
    if plot:
        apply_style()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(sizes.index.astype(str), sizes.values)

        title = "Cluster Sizes"
        if model_name:                      # 👈 añadimos el subtítulo
            title += f" — {model_name}"
        ax.set_title(title)

        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of observations")
        ax.grid(True, alpha=0.3)
        figs["cluster_sizes"] = fig
        if _auto_plot_enabled():
            plt.show()

    metrics_dict: Dict[str, Any] = {}
    if silhouette_val is not None:
        metrics_dict["silhouette"] = silhouette_val

    result = {
        "task": "clustering",
        "model_name": model_name,          # 👈 opcional pero útil
        "metrics": metrics_dict,
        "cluster_sizes": sizes,
        "labels": labels,
        "figures": figs,
    }

    _log_ml_step(
        "evaluate_clustering",
        {
            "n_samples": int(X.shape[0]),
            "n_clusters": int(n_clusters),
            "silhouette": float(silhouette_val) if silhouette_val is not None else None,
            "model_name": model_name,      # 👈 queda trazado en el log
        },
    )

    return result



#===============================================
# 7. CLUSTERING DIAGNOSTICS    
#===============================================

def plot_cluster_sizes(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    *,
    cluster_labels: Optional[Dict[Any, str]] = None,
    show_labels: bool = True,
    report: bool = False,
) -> plt.Figure:
    """
    Cluster size bar chart (counts + %), usando la misma paleta que PCA/cluster plots.

    - Colores: CONFIG.Style.VIZ_PALETTE
    - Orden: clusters ordenados por código (0, 1, 2, ...)
    - Labels: 'n (xx.x%)' encima de cada barra
    """

    apply_style()

    # Serie con clusters (sin NAs)
    s = df[cluster_col].dropna()

    if s.empty:
        raise ValueError(f"No data found in column '{cluster_col}'.")

    # Orden por índice para que 0,1,2,... queden fijos
    counts = s.value_counts(sort=False).sort_index()
    total = counts.sum()
    pct = counts / total * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))

    # Usamos order=counts.index para fijar el mapping categoría→color
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        ax=ax,
        palette=CONFIG.Style.VIZ_PALETTE,
        order=counts.index.astype(str),
    )

    # Ticks con nombres “bonitos” si vienen en cluster_labels
    if cluster_labels is not None:
        tick_labels = [cluster_labels.get(c, str(c)) for c in counts.index]
    else:
        tick_labels = [str(c) for c in counts.index]
    ax.set_xticklabels(tick_labels)

    ax.set_ylabel("Customers")
    ax.set_xlabel(cluster_col)
    ax.set_title(
        "Cluster Size Distribution",
        fontsize=15,
        color=CONFIG.Style.NAVY,
    )

    # Etiquetas n + %
    if show_labels:
        for i, p in enumerate(ax.patches):
            h = p.get_height()
            if h <= 0:
                continue
            label = f"{int(counts.iloc[i])} ({pct.iloc[i]:.1f}%)"
            ax.text(
                p.get_x() + p.get_width() / 2,
                h * 1.01,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.grid(axis="y", alpha=0.2)

    if report and CONFIG.verbosity != "silent":
        narrative_from_dict({
            "insight": "Cluster size distribution.",
            "evidence": "Counts and percentages per cluster.",
            "interpretation": "Shows dominant and niche segments.",
            "action": "Use to prioritise which clusters to target first.",
            "risk": "Very small clusters may be unstable or noisy.",
        })

    return fig
    

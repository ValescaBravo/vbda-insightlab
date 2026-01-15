"""
===============================================================================
INSIGHTLAB — VIZ MODULE (Data Visualization)
===============================================================================
Clean, modern, and consistent visualizations following the InsightLab / VBDA 
editorial style.

Includes:
    • Histogram + KDE
    • Boxplots
    • Scatterplots
    • Correlation heatmap (triangular)
    • Time-Series line chart
    • Category bar chart
    • Cluster scatter plot

Comentarios en español.
Salida en inglés.
===============================================================================
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .core import (
    CONFIG, 
    apply_style,
)

# Try to import the real narrative helper used across InsightLab
try:
    from insightlab.narrative.story import narrative_from_dict  # type: ignore
except Exception:
    try:
        # Fallback por si la estructura de paquetes es distinta
        from insightlab.story import narrative_from_dict  # type: ignore
    except Exception:
        # Último recurso: stub silencioso para no romper viz
        def narrative_from_dict(info: dict) -> str:
            return ""

# =============================================================================
# 1. CUSTOM COLORMAP
# =============================================================================

vbda_cmap = LinearSegmentedColormap.from_list(
    "vbda_cmap",
    [CONFIG.Style.GRAY_LIGHT, CONFIG.Style.TEAL, CONFIG.Style.NAVY]
)

def _set_style():
    """Aplica el estilo global de VBDA/InsightLab."""
    apply_style()

# =============================================================================
# 2. HISTOGRAM + KDE
# =============================================================================

def plot_distribution(df, col, bins=30, kde=True, report=False):
    """Genera histograma limpio y profesional (KDE opcional)."""

    _set_style()

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        df[col].dropna(),
        bins=bins,
        kde=kde,
        ax=ax,
        color=CONFIG.Style.NAVY,
        edgecolor="white"
    )

    ax.set_title(
        f"Distribution of {col}",
        fontsize=16,
        color=CONFIG.Style.NAVY,
        pad=14
    )
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

    if report and CONFIG.verbosity != "silent":
        narrative_from_dict({
            "insight": f"Distribution of '{col}' reviewed.",
            "evidence": "Histogram and density curve displayed.",
            "interpretation": "Assess if the variable is skewed or contains outliers.",
            "action": "Consider transformations if the distribution is heavily skewed.",
            "risk": "Visual inspection complements, but does not replace statistical testing."
        })

    return fig

# =============================================================================
# 3. BOX PLOT
# =============================================================================

def plot_box(df, col, by=None, group=None, report=False):
    """
    Genera un boxplot claro y moderno.

    Args:
        df: DataFrame de entrada
        col: nombre de la columna numérica a graficar
        by: columna categórica para agrupar (nuevo nombre preferido)
        group: alias legacy para compatibilidad hacia atrás
        report: si True, genera narrativa
    """

    _set_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    # Soportar ambos nombres: priorizar `by` si viene definido
    grouping_col = by if by is not None else group

    if grouping_col:
        sns.boxplot(
            data=df,
            x=grouping_col,
            y=col,
            palette=[CONFIG.Style.TEAL],
            ax=ax
        )
        title = f"{col} by {grouping_col}"
    else:
        sns.boxplot(
            x=df[col],
            color=CONFIG.Style.TEAL,
            ax=ax
        )
        title = f"Boxplot of {col}"

    ax.set_title(title, fontsize=16, color=CONFIG.Style.NAVY, pad=14)

    if report and CONFIG.verbosity != "silent":
        narrative_from_dict({
            "insight": "Boxplot displayed.",
            "evidence": f"Distribution of '{col}' inspected visually.",
            "interpretation": "Useful to detect extreme values, spread, and skew.",
            "action": "Investigate outliers and consider transformations if necessary.",
            "risk": "Outliers may represent real behaviour — validate before removing."
        })

    return fig


# =============================================================================
# 4. SCATTER PLOT
# =============================================================================

def plot_scatter(df, x, y, hue=None, report=False):
    """Professional scatter plot with VBDA palette."""

    _set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Construimos los kwargs de forma segura
    scatter_kwargs = dict(
        data=df,
        x=x,
        y=y,
        ax=ax,
    )

    if hue is not None:
        # Caso con categoría → usamos palette completo
        scatter_kwargs["hue"] = hue
        scatter_kwargs["palette"] = CONFIG.Style.VIZ_PALETTE
    else:
        # Caso sin hue → un solo color consistente con tu estilo
        scatter_kwargs["color"] = CONFIG.Style.TEAL

    sns.scatterplot(**scatter_kwargs)

    ax.set_title(
        f"{x} vs {y}",
        fontsize=15,
        color=CONFIG.Style.NAVY
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if report and CONFIG.verbosity != "silent":
        narrative_from_dict({
            "insight": f"Scatter plot of '{x}' vs '{y}'.",
            "evidence": "Relationship visually inspected.",
            "interpretation": "Possible correlations or clusters.",
            "action": "Use scatter to explore relationships.",
            "risk": "Overplotting may hide structure."
        })

    return fig


# =============================================================================
# 5. CORRELATION HEATMAP (TRIANGULAR)
# =============================================================================

def plot_correlation(
    df,
    figsize=(10, 8),
    cmap=vbda_cmap,
    report=False,
    threshold: float = 0.0,  # NEW: optional threshold
):
    """
    Heatmap triangular de correlaciones con estilo VBDA.

    Args:
        df: DataFrame de entrada.
        figsize: tamaño de la figura (ancho, alto).
        cmap: mapa de colores para el heatmap.
        report: si True, genera narrativa con narrative_from_dict.
        threshold: si > 0, oculta en el gráfico las correlaciones con |r| < threshold.
    """

    _set_style()

    # Solo variables numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Aplicar threshold si se indica
    if threshold is not None and threshold > 0:
        # Valores con |r| < threshold se convierten en NaN (no se muestran)
        corr_display = corr.mask(corr.abs() < threshold)
    else:
        corr_display = corr

    # Máscara para mostrar solo el triángulo superior
    mask = np.triu(np.ones_like(corr_display, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_display,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        cbar=True,
        square=True,
        ax=ax,
        annot_kws={"size": 9, "color": CONFIG.Style.GRAY_DARK},
    )

    title = "Correlation Matrix (Upper Triangle)"
    if threshold is not None and threshold > 0:
        title += f" — |r| ≥ {threshold:.2f}"

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        color=CONFIG.Style.NAVY,
        pad=14,
    )

    if report and CONFIG.verbosity != "silent":
        # Contar pares fuertes (excluyendo la diagonal)
        abs_corr = corr.abs()
        # Usamos np.triu para evitar contar el mismo par dos veces
        strong_mask = np.triu(abs_corr >= (threshold if threshold > 0 else 0.0), k=1)
        n_strong = int(strong_mask.sum())

        narrative_from_dict({
            "insight": "Correlation matrix generated.",
            "evidence": (
                f"{numeric_df.shape[1]} numeric variables analysed. "
                + (
                    f"{n_strong} variable pairs have |r| ≥ {threshold:.2f}."
                    if threshold and threshold > 0
                    else "All pairwise correlations are displayed."
                )
            ),
            "interpretation": (
                "Identifies strong relationships and potential collinearity between features."
            ),
            "action": (
                "Remove or combine highly correlated features before modelling, "
                "especially when |r| is high."
            ),
            "risk": "Remember: correlation does not imply causality.",
        })

    return fig


# =============================================================================
# 6. TIME SERIES PLOT
# =============================================================================

def plot_time_series(df, time_col, value_col, report=False):
    """Gráfico de línea elegante para series temporales."""

    _set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x=time_col,
        y=value_col,
        color=CONFIG.Style.NAVY,
        ax=ax
    )

    ax.set_title(
        f"Time Series: {value_col}",
        fontsize=16,
        color=CONFIG.Style.NAVY,
        pad=14
    )

    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)

    if report and CONFIG.verbosity != "silent":
        narrative_from_dict({
            "insight": "Time series reviewed.",
            "evidence": f"Trend over time for '{value_col}'.",
            "interpretation": "Useful to detect seasonality, trends, and anomalies.",
            "action": "Consider smoothing, forecasting, or anomaly detection.",
            "risk": "Missing dates or irregular intervals may distort interpretation."
        })

    return fig

# =============================================================================
# 7. CATEGORY PLOT
# =============================================================================

def plot_categories(df, col, top=None, report=False):
    """Gráfico de barras para categorías con paleta VBDA.

    Args:
        df: DataFrame con los datos.
        col: Nombre de la columna categórica.
        top: Si se indica (int), muestra solo las top N categorías por frecuencia.
        report: Si True, genera narrativa IEIAR.
    """
    _set_style()

    counts = df[col].value_counts(dropna=False)

    # Si se especifica `top`, nos quedamos con las N categorías más frecuentes
    if top is not None:
        try:
            top_int = int(top)
        except (TypeError, ValueError):
            top_int = None
        else:
            if top_int > 0 and top_int < len(counts):
                counts = counts.iloc[:top_int]

    # Ajustar paleta al número de barras para evitar el UserWarning de seaborn
    palette = CONFIG.Style.VIZ_PALETTE
    try:
        # Si es lista/tupla, recortamos; si no, lo dejamos tal cual
        if hasattr(palette, "__len__") and not isinstance(palette, dict):
            palette_plot = list(palette)[: len(counts)]
        else:
            palette_plot = palette
    except Exception:
        palette_plot = CONFIG.Style.VIZ_PALETTE

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        palette=palette_plot,
        ax=ax,
    )

    ax.set_title(
        f"Category Composition: {col}",
        fontsize=16,
        color=CONFIG.Style.NAVY,
        pad=14,
    )
    ax.set_ylabel("Count")
    ax.set_xlabel(col)

    plt.xticks(rotation=30, ha="right")

    if report and CONFIG.verbosity != "silent":
        if top is not None and len(counts) > 0:
            evidence_text = (
                f"Top {len(counts)} categories by frequency are displayed for '{col}'."
            )
        else:
            evidence_text = "Count per category displayed."

        narrative_from_dict(
            {
                "insight": f"Category distribution for '{col}'.",
                "evidence": evidence_text,
                "interpretation": "Detects class imbalance or sparse categories.",
                "action": "Consider grouping rare categories when appropriate.",
                "risk": "Small categories may produce unstable model estimates.",
            }
        )

    return fig


# =============================================================================
# 8. CLUSTER VISUALIZATION
# =============================================================================

def plot_cluster(df, x, y, labels, report=False):
    """Gráfico dispersión de clusters (2D)."""

    _set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        x=df[x],
        y=df[y],
        hue=labels,
        palette=CONFIG.Style.VIZ_PALETTE,
        s=70,
        alpha=0.85,
        ax=ax,
        edgecolor="white"
    )

    ax.set_title(
        "Cluster Visualization",
        fontsize=16,
        color=CONFIG.Style.NAVY,
        pad=14
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if report and CONFIG.verbosity != "silent":
        narrative_from_dict({
            "insight": "Cluster separation observed.",
            "evidence": "Cluster labels shown in 2D projection.",
            "interpretation": "Identifies segmentation patterns or overlap.",
            "action": "Use PCA/UMAP to improve separation if needed.",
            "risk": "Dimensionality reduction may distort cluster shape."
        })

    return fig


# =============================================================================
# 9. GENERIC BAR PLOT (VALUE BY CATEGORY)
# =============================================================================

def plot_bar(
    df,
    x,
    y,
    order=None,
    horizontal: bool = False,
    report: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    """
    Generic bar plot for aggregated values (e.g. avg spending by band).

    Args:
        df: DataFrame already aggregated (one row per category).
        x: column name to use on the category axis.
        y: column name with the numeric value to plot.
        order: optional explicit order for categories on the axis.
        horizontal: if True, horizontal bars (y vs x).
        report: if True, send a short narrative via `narrative_from_dict`.
        title: optional custom title.
        xlabel: optional custom label for the x-axis.
        ylabel: optional custom label for the y-axis.
    """

    _set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_kwargs = dict(data=df)

    # Orden de categorías si se especifica
    if order is not None:
        plot_kwargs["order"] = order

    # Un solo color sólido coherente con VBDA
    color = CONFIG.Style.TEAL

    if horizontal:
        sns.barplot(
            x=y,
            y=x,
            color=color,
            ax=ax,
            **plot_kwargs,
        )
    else:
        sns.barplot(
            x=x,
            y=y,
            color=color,
            ax=ax,
            **plot_kwargs,
        )

    # Título y etiquetas
    if title is None:
        title = f"{y} by {x}"

    ax.set_title(
        title,
        fontsize=16,
        color=CONFIG.Style.NAVY,
        pad=14,
    )

    ax.set_xlabel(xlabel if xlabel is not None else (x if not horizontal else y))
    ax.set_ylabel(ylabel if ylabel is not None else (y if not horizontal else x))

    if not horizontal:
        plt.xticks(rotation=20, ha="right")

    if report and CONFIG.verbosity != "silent":
        narrative_from_dict({
            "insight": f"Bar chart of '{y}' by '{x}'.",
            "evidence": f"One value per category in '{x}' is displayed.",
            "interpretation": "Helps compare average or total values across groups.",
            "action": "Use this view to identify high- and low-performing groups.",
            "risk": "Be mindful of very small groups, which may give unstable averages."
        })

    return fig


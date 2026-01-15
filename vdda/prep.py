"""
===============================================================================
INSIGHTLAB – DATA PREPARATION MODULE (Final Hybrid + Trace-Ready Edition)
===============================================================================
Clean, efficient and metadata-rich data preparation engine.

This module:
    • Detects variable types (enhanced)
    • Handles missing values
    • Flags missing indicators
    • Treats outliers (IQR, percentile, z-score)
    • Flags outliers (binary indicators)
    • Scales + normalises numeric variables
    • Encodes categorical variables
    • Extracts datetime features
    • Discretises variables
    • Removes constant or high-missing columns
    • Generates structured metadata for narrative engines (story_explore.py, story_prep.py)
    • Cleaner class logs all transformations + metadata
    • (Optionally) hooks into global CONFIG.trace if available

Comentarios: Español
Salidas técnicas: Inglés
===============================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    PowerTransformer,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 0. OPTIONAL TRACE HOOK (non-breaking)
# =============================================================================

try:  # pragma: no cover - optional import
    # En un paquete real: insightlab.core define CONFIG
    from insightlab.core import CONFIG  # type: ignore
except Exception:  # pragma: no cover - no dependency hard-fail
    CONFIG = None  # type: ignore


def _log_prep_step(step: str, details: Dict[str, Any]) -> None:
    """
    Helper opcional para registrar pasos de preparación de datos
    en el motor de trazas global (si existe).

    No hace nada si CONFIG o CONFIG.trace no están disponibles.
    """
    if CONFIG is None:
        return

    trace = getattr(CONFIG, "trace", None)
    if trace is None:
        return

    try:
        trace.log(stage="prep", step=step, details=details)
    except Exception:
        # Nunca romper el flujo analítico por problemas de logging.
        pass


# =============================================================================
# 1. TYPE DETECTION
# =============================================================================

def detect_types(df: pd.DataFrame, id_threshold: float = 0.95) -> Dict[str, List[str]]:
    """
    Detecta tipos de columnas de forma avanzada.
    Devuelve metadata usable para narrativa.

    id_threshold:
        Proporción de valores únicos / n_rows para considerar una columna como ID-like.
    """

    result: Dict[str, List[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "binary": [],
        "constant": [],
        "id_like": [],
        "high_cardinality": [],
    }

    n = len(df)

    for col in df.columns:
        series = df[col]

        # Constantes
        if series.nunique(dropna=False) == 1:
            result["constant"].append(col)
            continue

        # ID-like (muchos valores únicos)
        if n > 0 and series.nunique(dropna=False) / n >= id_threshold:
            result["id_like"].append(col)
            continue

        dtype = series.dtype

        # Numéricas
        if pd.api.types.is_numeric_dtype(dtype):
            result["numeric"].append(col)
            # binaria (0/1, True/False, etc.)
            if series.nunique(dropna=True) == 2:
                result["binary"].append(col)

        # Fechas
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            result["datetime"].append(col)

        # Categóricas
        else:
            result["categorical"].append(col)
            if series.nunique(dropna=True) == 2:
                result["binary"].append(col)
            if series.nunique(dropna=True) > 50:
                result["high_cardinality"].append(col)

    return result


def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Devuelve columnas numéricas."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_cols(df: pd.DataFrame) -> List[str]:
    """Devuelve columnas categóricas."""
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def get_datetime_cols(df: pd.DataFrame) -> List[str]:
    """Devuelve columnas datetime."""
    return [
        col for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
    ]



# =============================================================================
# 2. VALIDATION + COLUMN NORMALISATION
# =============================================================================

def validate_columns(df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[str]:
    """Valida que las columnas existen en el DataFrame."""
    if cols is None:
        return df.columns.tolist()

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    return cols


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normaliza nombres de columnas:
        - strip espacios
        - pasa a minúsculas
        - reemplaza espacios por _
        - elimina caracteres no alfanuméricos
    """
    old_cols = df.columns.tolist()
    new_cols = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )

    df2 = df.copy()
    df2.columns = new_cols

    metadata = {
        "old_columns": old_cols,
        "new_columns": new_cols.tolist(),
        "renamed_count": int(sum(o != n for o, n in zip(old_cols, new_cols))),
    }

    return df2, metadata


# =============================================================================
# 3. DUPLICATES
# =============================================================================

def drop_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Elimina duplicados y devuelve metadata."""
    before = len(df)
    out = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed = before - len(out)
    meta = {"duplicates_removed": int(removed), "rows_before": int(before), "rows_after": int(len(out))}
    return out, meta


def flag_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Marca duplicados con una columna booleana `_duplicate`."""
    dup_series = df.duplicated(subset=subset, keep="first")
    df2 = df.copy()
    df2["_duplicate"] = dup_series.astype(int)
    meta = {"duplicate_flag_column": "_duplicate", "n_duplicates": int(dup_series.sum())}
    return df2, meta


# =============================================================================
# 4. MISSING VALUES
# =============================================================================

def fill_missing(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    datetime_strategy: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Rellena valores faltantes por tipo.

    numeric_strategy: "mean" | "median" | "zero"
    categorical_strategy: "mode" | "missing"
    datetime_strategy: None | "forward_fill" | "backward_fill"
    """

    df2 = df.copy()
    types = detect_types(df)
    meta: Dict[str, Any] = {"missing_before": int(df.isna().sum().sum())}

    # Numéricas
    for col in types["numeric"]:
        if numeric_strategy == "mean":
            value = df2[col].mean()
        elif numeric_strategy == "median":
            value = df2[col].median()
        elif numeric_strategy == "zero":
            value = 0
        else:
            raise ValueError("numeric_strategy must be 'mean', 'median' or 'zero'")
        df2[col] = df2[col].fillna(value)

    # Categóricas
    for col in types["categorical"]:
        if categorical_strategy == "mode":
            mode = df2[col].mode()
            fill_value = mode.iloc[0] if len(mode) else "missing"
        elif categorical_strategy == "missing":
            fill_value = "missing"
        else:
            raise ValueError("categorical_strategy must be 'mode' or 'missing'")
        df2[col] = df2[col].fillna(fill_value)

    # Fechas
    for col in types["datetime"]:
        if datetime_strategy == "forward_fill":
            df2[col] = df2[col].ffill()
        elif datetime_strategy == "backward_fill":
            df2[col] = df2[col].bfill()
        elif datetime_strategy is None:
            # No imputar fechas si no se especifica estrategia
            pass
        else:
            raise ValueError("datetime_strategy must be None, 'forward_fill' or 'backward_fill'")

    meta["missing_after"] = int(df2.isna().sum().sum())
    meta["missing_filled"] = int(meta["missing_before"] - meta["missing_after"])

    return df2, meta


def flag_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Crea columnas *_missing para variables con NA."""
    df2 = df.copy()
    created: List[str] = []
    for col in df.columns:
        mask = df[col].isna()
        if mask.any():
            new_col = f"{col}_missing"
            df2[new_col] = mask.astype(int)
            created.append(new_col)
    meta = {"missing_indicators_created": created}
    return df2, meta


# =============================================================================
# 5. OUTLIER TREATMENT
# =============================================================================

def cap_outliers_iqr(
    df: pd.DataFrame,
    multiplier: float = 1.5,
    cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Tratamiento de outliers por IQR (clipping)."""
    df2 = df.copy()
    if cols is None:
        cols = get_numeric_cols(df2)
    else:
        cols = validate_columns(df2, cols)

    before: Dict[str, int] = {}
    after: Dict[str, int] = {}
    thresholds: Dict[str, Dict[str, float]] = {}

    for col in cols:
        series = df2[col].astype(float)
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        low, high = q1 - multiplier * iqr, q3 + multiplier * iqr

        before[col] = int(((series < low) | (series > high)).sum())
        df2[col] = series.clip(lower=low, upper=high)
        after[col] = int(((df2[col] < low) | (df2[col] > high)).sum())
        thresholds[col] = {"low": float(low), "high": float(high)}

    meta: Dict[str, Any] = {
        "method": "iqr",
        "multiplier": float(multiplier),
        "outliers_before": before,
        "outliers_after": after,
        "thresholds": thresholds,
    }
    return df2, meta


def cap_outliers_percentile(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Winsorización por percentiles [lower, upper]."""
    df2 = df.copy()
    if cols is None:
        cols = get_numeric_cols(df2)
    else:
        cols = validate_columns(df2, cols)

    thresholds: Dict[str, Dict[str, float]] = {}

    for col in cols:
        series = df2[col].astype(float)
        lo, hi = series.quantile(lower), series.quantile(upper)
        thresholds[col] = {"lower": float(lo), "upper": float(hi)}
        df2[col] = series.clip(lo, hi)

    meta: Dict[str, Any] = {
        "method": "percentile",
        "lower": float(lower),
        "upper": float(upper),
        "thresholds": thresholds,
    }
    return df2, meta


def cap_outliers_zscore(
    df: pd.DataFrame,
    z_thresh: float = 3.0,
    cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Winsorización por z-score.

    Valores con |z| > z_thresh se recortan a mean ± z_thresh * std.
    """
    df2 = df.copy()
    if cols is None:
        cols = get_numeric_cols(df2)
    else:
        cols = validate_columns(df2, cols)

    thresholds: Dict[str, Dict[str, float]] = {}
    before: Dict[str, int] = {}
    after: Dict[str, int] = {}

    for col in cols:
        series = df2[col].astype(float)
        mean = series.mean()
        std = series.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            continue
        low = mean - z_thresh * std
        high = mean + z_thresh * std
        mask = (series < low) | (series > high)
        before[col] = int(mask.sum())
        df2[col] = series.clip(low, high)
        after[col] = int(((df2[col] < low) | (df2[col] > high)).sum())
        thresholds[col] = {"low": float(low), "high": float(high)}

    meta: Dict[str, Any] = {
        "method": "zscore",
        "z_thresh": float(z_thresh),
        "outliers_before": before,
        "outliers_after": after,
        "thresholds": thresholds,
    }
    return df2, meta


def flag_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    cols: Optional[List[str]] = None,
    z_thresh: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Crea indicadores *_outlier para columnas numéricas.

    method: "iqr" | "zscore"
    """
    df2 = df.copy()
    if cols is None:
        cols = get_numeric_cols(df2)
    else:
        cols = validate_columns(df2, cols)

    created: Dict[str, int] = {}
    counts: Dict[str, int] = {}

    for col in cols:
        series = df2[col].astype(float)

        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                continue
            low, high = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
            mask = (series < low) | (series > high)
        elif method == "zscore":
            mean = series.mean()
            std = series.std(ddof=0)
            if not np.isfinite(std) or std == 0:
                continue
            z = (series - mean) / std
            mask = z.abs() > z_thresh
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

        new_col = f"{col}_outlier"
        df2[new_col] = mask.astype(int)
        created[new_col] = int(mask.sum())
        counts[col] = int(mask.sum())

    meta = {
        "method": method,
        "created_flags": list(created.keys()),
        "outlier_counts": counts,
        "z_thresh": float(z_thresh),
        "iqr_multiplier": float(iqr_multiplier),
    }
    return df2, meta



# =============================================================================
# 6. SCALING
# =============================================================================

def scale_numeric(
    df: pd.DataFrame,
    method: str = "standard",
    cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Escala variables numéricas y guarda metadata.

    method:
        - "standard" → media 0, varianza 1
        - "minmax"   → [0, 1]
    """
    df2 = df.copy()
    if cols is None:
        cols = get_numeric_cols(df2)
    else:
        cols = validate_columns(df2, cols)

    if not cols:
        return df2, {"scaled_columns": [], "method": method}

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")

    df2[cols] = scaler.fit_transform(df2[cols])

    return df2, {
        "scaled_columns": cols,
        "method": method,
        "scaler": scaler,
    }


# =============================================================================
# 7. TRANSFORMATIONS
# =============================================================================

def log_transform(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    add_constant: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aplica transformación logarítmica (útil para distribuciones sesgadas).

    Notas:
        - Se suma add_constant antes del log.
        - Valores donde (x + add_constant) <= 0 se convierten en NaN
          para evitar -inf y errores numéricos.
    """
    df2 = df.copy()
    if cols is None:
        cols = get_numeric_cols(df2)
    else:
        cols = validate_columns(df2, cols)

    for col in cols:
        shifted = df2[col].astype(float) + add_constant
        mask_non_positive = shifted <= 0
        shifted[mask_non_positive] = np.nan
        df2[col] = np.log(shifted)

    return df2, {"log_transformed": cols, "add_constant": float(add_constant)}


def power_transform(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    method: str = "yeo-johnson",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aplica transformación de potencia (Yeo-Johnson o Box-Cox) para normalizar.

    method:
        - "yeo-johnson" (admite ceros y negativos)
        - "box-cox" (requiere valores > 0)
    """
    df2 = df.copy()
    if cols is None:
        cols = get_numeric_cols(df2)
    else:
        cols = validate_columns(df2, cols)

    if not cols:
        return df2, {"power_transformed": [], "method": method}

    transformer = PowerTransformer(method=method, standardize=False)
    df2[cols] = transformer.fit_transform(df2[cols].values)

    meta = {
        "power_transformed": cols,
        "method": method,
        "lambdas": transformer.lambdas_.tolist(),
    }
    return df2, meta


# =============================================================================
# 8. ENCODING
# =============================================================================

def encode_onehot(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    drop_first: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    One-hot encoding para variables categóricas.
    """
    df2 = df.copy()
    if cols is None:
        cols = get_categorical_cols(df2)
    else:
        cols = validate_columns(df2, cols)

    if not cols:
        return df2, {"onehot_encoded": [], "n_dummy_columns": 0}

    out = pd.get_dummies(df2, columns=cols, drop_first=drop_first)
    meta = {
        "onehot_encoded": cols,
        "drop_first": bool(drop_first),
        "n_dummy_columns": int(out.shape[1] - df2.shape[1]),
    }
    return out, meta


def encode_ordinal(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encoding ordinal simple (categorías se asignan a 0,1,2,...) por orden alfabético.
    """
    df2 = df.copy()
    if cols is None:
        cols = get_categorical_cols(df2)
    else:
        cols = validate_columns(df2, cols)

    if not cols:
        return df2, {"ordinal_encoded": [], "categories": []}

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df2[cols] = enc.fit_transform(df2[cols].astype("category"))

    meta = {
        "ordinal_encoded": cols,
        "categories": [list(c) for c in enc.categories_],
    }
    return df2, meta


def encode_target(
    df: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simple target / mean encoding.

    ADVERTENCIA: puede introducir leakage si se hace antes de split train/test.
    Idealmente se usa solo en el set de entrenamiento.
    """
    df2 = df.copy()
    y_series = pd.Series(y, index=df2.index, name="__target__")

    if cols is None:
        cols = get_categorical_cols(df2)
    else:
        cols = validate_columns(df2, cols)

    if not cols:
        return df2, {"target_encoded": {}, "warning": "no categorical columns"}

    mappings: Dict[str, Dict[Any, float]] = {}

    for col in cols:
        tmp = pd.DataFrame({"col": df2[col], "y": y_series})
        means = tmp.groupby("col")["y"].mean()
        df2[col] = df2[col].map(means)
        mappings[col] = means.to_dict()

    meta = {
        "target_encoded": {c: {"n_categories": len(mappings[c])} for c in mappings},
        "warning": "Use only on training data to avoid leakage.",
    }
    return df2, meta


def encode_categoricals(
    df: pd.DataFrame,
    method: str = "onehot",
    cols: Optional[List[str]] = None,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    drop_first: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Helper de alto nivel para encoding categórico.

    method:
        - "onehot"  → usa encode_onehot
        - "ordinal" → usa encode_ordinal
        - "target"  → usa encode_target (requiere y)
    """
    method_norm = (method or "").lower()

    if method_norm in ("onehot", "dummy"):
        out, meta = encode_onehot(df, cols=cols, drop_first=drop_first)
        meta["method"] = "onehot"
        return out, meta

    if method_norm in ("ordinal", "ordered"):
        out, meta = encode_ordinal(df, cols=cols)
        meta["method"] = "ordinal"
        return out, meta

    if method_norm in ("target", "mean"):
        if y is None:
            raise ValueError("encode_categoricals(method='target') requires y.")
        out, meta = encode_target(df, y=y, cols=cols)
        meta["method"] = "target"
        return out, meta

    raise ValueError("method must be 'onehot', 'ordinal' or 'target'")


# =============================================================================
# 9. DATE FEATURES
# =============================================================================

def extract_date_features(
    df: pd.DataFrame,
    col: str,
    features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extrae features de fecha comunes: year, month, day, dayofweek, quarter.
    """
    df2 = df.copy()
    if col not in df2.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    if not pd.api.types.is_datetime64_any_dtype(df2[col]):
        df2[col] = pd.to_datetime(df2[col], errors="coerce")

    feats = features or ["year", "month", "day", "dayofweek", "quarter"]
    created: List[str] = []

    for f in feats:
        if f == "year":
            new_col = f"{col}_year"
            df2[new_col] = df2[col].dt.year
        elif f == "month":
            new_col = f"{col}_month"
            df2[new_col] = df2[col].dt.month
        elif f == "day":
            new_col = f"{col}_day"
            df2[new_col] = df2[col].dt.day
        elif f == "dayofweek":
            new_col = f"{col}_dayofweek"
            df2[new_col] = df2[col].dt.dayofweek
        elif f == "quarter":
            new_col = f"{col}_quarter"
            df2[new_col] = df2[col].dt.quarter
        else:
            continue
        created.append(new_col)

    meta = {"date_features_added": created, "source_column": col}
    return df2, meta


# =============================================================================
# 10. DISCRETISATION
# =============================================================================

def discretize_equal_width(
    df: pd.DataFrame,
    col: str,
    bins: int = 5,
    labels: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Discretización de igual ancho (pd.cut)."""
    df2 = df.copy()
    if col not in df2.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    new_col = f"{col}_bin"
    df2[new_col] = pd.cut(df2[col].astype(float), bins=bins, labels=labels)
    return df2, {"equal_width_bins": {col: int(bins)}, "new_column": new_col}


def discretize_equal_freq(
    df: pd.DataFrame,
    col: str,
    q: int = 5,
    labels: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Discretización por cuantiles (pd.qcut)."""
    df2 = df.copy()
    if col not in df2.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    new_col = f"{col}_qbin"
    df2[new_col] = pd.qcut(df2[col].astype(float), q=q, labels=labels, duplicates="drop")
    return df2, {"equal_freq_bins": {col: int(q)}, "new_column": new_col}


# =============================================================================
# 11. QUALITY REPORT & COLUMN FILTERS
# =============================================================================

def remove_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Elimina columnas constantes."""
    nunique = df.nunique(dropna=False)
    cols = nunique[nunique <= 1].index.tolist()
    df2 = df.drop(columns=cols) if cols else df.copy()
    meta = {"removed_columns": cols, "n_removed": int(len(cols))}
    return df2, meta


def remove_high_missing(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Elimina columnas con proporción de missing >= threshold.
    """
    missing_frac = df.isna().mean()
    cols = missing_frac[missing_frac >= threshold].index.tolist()
    df2 = df.drop(columns=cols) if cols else df.copy()

    meta = {
        "threshold": float(threshold),
        "removed_columns": cols,
        "missing_fraction": {c: float(missing_frac[c]) for c in cols},
    }
    return df2, meta


def data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera un reporte estructurado de calidad de datos,
    alineado con story_explore.story_data_quality.
    """
    types = detect_types(df)
    missing_cnt = df.isna().sum()
    missing_pct = df.isna().mean()

    report: Dict[str, Any] = {
        "shape": df.shape,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "duplicate_count": int(df.duplicated().sum()),
        "missing_summary": missing_cnt.to_dict(),
        "missing_pct": {col: float(p) for col, p in missing_pct.items()},
        "type_summary": {k: len(v) for k, v in types.items()},
        "constant_cols": types["constant"],
        "id_like_cols": types["id_like"],
        "high_cardinality_cols": types["high_cardinality"],
    }
    return report


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Alias público / retrocompatible para data_quality().

    Usado por:
        - story_explore.story_data_quality
        - insightlab.__init__ lazy map
    """
    return data_quality(df)


# =============================================================================
# 12. TRAIN/TEST SPLIT HELPER
# =============================================================================

def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    *,
    stratify_by: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Helper ligero para train/test split basado en DataFrame completo.

    stratify_by:
        Nombre de columna a usar como y para estratificar (clasificación).
    """
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")

    if stratify_by is not None:
        if stratify_by not in df.columns:
            raise ValueError(f"Column '{stratify_by}' not found in DataFrame.")
        stratify_vals = df[stratify_by]
    else:
        stratify_vals = None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_vals,
    )

    meta: Dict[str, Any] = {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "stratify_by": stratify_by,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), meta


# =============================================================================
# 13. HIGH-LEVEL SUMMARY
# =============================================================================

def prep_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Resumen de alto nivel de preparación:
        - tipos de variables
        - calidad de datos básica

    Útil para pasar a story_prep o para logs rápidos.
    """
    types = detect_types(df)
    quality = data_quality(df)

    summary = {
        "types": types,
        "quality": quality,
    }

    _log_prep_step("prep_summary", {"n_rows": quality["n_rows"], "n_cols": quality["n_cols"]})
    return summary


# =============================================================================
# 14. PREP RESULT CONTAINER
# =============================================================================

@dataclass
class PrepResult:
    """
    Contenedor estructurado de resultados de preparación.

    df: DataFrame limpio
    log: lista de pasos aplicados (texto)
    metadata: dict con metadata por paso
    quality: reporte de calidad final
    """
    df: pd.DataFrame
    log: List[str]
    metadata: Dict[str, Any]
    quality: Dict[str, Any]


# =============================================================================
# 15. CLEANER CLASS (PIPELINE + METADATA + TRACE)
# =============================================================================

class Cleaner:
    """
    Pipeline de limpieza con metadata enriquecida y logging opcional en trace.

    Uso típico:
        cleaner = Cleaner(df)
        df_clean = (
            cleaner
                .rename()
                .duplicates()
                .missing()
                .outliers(method="iqr")
                .scale()
                .get()
        )
    """

    def __init__(self, df: pd.DataFrame):
        self.data = df.copy()
        self.original = df.copy()
        self.metadata: Dict[str, Any] = {}
        self.log: List[str] = []

    def _update(self, name: str, meta: Dict[str, Any]) -> None:
        """Registra metadata + log + trace."""
        self.metadata[name] = meta
        self.log.append(name)
        _log_prep_step(name, meta)

    # ---------------------------- core steps ---------------------------------

    def rename(self) -> "Cleaner":
        self.data, meta = normalize_columns(self.data)
        self._update("rename", meta)
        return self

    def duplicates(self, subset: Optional[List[str]] = None) -> "Cleaner":
        self.data, meta = drop_duplicates(self.data, subset=subset)
        self._update("duplicates", meta)
        return self

    def missing(
        self,
        numeric: str = "median",
        categorical: str = "mode",
        datetime: Optional[str] = None,
    ) -> "Cleaner":
        self.data, meta = fill_missing(self.data, numeric, categorical, datetime)
        self._update("missing", meta)
        return self

    def missing_flags(self) -> "Cleaner":
        self.data, meta = flag_missing(self.data)
        self._update("missing_flags", meta)
        return self

    # -------------------------------------------------------------------------
    # Outliers: detección NO destructiva  c.detect_outliers()
    # -------------------------------------------------------------------------
    def detect_outliers(
        self,
        method: str = "iqr",
        cols: Optional[List[str]] = None,
        z_thresh: float = 3.0,
        iqr_multiplier: float = 1.5,
    ) -> Dict[str, int]:
        """
        Detección de outliers sin modificar self.data.

        Devuelve un dict {columna: n_outliers} y guarda metadata en
        self.metadata["detect_outliers"] para narrativa.
        """
        df = self.data  # no copiamos, no vamos a modificar

        # Columnas a inspeccionar
        if cols is None:
            cols = get_numeric_cols(df)
        else:
            cols = validate_columns(df, cols)

        outlier_counts: Dict[str, int] = {}

        for col in cols:
            series = df[col].astype(float)

            if method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                if not np.isfinite(iqr) or iqr == 0:
                    outlier_counts[col] = 0
                    continue
                low = q1 - iqr_multiplier * iqr
                high = q3 + iqr_multiplier * iqr
                mask = (series < low) | (series > high)

            elif method == "zscore":
                mean = series.mean()
                std = series.std(ddof=0)
                if not np.isfinite(std) or std == 0:
                    outlier_counts[col] = 0
                    continue
                z = (series - mean) / std
                mask = z.abs() > z_thresh

            else:
                raise ValueError("method must be 'iqr' or 'zscore'")

            outlier_counts[col] = int(mask.sum())

        meta = {
            "method": method,
            "outlier_counts": outlier_counts,
            "z_thresh": float(z_thresh),
            "iqr_multiplier": float(iqr_multiplier),
        }

        # Guardamos metadata y trace, sin mutar datos
        self.metadata["detect_outliers"] = meta
        _log_prep_step("detect_outliers", meta)

        return outlier_counts

    # -------------------------------------------------------------------------
    # Outliers: tratamiento (capping / winsor / z-score)
    # -------------------------------------------------------------------------
    def outliers(
        self,
        method: str = "iqr",
        cols: Optional[List[str]] = None,
        z_thresh: float = 3.0,
        iqr_multiplier: float = 1.5,
    ) -> "Cleaner":
        """
        Trata outliers y almacena metadata bajo la clave estándar 'outliers',
        para que story_prep/story_explore puedan consumirla.

        Usa las funciones de módulo:
            - cap_outliers_iqr
            - cap_outliers_percentile
            - cap_outliers_zscore
        """
        if method == "iqr":
            self.data, meta = cap_outliers_iqr(
                self.data,
                multiplier=iqr_multiplier,
                cols=cols,
            )
        elif method == "percentile":
            self.data, meta = cap_outliers_percentile(
                self.data,
                cols=cols,
            )
        elif method == "zscore":
            self.data, meta = cap_outliers_zscore(
                self.data,
                z_thresh=z_thresh,
                cols=cols,
            )
        else:
            raise ValueError("method must be 'iqr', 'percentile' or 'zscore'")

        meta["method"] = method
        # story_prep espera metadata["outliers"]
        self._update("outliers", meta)
        return self

    def outlier_flags(
        self,
        method: str = "iqr",
        cols: Optional[List[str]] = None,
        z_thresh: float = 3.0,
        iqr_multiplier: float = 1.5,
    ) -> "Cleaner":
        """
        Crea indicadores *_outlier sin modificar los valores originales.

        Guarda la metadata bajo 'outlier_flags' con el campo estándar
        'outlier_flags_created' que espera story_prep._story_outlier_flags.
        """
        self.data, meta = flag_outliers(
            self.data,
            method=method,
            cols=cols,
            z_thresh=z_thresh,
            iqr_multiplier=iqr_multiplier,
        )

        # Compatibilidad con story_prep
        indicators = meta.get("indicator_columns", [])
        meta["outlier_flags_created"] = indicators
        meta["method"] = method

        self._update("outlier_flags", meta)
        return self

    # -------------------------------------------------------------------------
    # Escalado y transformaciones
    # -------------------------------------------------------------------------
    def scale(
        self,
        method: str = "standard",
        cols: Optional[List[str]] = None,
    ) -> "Cleaner":
        self.data, meta = scale_numeric(self.data, method=method, cols=cols)
        self._update("scale", meta)
        return self

    def log_transform(
        self,
        cols: Optional[List[str]] = None,
        add_constant: float = 1.0,
    ) -> "Cleaner":
        self.data, meta = log_transform(
            self.data,
            cols=cols,
            add_constant=add_constant,
        )
        self._update("log_transform", meta)
        return self

    def power_transform(
        self,
        cols: Optional[List[str]] = None,
        method: str = "yeo-johnson",
    ) -> "Cleaner":
        self.data, meta = power_transform(
            self.data,
            cols=cols,
            method=method,
        )
        self._update("power_transform", meta)
        return self

    # -------------------------------------------------------------------------
    # Codificaciones
    # -------------------------------------------------------------------------
    def onehot(
        self,
        cols: Optional[List[str]] = None,
        drop_first: bool = True,
    ) -> "Cleaner":
        self.data, meta = encode_onehot(
            self.data,
            cols=cols,
            drop_first=drop_first,
        )
        self._update("onehot", meta)
        return self

    def ordinal(
        self,
        cols: Optional[List[str]] = None,
    ) -> "Cleaner":
        self.data, meta = encode_ordinal(self.data, cols=cols)
        self._update("ordinal", meta)
        return self

    def target_encode(
        self,
        y: Union[pd.Series, np.ndarray],
        cols: Optional[List[str]] = None,
    ) -> "Cleaner":
        self.data, meta = encode_target(self.data, y=y, cols=cols)
        self._update("target_encode", meta)
        return self

    # -------------------------------------------------------------------------
    # Features de fecha y discretización
    # -------------------------------------------------------------------------
    def date_features(
        self,
        col: str,
        features: Optional[List[str]] = None,
    ) -> "Cleaner":
        self.data, meta = extract_date_features(
            self.data,
            col=col,
            features=features,
        )
        self._update("date_features", meta)
        return self

    def discretize_width(
        self,
        col: str,
        bins: int = 5,
        labels: bool = False,
    ) -> "Cleaner":
        self.data, meta = discretize_equal_width(
            self.data,
            col=col,
            bins=bins,
            labels=labels,
        )
        self._update("discretize_equal_width", meta)
        return self

    def discretize_freq(
        self,
        col: str,
        q: int = 5,
        labels: bool = False,
    ) -> "Cleaner":
        self.data, meta = discretize_equal_freq(
            self.data,
            col=col,
            q=q,
            labels=labels,
        )
        self._update("discretize_equal_freq", meta)
        return self

    # -------------------------------------------------------------------------
    # Filtros de columnas
    # -------------------------------------------------------------------------
    def remove_constants(self) -> "Cleaner":
        self.data, meta = remove_constant_columns(self.data)
        self._update("remove_constant_columns", meta)
        return self

    def remove_high_missing_cols(
        self,
        threshold: float = 0.5,
    ) -> "Cleaner":
        self.data, meta = remove_high_missing(
            self.data,
            threshold=threshold,
        )
        self._update("remove_high_missing_cols", meta)
        return self

    # ------------------------------ outputs ---------------------------------

    def get(self) -> pd.DataFrame:
        """Retorna el DataFrame limpio (copia)."""
        return self.data.copy()

    def report(self) -> Dict[str, Any]:
        """Retorna metadata detallada por paso."""
        return self.metadata

    def summary(self) -> List[str]:
        """Retorna log textual de operaciones."""
        return self.log

    def quality_report(self) -> Dict[str, Any]:
        """Genera reporte de calidad de datos del estado actual."""
        return data_quality(self.data)

    def to_result(self) -> PrepResult:
        """
        Convierte el estado actual del Cleaner en un PrepResult estructurado.
        Ideal para pasarlo a story_explore o export.
        """
        return PrepResult(
            df=self.get(),
            log=self.summary(),
            metadata=self.report(),
            quality=self.quality_report(),
        )

    def reset(self) -> "Cleaner":
        """Reinicia al DataFrame original."""
        self.data = self.original.copy()
        self.metadata = {}
        self.log = []
        _log_prep_step("reset", {})
        return self



# =============================================================================
# Visual Helper
# =============================================================================

def auto_plot_enabled() -> bool:
    return bool(getattr(CONFIG, "auto_plot", True))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Type detection
    "detect_types",
    "get_numeric_cols",
    "get_categorical_cols",
    "get_datetime_cols",
    # Validation & columns
    "normalize_columns",
    "validate_columns",
    # Duplicates
    "drop_duplicates",
    "flag_duplicates",
    # Missing
    "fill_missing",
    "flag_missing",
    # Outliers
    "cap_outliers_iqr",
    "cap_outliers_percentile",
    "cap_outliers_zscore",
    "flag_outliers",
    # Scaling & transforms
    "scale_numeric",
    "log_transform",
    "power_transform",
    # Encoding
    "encode_onehot",
    "encode_ordinal",
    "encode_target",
    "encode_categoricals",
    # Date & discretisation
    "extract_date_features",
    "discretize_equal_width",
    "discretize_equal_freq",
    # Quality & filters
    "remove_constant_columns",
    "remove_high_missing",
    "data_quality",
    "get_data_quality_report",
    # High-level helpers
    "split_train_test",
    "prep_summary",
    "Cleaner",
    "PrepResult",
]


import pandas as pd
import numpy as np
from collections import Counter
from typing import Optional
from data_dictionary_generator.data_type import infer_data_type, infer_column_data_type

def compute_column_stats(
    series: pd.Series,
    sample_size: int = 5000,
    inferred_dtype: Optional[str] = None,
    precomputed_sample: Optional[pd.Series] = None,
) -> dict:
    """
    Compute statistics for a column, based on inferred data type.
    """

    if series.empty:
        return None

    if precomputed_sample is not None:
        non_null_series = precomputed_sample
        if non_null_series.empty:
            return None
    else:
        non_null_series = series.dropna()
        if non_null_series.empty:
            return None

        if len(non_null_series) > sample_size:
            non_null_series = non_null_series.sample(sample_size, random_state=42)

    col_type = inferred_dtype if inferred_dtype is not None else infer_column_data_type(non_null_series)

    stats = {}

    if col_type in ["integer", "float"]:
        numeric = pd.to_numeric(non_null_series, errors="coerce").dropna()
        if not numeric.empty:
            stats = {
                "type": "numeric",
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
                "median": float(numeric.median()),
                "std": float(numeric.std())
            }

    elif col_type == "datetime":
        dates = pd.to_datetime(non_null_series, errors="coerce", format="mixed").dropna()
        if not dates.empty:
            stats = {
                "type": "datetime",
                "min_date": dates.min().isoformat(),
                "max_date": dates.max().isoformat()
            }

    elif col_type in ["string", "boolean"]:
        counts = Counter(non_null_series.astype(str))
        top_values = counts.most_common(10)
        stats = {
            "type": "categorical",
            "top_values": top_values,
            "unique_count": len(counts)
        }

    else:
        return None

    return stats if stats else None

import pandas as pd
import numpy as np
from collections import Counter
from data_type import infer_data_type

def compute_column_stats(series: pd.Series, sample_size: int = 5000) -> dict:
    """
    Compute statistics for a column, based on inferred data type.
    """

    if series.empty:
        return None

    non_null_series = series.dropna()
    if non_null_series.empty:
        return None

    if len(non_null_series) > sample_size:
        non_null_series = non_null_series.sample(sample_size, random_state=42)

    inferred_types = non_null_series.apply(infer_data_type)
    col_type = inferred_types.value_counts().idxmax()

    stats = {}

    if col_type in ["integer", "float"]:
        numeric = pd.to_numeric(non_null_series, errors="coerce").dropna()
        if not numeric.empty:
            stats = {
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
                "min_date": dates.min().isoformat(),
                "max_date": dates.max().isoformat()
            }

    elif col_type in ["string", "boolean"]:
        counts = Counter(non_null_series.astype(str))
        top_values = counts.most_common(10)
        stats = {
            "top_values": top_values,
            "unique_count": len(counts)
        }

    else:
        return None

    return stats if stats else None

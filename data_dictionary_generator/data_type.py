from dateutil.parser import parse as dateparse
import pandas as pd
import re

_date_pattern = re.compile(
    r"(\d{4}[-/]\d{2}[-/]\d{2})"           # 2024-01-31
    r"|(\d{2}[-/]\d{2}[-/]\d{4})"           # 31-01-2024
    r"|(\d{4}[-/]\d{2}[-/]\d{2}T\d{2}:\d{2})"  # 2024-01-31T12:00
)

def infer_data_type(value) -> str:
    if pd.isna(value):
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if not isinstance(value, str):
        return "string"
    v = value.strip()
    if v == "":
        return "null"
    lower = v.lower()
    if lower in ("true", "false", "yes", "no", "y", "n", "t", "f"):
        return "boolean"
    temp_v = v.replace(",", "")
    has_sign = temp_v.startswith(("+", "-"))
    numeric_part = temp_v[1:] if has_sign else temp_v
    if "+" in numeric_part or "-" in numeric_part:
        return "string"
    if numeric_part.isdigit():
        return "integer"
    try:
        if "." in numeric_part:
            if numeric_part.replace(".", "", 1).isdigit():
                return "float"
        float(temp_v)
        return "float"
    except ValueError:
        pass

    if _date_pattern.search(v):
        try:
            dateparse(v, fuzzy=False)
            return "datetime"
        except (ValueError, OverflowError):
            pass

    return "string"

def infer_column_data_type(series: pd.Series, sample_size: int = 100) -> str:
    """
    Infer the most likely data type of a column by sampling non-null values
    and selecting the most frequent inferred type.
    """
    non_null_series = series.dropna()
    if non_null_series.empty:
        return "null"
    sample = non_null_series.sample(
        min(sample_size, len(non_null_series)), random_state=42
    )
    inferred_types = sample.apply(infer_data_type)
    most_common = inferred_types.value_counts().idxmax()
    return most_common

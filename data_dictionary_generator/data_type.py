from dateutil.parser import parse as dateparse
import numpy as np
import pandas as pd

def infer_data_type(value) -> str:
    """
    Infer the most likely data type of a single value.
    """
    if pd.isna(value):
        return "null"

    v = value
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return "null"
        lower = v.lower()
    
    if isinstance(v, bool) or (isinstance(v, str) and lower in ["true", "false", "yes", "no", "y", "n", "t", "f"]):
        return "boolean"
    
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "float"

    if not isinstance(v, str):
        return "string"

    try:
        dateparse(v, fuzzy=False, dayfirst=None)
        return "datetime"
    except:
        pass 
    
    temp_v = v.replace(",", "")
    
    has_sign = temp_v.startswith('+') or temp_v.startswith('-')
    numeric_part = temp_v[1:] if has_sign else temp_v
    
    if '+' in numeric_part or '-' in numeric_part:
        return "string"

    if numeric_part.isdigit():
        return "integer"

    try:
        if '.' in numeric_part:
            numeric_part_no_dot = numeric_part.replace('.', '', 1) 
            if numeric_part_no_dot.isdigit():
                return "float"
        
        float(temp_v)
        return "float"
    except:
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
    
    sample = non_null_series.sample(min(sample_size, len(non_null_series)), random_state=42)
    inferred_types = sample.apply(infer_data_type)
    return inferred_types.value_counts().idxmax()

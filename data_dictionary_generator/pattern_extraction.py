import re
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from dateutil.parser import parse as dateparse
from data_dictionary_generator.data_type import infer_column_data_type

def normalize_pattern(value: str) -> str:
    """Normalize a string into a pattern mask (A/a/9/special)."""
    pattern = []
    for ch in value:
        if ch.isdigit():
            pattern.append("9")
        elif ch.isalpha():
            pattern.append("A" if ch.isupper() else "a")
        else:
            pattern.append(ch)
    return "".join(pattern)

def extract_patterns(series: pd.Series, sample_size: int = 5000) -> pd.Series:
    """Return value_counts of normalized patterns in the column."""
    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return pd.Series([], dtype=int)

    sample = clean.sample(min(sample_size, len(clean)), random_state=42)
    patterns = sample.apply(normalize_pattern)
    return patterns.value_counts()

def detect_semantic_type(series: pd.Series, column_name: str = "") -> Optional[str]:
    """
    Infer semantic type using column name, structural patterns, and value heuristics.
    """
    col = column_name.lower()
    if any(k in col for k in ["subject_id", "patient_id", "stay_id", "mrn", "hadm_id", "id"]):
        return "identifier"
    if any(k in col for k in ["charttime", "admittime", "dischtime", "time", "date", "ts"]):
        return "timestamp"
    if any(k in col for k in ["age"]):
        return "age"
    if any(k in col for k in ["gender", "sex"]):
        return "gender"
    if any(k in col for k in ["ethnicity", "race"]):
        return "ethnicity"
    if any(k in col for k in ["heart_rate", "hr", "bp", "temp"]):
        return "clinical_measurement"

    patterns = extract_patterns(series)
    
    if len(patterns) > 0:
        top = patterns.index[0]

        if re.fullmatch(r"[aA]{3,5}://[aA9\.-]+", top):
             return "url"
        
        if re.search(r"@", top):
            return "email"

        if re.fullmatch(r"9{3}-9{2}-9{4}", top):
             return "ssn"
             
        if re.fullmatch(r"(\(9{3}\) ?|9{3}[- ]?)9{3}[- ]9{4}", top): 
             return "phone_number_us"

        if re.fullmatch(r"9{5}(-9{4})?", top):
            return "zip_code"
        
        if re.search(r"\$", top) and re.search(r"[9]+(\.[9]{2})", top):
            return "currency"
            
        if re.fullmatch(r"A99\.9|A\d{2}\.\d+", top):
            return "icd_code"
            
        if re.fullmatch(r"9{4}[-/ ]9{2}[-/ ]9{2}", top):
            return "date"

        if re.fullmatch(r"9{6,}", top):
            return "identifier_numeric"
        
    inferred_dtype = infer_column_data_type(series) 
    
    if inferred_dtype == "float":
        if series.min() >= 0 and series.max() <= 1:
            return "probability_score"
            
    if inferred_dtype == "boolean":
         return "flag_binary"

    if inferred_dtype == "datetime":
         return "timestamp" 

    if inferred_dtype == "string" or series.dtype == object:
        unique_count = series.nunique(dropna=True)
        if unique_count > 1 and unique_count < 20 and len(series) > 100:
            return "categorical"
            
    return None

def profile_column(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Create a full profile of the column."""
    patterns = extract_patterns(series)
    semantic = detect_semantic_type(series, column_name)

    return {
        "column_name": column_name,
        "semantic_type": semantic or "unknown",
        "unique_patterns": len(patterns),
        "top_patterns": patterns.head(10).to_dict(),
        "example_values": (
            series.dropna().astype(str).sample(min(5, len(series)), random_state=42).tolist()
            if len(series.dropna()) > 0 else []
        ),
    }

def profile_dataframe(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Return patterns + type inference for every column."""
    results = {}
    for col in df.columns:
        results[col] = profile_column(df[col], col)
    return results

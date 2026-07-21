import re
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from dateutil.parser import parse as dateparse
from data_dictionary_generator.data_type import infer_column_data_type

def normalize_pattern(value: str) -> str:
    """Normalize a string into a pattern mask (A/a/9/special)."""
    table = _char_translation_table(set(value))
    return value.translate(table)

def _char_translation_table(unique_chars) -> Dict[int, str]:
    """
    Build a str.translate table mapping each distinct character to its
    pattern class (9/A/a). Characters that fall through (punctuation,
    whitespace, etc.) are left untouched by translate() automatically.
    """
    table: Dict[int, str] = {}
    for ch in unique_chars:
        if ch.isdigit():
            table[ord(ch)] = "9"
        elif ch.isalpha():
            table[ord(ch)] = "A" if ch.isupper() else "a"
    return table

def extract_patterns(
    series: pd.Series,
    sample_size: int = 5000,
    precomputed_sample: Optional[pd.Series] = None,
) -> pd.Series:
    """Return value_counts of normalized patterns in the column.
    """
    if precomputed_sample is not None:
        if len(precomputed_sample) == 0:
            return pd.Series([], dtype=int)
        sample = precomputed_sample.astype(str)
    else:
        clean = series.dropna()
        if len(clean) == 0:
            return pd.Series([], dtype=int)
        sample = clean.sample(min(sample_size, len(clean)), random_state=42).astype(str)

    unique_chars = set("".join(sample))
    table = _char_translation_table(unique_chars)
    patterns = sample.str.translate(table)
    return patterns.value_counts()

_STRICT_BOUNDARY_KEYWORDS = {"ts", "hr", "age"}


def _keyword_hits(col: str, keyword: str) -> bool:
    """Check whether `keyword` appears in `col` as a real word/segment,
    not merely as an accidental substring inside an unrelated word."""
    for m in re.finditer(re.escape(keyword), col):
        start, end = m.start(), m.end()
        before_is_boundary = start == 0 or not col[start - 1].isalpha()
        after_is_boundary = end == len(col) or not col[end].isalpha()
        if keyword in _STRICT_BOUNDARY_KEYWORDS:
            if before_is_boundary and after_is_boundary:
                return True
        else:
            if before_is_boundary or after_is_boundary:
                return True
    return False


def _any_keyword_hits(col: str, keywords) -> bool:
    return any(_keyword_hits(col, k) for k in keywords)


def detect_semantic_type(
    series: pd.Series,
    column_name: str = "",
    patterns: Optional[pd.Series] = None,
    inferred_dtype: Optional[str] = None,
) -> Optional[str]:
    """
    Infer semantic type using column name, structural patterns, and value heuristics.
    """
    col = column_name.lower()
    if _any_keyword_hits(col, ["subject_id", "patient_id", "stay_id", "mrn", "hadm_id", "id"]):
        return "identifier"
    if _any_keyword_hits(col, ["charttime", "admittime", "dischtime", "time", "date", "ts"]):
        return "timestamp"
    if _any_keyword_hits(col, ["age"]):
        return "age"
    if _any_keyword_hits(col, ["gender", "sex"]):
        return "gender"
    if _any_keyword_hits(col, ["ethnicity", "race"]):
        return "ethnicity"
    if _any_keyword_hits(col, ["heart_rate", "hr", "bp", "temp"]):
        return "clinical_measurement"

    if patterns is None:
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
        
    if inferred_dtype is None:
        inferred_dtype = infer_column_data_type(series)

    if inferred_dtype == "float":
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) > 0:
            if numeric.min() >= 0 and numeric.max() <= 1:
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

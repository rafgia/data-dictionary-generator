import pathlib
from typing import Dict, List
import pandas as pd
from pydantic import BaseModel


class RelationshipMeta(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str
    confidence: float

    class Config:
        extra = "allow"

def normalize_column_name(col_name: str) -> str:
    normalized = col_name.lower().strip().replace(" ", "_").replace("-", "_")
    for suffix in ["_id", "id", "_key", "key", "_code", "code"]:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[:-len(suffix)]
            break
    return normalized

def calculate_containment(child: pd.Series, parent: pd.Series, sample_size: int = 10000) -> float:
    child_clean = child.dropna()
    parent_clean = parent.dropna()
    if len(child_clean) == 0 or len(parent_clean) == 0:
        return 0.0
    if len(child_clean) > sample_size:
        child_clean = child_clean.sample(sample_size, random_state=42)
    child_set = set(child_clean.astype(str).unique())
    parent_set = set(parent_clean.astype(str).unique())
    if not child_set:
        return 0.0
    child_values = child_clean.astype(str)
    parent_set = set(parent_clean.astype(str))

    return child_values.isin(parent_set).mean()

def infer_cardinality(parent_series: pd.Series, child_series: pd.Series) -> str:
    parent_unique = parent_series.nunique() / len(parent_series) if len(parent_series) > 0 else 0
    child_unique = child_series.nunique() / len(child_series) if len(child_series) > 0 else 0
    if parent_unique > 0.85 and child_unique < 0.85:
        return "1:N"
    elif parent_unique > 0.85 and child_unique > 0.85:
        return "1:1"
    else:
        return "N:M"

def detect_relationships_between_tables(
    table1_name: str, table1_meta, table1_df: pd.DataFrame,
    table2_name: str, table2_meta, table2_df: pd.DataFrame,
    min_containment_threshold: float = 0.3,
    min_confidence: float = 0.7
) -> List[RelationshipMeta]:
    stats = {
    "total_pairs": 0,
    "parent_invalid": 0,
    "parent_low_unique": 0,
    "child_invalid": 0,
    "dtype_mismatch": 0,
    "low_containment": 0,
    "low_confidence": 0,
    "accepted": 0}

    relationships = []
    temporal_keywords = ["time", "date", "timestamp"]
    

    def is_valid(col_name, col_meta, series):
        name = col_name.lower()
        if any(k in name for k in ["time", "date", "timestamp"]):
            return False
        return True
    table_pairs = [
        (table1_name, table1_meta, table1_df, table2_name, table2_meta, table2_df),
        (table2_name, table2_meta, table2_df, table1_name, table1_meta, table1_df),
    ]
    for parent_table, parent_meta, parent_df, child_table, child_meta, child_df in table_pairs:
        for parent_col, parent_col_meta in parent_meta.columns.items():
            parent_series = parent_df[parent_col]
            parent_unique_ratio = parent_series.nunique() / len(parent_series) if len(parent_series) else 0
            if parent_unique_ratio < 0.9:
                stats["parent_low_unique"] += 1
                continue
            if not is_valid(parent_col, parent_col_meta, parent_series):
                stats["parent_invalid"] += 1
                continue
            parent_unique_ratio = parent_series.nunique() / len(parent_series) if len(parent_series) else 0
            if parent_unique_ratio < 0.8:
                continue
            for child_col, child_col_meta in child_meta.columns.items():
                stats["total_pairs"] += 1
                child_series = child_df[child_col]
                if not is_valid(child_col, child_col_meta, child_series):
                    stats["child_invalid"] += 1
                    continue
                norm_parent = normalize_column_name(parent_col)
                norm_child = normalize_column_name(child_col)
                name_match_score = 0.0
                if norm_parent == norm_child:
                    name_match_score = 0.6
                if "id" in parent_col.lower() and "id" in child_col.lower():
                    name_match_score = max(name_match_score, 0.8)
                containment = calculate_containment(child_series, parent_series)
                if containment < min_containment_threshold:
                    stats["low_containment"] += 1
                    continue
                confidence = max(name_match_score, containment)
                if norm_parent.endswith("id") or norm_child.endswith("id"):
                    confidence += 0.1
                if confidence < min_confidence:
                    stats["low_confidence"] += 1
                    continue
                stats["accepted"] += 1
                cardinality = infer_cardinality(parent_series, child_series)
                relationships.append(
                    RelationshipMeta(
                        from_table=parent_table,
                        from_column=parent_col,
                        to_table=child_table,
                        to_column=child_col,
                        relationship_type=cardinality,
                        confidence=confidence,
                    )
                )
    print(f"\n[STATS] {table1_name} ↔ {table2_name}")
    for k, v in stats.items():
        print(f"{k}: {v}")
    return relationships

def infer_all_relationships(
    dataset_meta,
    dataframes: Dict[str, pd.DataFrame],
    min_containment_threshold: float,
    min_confidence: float
) -> List[RelationshipMeta]:

    all_relationships = []
    table_names = list(dataset_meta.tables.keys())
    for i, t1 in enumerate(table_names):
        for t2 in table_names[i + 1:]:
            rels = detect_relationships_between_tables(
                t1, dataset_meta.tables[t1], dataframes[t1],
                t2, dataset_meta.tables[t2], dataframes[t2],
                min_containment_threshold,
                min_confidence
            )
            all_relationships.extend(rels)
    return all_relationships

def deduplicate_relationships(relationships: List[RelationshipMeta]) -> List[RelationshipMeta]:
    best = {}
    for rel in relationships:
        key = tuple(sorted([rel.from_table, rel.to_table])) + (
            rel.from_column if rel.from_table < rel.to_table else rel.to_column,
        )
        if key not in best or rel.confidence > best[key].confidence:
            best[key] = rel
    return list(best.values())

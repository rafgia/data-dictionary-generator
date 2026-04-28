import pathlib
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from pydantic import BaseModel, Field

class RelationshipMeta(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str 
    confidence: float 
    inference_method: str 
    
    class Config:
        extra = 'allow'

false_positives = {'age', 'value', 'label', 'unit', 'description', 'causes', 'full_name'} # can cause false positives in value overlap

def normalize_column_name(col_name: str) -> str:
    normalized = col_name.lower().strip().replace(" ", "_").replace("-", "_")
    for suffix in ['_id', 'id', '_key', 'key', '_code', 'code']:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[:-len(suffix)]
            break
    return normalized

def calculate_value_overlap(series1: pd.Series, series2: pd.Series, sample_size: int = 10000) -> float:
    s1_clean = series1.dropna()
    s2_clean = series2.dropna()
    if len(s1_clean) == 0 or len(s2_clean) == 0:
        return 0.0
    
    if len(s1_clean) > sample_size:
        s1_clean = s1_clean.sample(sample_size, random_state=42)
    
    set1 = set(s1_clean.astype(str).unique())
    set2 = set(s2_clean.astype(str).unique())
    
    if not set1:
        return 0.0
    
    overlap = len(set1.intersection(set2))
    return (overlap / len(set1)) * 100

def infer_cardinality(from_series: pd.Series, to_series: pd.Series) -> str:
    from_unique = from_series.nunique() / len(from_series) if len(from_series) > 0 else 0
    to_unique = to_series.nunique() / len(to_series) if len(to_series) > 0 else 0
    
    if from_unique > 0.98 and to_unique > 0.98:
        return "1:1"
    elif from_unique > 0.98:
        return "1:N" 
    elif to_unique > 0.98:
        return "N:1" 
    else:
        return "N:M"

def detect_relationships_between_tables(
    table1_name: str, table1_meta, table1_df: pd.DataFrame,
    table2_name: str, table2_meta, table2_df: pd.DataFrame,
    min_overlap_threshold: float = 80.0,
    min_confidence: float = 0.7
) -> List[RelationshipMeta]:
    relationships = []
    
    for col1_name, col1_meta in table1_meta.columns.items():
        if col1_name.lower() in false_positives: continue
            
        for col2_name, col2_meta in table2_meta.columns.items():
            if col2_name.lower() in false_positives: continue
            if col1_meta.data_type != col2_meta.data_type: continue
            if col1_meta.data_type in ["datetime", "float"]: continue
            
            norm_col1 = normalize_column_name(col1_name)
            norm_col2 = normalize_column_name(col2_name)
            
            name_match_score = 0.0
            inference_methods = []
            
            if norm_col1 == norm_col2:
                name_match_score = 0.5
                inference_methods.append("exact_name_match")
            
            overlap_percentage = calculate_value_overlap(table1_df[col1_name], table2_df[col2_name])
            if overlap_percentage < min_overlap_threshold:
                continue

            overlap_score = (overlap_percentage / 100.0) * 0.5
            inference_methods.append(f"overlap_{overlap_percentage:.1f}%")
            
            confidence = name_match_score + overlap_score
            
            if confidence >= min_confidence:
                cardinality = infer_cardinality(table1_df[col1_name], table2_df[col2_name])
                relationships.append(RelationshipMeta(
                    from_table=table1_name, from_column=col1_name,
                    to_table=table2_name, to_column=col2_name,
                    relationship_type=cardinality, confidence=confidence,
                    inference_method=" + ".join(inference_methods)
                ))
    return relationships

def infer_all_relationships(
    dataset_meta, 
    dataframes: Dict[str, pd.DataFrame], 
    min_overlap: float, 
    min_confidence: float
) -> List[RelationshipMeta]:
    all_relationships = []
    table_names = list(dataset_meta.tables.keys())
    
    for i, t1_name in enumerate(table_names):
        for t2_name in table_names[i+1:]:
            rels = detect_relationships_between_tables(
                t1_name, dataset_meta.tables[t1_name], dataframes[t1_name],
                t2_name, dataset_meta.tables[t2_name], dataframes[t2_name],
                min_overlap, min_confidence
            )
            all_relationships.extend(rels)
    return all_relationships

def deduplicate_relationships(relationships: List[RelationshipMeta]) -> List[RelationshipMeta]:
    relationship_map = {}
    for rel in relationships:
        table_pair = tuple(sorted([rel.from_table, rel.to_table]))
        key = (table_pair, rel.from_column if rel.from_table < rel.to_table else rel.to_column)
        
        if key not in relationship_map or rel.confidence > relationship_map[key].confidence:
            relationship_map[key] = rel
    return list(relationship_map.values())

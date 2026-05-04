import pathlib
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from pydantic import BaseModel, Field
import warnings
from data_dictionary_generator.pattern_extraction import extract_patterns, detect_semantic_type
from data_dictionary_generator.data_type import infer_column_data_type
from data_dictionary_generator.column_stats import compute_column_stats
from data_dictionary_generator.load_dataset import read_dataset


class ColumnMeta(BaseModel):
    column_name: str
    data_type: str
    semantic_type: Optional[str] = None
    sample_values: List[Any] = Field(default_factory=list)
    data_patterns: Dict[str, int] = Field(default_factory=dict)
    total_rows: int
    null_count: int
    unique_count: int
    stats: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

class TableMeta(BaseModel):
    table_name: str
    row_count: int
    number_of_columns: int
    list_of_columns: List[str]
    columns: Dict[str, ColumnMeta] = Field(default_factory=dict)
    time_period_covered: Optional[Dict[str, str]] = None
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)
    description: Optional[str] = None

    class Config:
        extra = 'allow'

class DatasetMeta(BaseModel):
    dataset_name: str
    list_of_tables: List[str]
    tables: Dict[str, TableMeta]
    time_period_covered: Optional[Dict[str, str]] = None
    domain_covered: Optional[str] = None
    description: Optional[str] = None

def extract_column_metadata(column_name: str, series: pd.Series) -> ColumnMeta:
    """
    Extracts metadata for a single column.
    """
    non_null_series = series.dropna()
    null_count = len(series) - len(non_null_series)
    data_type = infer_column_data_type(non_null_series)
    unique_count = series.nunique(dropna=False)
    sample_values = [
        item.item() if hasattr(item, 'item') else item
        for item in non_null_series.sample(min(50, len(non_null_series)), replace=False).tolist()
    ] if len(non_null_series) > 0 else []

    data_patterns = extract_patterns(series).to_dict()
    semantic_type = detect_semantic_type(series, column_name)
    stats = compute_column_stats(series)

    return ColumnMeta(
        column_name=column_name,
        data_type = data_type,
        semantic_type=semantic_type,
        data_patterns=data_patterns,
        sample_values=sample_values,
        total_rows=len(series),
        null_count=null_count,
        unique_count=unique_count,
        stats=stats,
    )

def extract_table_metadata(table_file: pathlib.Path, num_sample_rows: int = 3) -> Tuple[Optional[TableMeta], Optional[pd.DataFrame]]:
    """
    Extract metadata for a single table including columns and sample rows.
    Returns the TableMeta and the DataFrame
    """
    table_name = table_file.stem
    df = read_dataset(table_file)
    earliest_dt: Optional[datetime.datetime] = None
    latest_dt: Optional[datetime.datetime] = None
    columns_meta: Dict[str, ColumnMeta] = {}

    for col in df.columns:
        series = df[col]
        col_meta = extract_column_metadata(col, series)
        columns_meta[col] = col_meta

        if col_meta.data_type == "datetime":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='Could not infer format, so each element will be parsed individually, falling back to `dateutil`'
                )
                dt = pd.to_datetime(series, errors="coerce").dropna()

            if not dt.empty:
                current_min = dt.min().to_pydatetime()
                current_max = dt.max().to_pydatetime()

                if earliest_dt is None or current_min < earliest_dt:
                    earliest_dt = current_min
                if latest_dt is None or current_max > latest_dt:
                    latest_dt = current_max

    time_period = {
        "start": earliest_dt.isoformat(),
        "end": latest_dt.isoformat()
    } if earliest_dt and latest_dt else None

    actual_sample_rows: List[Dict[str, Any]] = []
    if not df.empty:
        sampled_df = df.sample(min(num_sample_rows, len(df)), random_state=42).reset_index(drop=True)
        for _, row in sampled_df.iterrows():
            cleaned_row = {col: (item.item() if hasattr(item, 'item') else item) for col, item in row.items()}
            actual_sample_rows.append(cleaned_row)
  
    table_meta = TableMeta(
        table_name=table_name,
        row_count=len(df),
        number_of_columns=len(df.columns),
        list_of_columns=df.columns.tolist(),
        columns=columns_meta,
        time_period_covered=time_period,
        sample_rows=actual_sample_rows
    )

    return table_meta, df

def extract_dataset_metadata(path: pathlib.Path, domain_covered: Optional[str] = None, output_dir: Optional[pathlib.Path] = None) -> DatasetMeta:
    """
    Extract metadata for all tables in a dataset.
    If the domain is not provided, the user will be prompted for input.
    """
    tables_meta: Dict[str, TableMeta] = {}
    table_names: List[str] = []
    dataframes: Dict[str, pd.DataFrame] = {}
    all_starts: List[datetime.datetime] = []
    all_ends: List[datetime.datetime] = []

    if path.is_dir():
        table_files = [f for f in path.iterdir() if f.is_file()]
    else:
        table_files = [path]

    for table_file in table_files:
        try:
            table_meta, df = extract_table_metadata(table_file)
        except Exception as e:
            print(f"Warning: Skipping file {table_file.name} due to error: {e}")
            continue

        if table_meta is None:
            continue

        tables_meta[table_meta.table_name] = table_meta
        dataframes[table_meta.table_name] = df
        table_names.append(table_meta.table_name)

        if table_meta.time_period_covered:
            all_starts.append(datetime.datetime.fromisoformat(table_meta.time_period_covered['start']))
            all_ends.append(datetime.datetime.fromisoformat(table_meta.time_period_covered['end']))

    if all_starts and all_ends:
        min_start = min(all_starts)
        max_end = max(all_ends)
        dataset_time_period = {"start": min_start.isoformat(), "end": max_end.isoformat()}
    else:
        dataset_time_period = None

    if domain_covered is None:
        try:
            user_input = input("Please enter the domain covered by this dataset (e.g., Finance, Healthcare, or leave blank): ")
            if user_input.strip():
                domain_covered = user_input.strip()
            else:
                domain_covered = "Unspecified"
        except EOFError:
            domain_covered = "Unspecified"

    if domain_covered is None:
        domain_covered = "Unspecified"

    dataset_meta = DatasetMeta(
        dataset_name=path.stem,
        list_of_tables=table_names,
        tables=tables_meta,
        time_period_covered=dataset_time_period,
        domain_covered=domain_covered
    )

    for table_name, df in dataframes.items():
        dataset_meta.tables[table_name].dataframe = df

    #for table_meta in dataset_meta.tables.values():
        #if hasattr(table_meta, 'dataframe'):
            #del table_meta.dataframe

    return dataset_meta
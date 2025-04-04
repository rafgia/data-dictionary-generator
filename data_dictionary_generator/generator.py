import os
import pandas as pd
import subprocess
import logging
import re
from typing import Optional, List, Dict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ollama_model(prompt: str, model: str = "llama3.1") -> Optional[str]:
    """
    Uses the Ollama CLI to generate a response with strict formatting requirements.
    """
    try:
        formatted_prompt = (
            f"{prompt}\n"
            "Provide only a very short (10-15 words max), precise description. "
            "Use clinical terminology where appropriate. "
            "Do not include any explanations or notes."
        )

        command = ["ollama", "run", model, formatted_prompt]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Error running Ollama model: {result.stderr}")
            return None

        response = result.stdout.strip()
        clean_response = re.sub(
            r"<think>.*?</think>|\n|\"|'|`", "", response, flags=re.DOTALL
        ).strip()

        if len(clean_response.split()) > 20:
            clean_response = " ".join(clean_response.split()[:15]) + "..."

        return clean_response
    except Exception as e:
        logger.error(f"Failed to generate description: {e}")
        return None


def infer_data_type(series: pd.Series) -> str:
    """
    Infer data type using pandas functionality first, then augment with LLM if needed.
    """
    dtype = str(series.dtype)

    dtype_mapping = {
        "object": "string",
        "int64": "integer",
        "float64": "float",
        "bool": "boolean",
        "datetime64": "datetime",
    }

    return dtype_mapping.get(dtype.split("[")[0], dtype)


def generate_table_description(
    df: pd.DataFrame, table_name: str, dataset_name: str, model: str
) -> str:
    """
    Generate a concise table description focusing on clinical relevance.
    """
    columns_list = ", ".join(df.columns[:5])
    if len(df.columns) > 5:
        columns_list += f" and {len(df.columns) - 5} more"

    prompt = (
        f"Describe the clinical purpose of the '{table_name}' table in the '{dataset_name}' dataset "
        f"containing columns like {columns_list}. Be extremely concise (max 15 words)."
    )

    description = run_ollama_model(prompt, model)
    return description or f"Clinical data table containing {columns_list}"


def generate_column_description(
    column: str, sample_values: List, table_name: str, dataset_name: str, model: str
) -> str:
    """
    Generate a precise column description using sample values for context.
    """
    samples_str = ", ".join(str(v) for v in sample_values[:3])
    if len(sample_values) > 3:
        samples_str += ", ..."

    prompt = (
        f"Briefly describe the '{column}' field in the '{table_name}' table of the '{dataset_name}' dataset. "
        f"Example values: {samples_str}. Use clinical terms. Max 15 words."
    )

    description = run_ollama_model(prompt, model)
    return description or f"Clinical measurement: {column}"


def generate_metadata_for_table(
    df: pd.DataFrame, table_name: str, dataset_name: str, model: str
) -> pd.DataFrame:
    """
    Generates concise metadata for the columns in a clinical data table.
    """
    metadata = []

    table_description = generate_table_description(df, table_name, dataset_name, model)

    for column in df.columns:
        sample_data = df[column].dropna().loc[df[column] != ""].head(5).tolist()
        column_description = generate_column_description(
            column, sample_data, table_name, dataset_name, model
        )

        datatype = infer_data_type(df[column])

        metadata.append(
            {
                "table_name": table_name,
                "dataset_name": dataset_name,
                "table_description": table_description,
                "column_name": column,
                "column_description": column_description,
                "sample_data": sample_data[:3],
                "datatype": datatype,
                "number_of_rows": len(df),
                "number_of_columns": len(df.columns),
            }
        )

    return pd.DataFrame(metadata)


def generate_data_quality_report(
    df: pd.DataFrame, table_name: str
) -> List[Dict[str, str]]:
    data_quality_rows = []

    duplicate_rows = df.duplicated().any()
    num_duplicate_rows = df.duplicated().sum()
    duplicate_columns = df.T.duplicated().any()
    num_duplicate_columns = df.T.duplicated().sum()

    for col in df.columns:
        col_data = df[col]

        if col_data.dtype in [np.float64, np.int64, "float64", "int64"]:
            col_series = col_data.dropna()
            if not col_series.empty:
                q1 = col_series.quantile(0.25)
                q3 = col_series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = (col_series < lower_bound) | (col_series > upper_bound)
                num_outliers = outliers.sum()
                has_outliers = outliers.any()
            else:
                num_outliers = 0
                has_outliers = False
        else:
            num_outliers = 0
            has_outliers = False

        missing_percentage = df[col].isna().mean() * 100

        data_quality_rows.append(
            {
                "table_name": str(table_name),
                "column_name": str(col),
                "missing_percentage": f"{missing_percentage:.2f}",
                "has_outliers": str(has_outliers),
                "num_outliers": str(num_outliers),
            }
        )

    data_quality_rows.append(
        {
            "table_name": str(table_name),
            "column_name": "ALL_COLUMNS",
            "missing_percentage": "N/A",
            "has_outliers": "N/A",
            "num_outliers": "N/A",
            "has_duplicate_rows": str(duplicate_rows),
            "num_duplicate_rows": str(num_duplicate_rows),
            "has_duplicate_columns": str(duplicate_columns),
            "num_duplicate_columns": str(num_duplicate_columns),
        }
    )

    return data_quality_rows


def generate_relationships_between_tables(
    tables_metadata: List[Dict], model: str
) -> pd.DataFrame:
    """
    Uses LLM model to detect relationships between tables.
    For simplicity, we'll assume relationships are identified through shared column names or content similarity.
    """
    relationships = []

    if not all(isinstance(metadata, dict) for metadata in tables_metadata):
        logger.error("Expected a list of dictionaries, but found an incorrect format.")
        return pd.DataFrame()

    try:
        table_names = list(
            set([metadata["table_name"] for metadata in tables_metadata])
        )
    except KeyError as e:
        logger.error(f"Missing expected key in metadata: {e}")
        return pd.DataFrame()

    for i, table1 in enumerate(table_names):
        for j, table2 in enumerate(table_names):
            if i >= j:
                continue

            table1_columns = {
                metadata["column_name"]
                for metadata in tables_metadata
                if metadata["table_name"] == table1
            }
            table2_columns = {
                metadata["column_name"]
                for metadata in tables_metadata
                if metadata["table_name"] == table2
            }

            common_columns = table1_columns.intersection(table2_columns)

            if common_columns:
                prompt = (
                    f"Are the tables '{table1}' and '{table2}' related based on columns: {', '.join(common_columns)}? "
                    "If so, describe the relationship in brief clinical terms (max 15 words)."
                )
                relationship_description = run_ollama_model(prompt, model)

                if relationship_description:
                    relationships.append(
                        {
                            "table1": table1,
                            "table2": table2,
                            "common_columns": ", ".join(common_columns),
                            "relationship": relationship_description,
                        }
                    )

    return pd.DataFrame(relationships)


def process_csv(
    file_path: str,
    dataset_name: str,
    model: str = "llama3.1",
    output_path: Optional[str] = ".",
    all_tables_metadata: Optional[List[Dict]] = None,  # Change here
) -> Optional[tuple]:
    """
    Process a CSV file to generate concise metadata for the clinical data table, generate data quality report,
    and detect relationships between tables.
    """
    if all_tables_metadata is None:  # Initialize the list if None is passed
        all_tables_metadata = []
    try:
        df = pd.read_csv(file_path)
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        metadata = generate_metadata_for_table(df, table_name, dataset_name, model)

        quality_report = generate_data_quality_report(df, table_name)

        all_tables_metadata.extend(metadata)

        return metadata, quality_report, all_tables_metadata
    except Exception as e:
        logger.error(f"Failed to process CSV {file_path}: {e}")
        return None

import os
import pandas as pd
import subprocess
import logging
import re
from typing import Optional, List

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


def process_csv(
    file_path: str, dataset_name: str, model: str = "llama3.1"
) -> Optional[pd.DataFrame]:
    """
    Process a CSV file to generate concise metadata for the clinical data table.
    """
    try:
        df = pd.read_csv(file_path)
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        metadata = generate_metadata_for_table(df, table_name, dataset_name, model)
        return metadata
    except Exception as e:
        logger.error(f"Failed to process CSV {file_path}: {e}")
        return None

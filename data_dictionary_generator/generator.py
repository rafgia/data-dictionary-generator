import os
import pandas as pd
import subprocess
import logging
import re
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ollama_model(prompt: str, model: str = "deepseek-r1:1.5b") -> Optional[str]:
    """
    This function uses the Ollama CLI to generate a response based on a prompt.
    The prompt requests only the answer without explanation.
    """
    try:
        command = [
            "ollama",
            "run",
            model,
            prompt,
            f"{prompt}\nPlease provide only the answer, without explanation.",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Error running Ollama model: {result.stderr}")
            return None
        response = result.stdout.strip()
        clean_response = re.sub(
            r"<think>.*?</think>", "", response, flags=re.DOTALL
        ).strip()
        return clean_response
    except Exception as e:
        logger.error(f"Failed to generate description: {e}")
        return None


def generate_metadata_for_table(
    df: pd.DataFrame, table_name: str, dataset_name: str, model: str = "deepseek-r1:7b"
) -> pd.DataFrame:
    """
    Generates metadata for the columns in a table by calling an Ollama model
    for descriptions and datatype inference.
    """
    metadata = []
    table_description = run_ollama_model(
        f"Give a short and accurate description of what the {table_name} table "
        f"from {dataset_name} dataset contains.",
        model,
    )
    for column in df.columns:
        column_description = run_ollama_model(
            f"Give a short and accurate description of {column} from {table_name} "
            f"from {dataset_name} dataset.",
            model,
        )
        sample_data = df[column].dropna().loc[df[column] != ""].head(5).tolist()
        datatype_inference = run_ollama_model(
            f"Print the datatype of {', '.join(map(str, sample_data))}? "
            f"Give me only the data type, for example if the sample data is "
            f"[2,3,4,5,6] give me 'int64' ",
            model,
        )
        metadata.append(
            {
                "table_name": table_name,
                "dataset_name": dataset_name,
                "table_description": table_description,
                "column_name": column,
                "column_description": column_description,
                "sample_data": sample_data,
                "datatype": datatype_inference,
                "number_of_rows": len(df),
                "number_of_columns": len(df.columns),
            }
        )
    return pd.DataFrame(metadata)


def process_csv(
    file_path: str, dataset_name: str, model: str = "deepseek-r1:1.5b"
) -> Optional[pd.DataFrame]:
    """
    Process a CSV file to generate metadata for the table.
    """
    try:
        df = pd.read_csv(file_path)
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        metadata = generate_metadata_for_table(df, table_name, dataset_name, model)
        return metadata
    except Exception as e:
        logger.error(f"Failed to process CSV {file_path}: {e}")
        return None

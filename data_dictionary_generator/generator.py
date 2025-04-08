import os
import pandas as pd
import subprocess
import logging
import re
from typing import Optional, List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import normalized_mutual_info_score


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")


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
    tables_metadata: List[Dict], data_path: str
) -> pd.DataFrame:
    """
    Generate relationships between tables with semantic and statistical analysis.
    """
    DEFAULT_CONFIDENCE = 0.0

    relationship_columns = [
        "table_a",
        "column_a",
        "table_b",
        "column_b",
        "semantic_similarity",
        "mutual_info",
        "cardinality",
        "confidence",
    ]
    relationships_df = pd.DataFrame(columns=relationship_columns)

    if not tables_metadata or len(tables_metadata) < 2:
        return relationships_df

    try:
        df_meta = pd.DataFrame(tables_metadata).fillna("")
        if df_meta.empty:
            return relationships_df

        tables = {}
        for table_name, group in df_meta.groupby("table_name"):
            tables[table_name] = {
                "columns": group["column_name"].tolist(),
                "file_path": os.path.join(data_path, f"{table_name}.csv"),
            }

        relationships = []
        table_names = list(tables.keys())

        for i, table_a in enumerate(table_names[:-1]):
            for table_b in table_names[i + 1 :]:
                try:
                    df_a = pd.read_csv(tables[table_a]["file_path"])
                    df_b = pd.read_csv(tables[table_b]["file_path"])

                    for col_a in tables[table_a]["columns"]:
                        for col_b in tables[table_b]["columns"]:
                            try:
                                name_score = semantic_similarity(col_a, col_b)

                                if name_score > 0.7:
                                    vals_a = df_a[col_a].dropna().astype(str).tolist()
                                    vals_b = df_b[col_b].dropna().astype(str).tolist()

                                    mi_score = compute_mutual_information(
                                        vals_a, vals_b
                                    )
                                    cardinality = estimate_cardinality(vals_a, vals_b)

                                    relationships.append(
                                        {
                                            "table_a": table_a,
                                            "column_a": col_a,
                                            "table_b": table_b,
                                            "column_b": col_b,
                                            "semantic_similarity": round(name_score, 3),
                                            "mutual_info": round(mi_score, 3),
                                            "cardinality": cardinality,
                                            "confidence": round(
                                                0.4 * name_score + 0.6 * mi_score, 3
                                            ),
                                        }
                                    )
                            except Exception as col_error:
                                logger.warning(
                                    f"Skipping {table_a}.{col_a} ↔ {table_b}.{col_b}: {str(col_error)}"
                                )
                                continue

                except Exception as table_error:
                    logger.warning(
                        f"Skipping {table_a} ↔ {table_b} comparison: {str(table_error)}"
                    )
                    continue

        if relationships:
            relationships_df = pd.DataFrame(relationships)
            for col in relationship_columns:
                if col not in relationships_df.columns:
                    relationships_df[col] = (
                        DEFAULT_CONFIDENCE if col == "confidence" else ""
                    )

            relationships_df = relationships_df.sort_values(
                "confidence", ascending=False
            )

    except Exception as e:
        logger.error(f"Relationship generation failed: {str(e)}")

    return relationships_df


def semantic_similarity(col1: str, col2: str) -> float:
    embedding1 = model.encode(col1, convert_to_tensor=True)
    embedding2 = model.encode(col2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding1, embedding2).item()
    return score


def estimate_cardinality(col_a_vals: List[str], col_b_vals: List[str]) -> str:
    unique_a = len(set(col_a_vals))
    unique_b = len(set(col_b_vals))
    if unique_a == 0 or unique_b == 0:
        return "Unknown"
    ratio = unique_a / unique_b
    if 0.9 < ratio < 1.1:
        return "One-to-One"
    elif ratio < 0.9:
        return "Many-to-One"
    else:
        return "One-to-Many"


def compute_mutual_information(col_a_vals: List[str], col_b_vals: List[str]) -> float:
    try:
        le_a = LabelEncoder()
        le_b = LabelEncoder()
        a_encoded = le_a.fit_transform(col_a_vals)
        b_encoded = le_b.fit_transform(col_b_vals)
        return normalized_mutual_info_score(a_encoded, b_encoded)
    except ValueError as e:
        logger.warning(f"ValueError in mutual info calculation: {e}")
        return 0.0


def process_csv(
    file_path: str,
    dataset_name: str,
    model: str = "llama3.1",
    output_path: Optional[str] = ".",
    all_tables_metadata: Optional[List[Dict]] = None,
) -> Optional[tuple]:
    """
    Process a CSV file to generate concise metadata for the clinical data table, generate data quality report,
    and detect relationships between tables.
    """
    try:
        df = pd.read_csv(file_path)
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        metadata_df = generate_metadata_for_table(df, table_name, dataset_name, model)

        metadata = metadata_df.to_dict("records")

        if all_tables_metadata is not None:
            all_tables_metadata.extend(metadata)

        quality_report = generate_data_quality_report(df, table_name)
        return metadata_df, quality_report, all_tables_metadata
    except Exception as e:
        logger.error(f"Failed to process CSV {file_path}: {e}")
        return None


def save_relationships_to_markdown(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = ["# Detected Table Relationships\n"]
    for _, rel in df.sort_values(by="confidence", ascending=False).iterrows():
        lines.append(
            f"## {rel['table_a']}.{rel['column_a']} ↔ {rel['table_b']}.{rel['column_b']}"
        )
        lines.append(f"- **Semantic Similarity**: {rel['semantic_similarity']}")
        lines.append(f"- **Mutual Information**: {rel['mutual_info']}")
        lines.append(f"- **Cardinality**: {rel['cardinality']}")
        lines.append(f"- **Confidence Score**: {rel['confidence']}\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

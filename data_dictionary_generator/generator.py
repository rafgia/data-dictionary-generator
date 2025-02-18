import os
import pandas as pd
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ollama_model(prompt, model="deepseek-r1:1.5b"):
    """
    This function uses the Ollama CLI to generate a response based on a prompt.
    """
    try:
        command = ["ollama", "run", model, "--prompt", prompt]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Error running Ollama model: {result.stderr}")
            return None
        return result.stdout.strip()
    
    except Exception as e:
        logger.error(f"Failed to generate description: {e}")
        return None

def generate_metadata_for_table(df, table_name, dataset_name, model="deepseek-r1:1.5b"):
    metadata = []
    
    table_description = run_ollama_model(f"Give a description of what the {table_name} table from {dataset_name} dataset contains.", model)

    for column in df.columns:
        column_description = run_ollama_model(f"Give a description of {column} from {table_name} from {dataset_name} dataset.", model)
        
        sample_data = df[column].sample(5).tolist()
        
        datatype_inference = run_ollama_model(f"What kind of data is {', '.join(map(str, sample_data))}?", model)

        metadata.append({
            "table_name": table_name,
            "dataset_name": dataset_name,
            "table_description": table_description,
            "column_name": column,
            "column_description": column_description,
            "sample_data": sample_data,
            "datatype_inference": datatype_inference,
            "number_of_rows": len(df),
            "number_of_columns": len(df.columns),
        })
    
    return pd.DataFrame(metadata)

def process_csv(file_path, dataset_name, model="deepseek-r1:1.5b"):
    try:
        df = pd.read_csv(file_path)
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        metadata = generate_metadata_for_table(df, table_name, dataset_name, model)
        return metadata
    except Exception as e:
        logger.error(f"Failed to process CSV {file_path}: {e}")
        return None
    
import os
import pandas as pd
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ollama_model(column_name, table_name, dataset_name, model="deepseek-r1:1.5b"):
    """
    This function uses the Ollama CLI to generate a description for the given column using a model.
    """
    try:
        input_text = f"Give a description of {column_name} from {table_name} from {dataset_name} dataset."
        command = ["ollama", "run", model, "--prompt", input_text]

        # Run the command via subprocess
        result = subprocess.run(command, capture_output=True, text=True)

        # Capture the result
        if result.returncode != 0:
            logger.error(f"Error running Ollama model: {result.stderr}")
            return None
        return result.stdout.strip()
    
    except Exception as e:
        logger.error(f"Failed to generate description for {column_name}: {e}")
        return None

def generate_metadata_for_table(df, table_name, dataset_name, model="deepseek-r1:1.5b"):
    metadata = []
    for column in df.columns:
        description = run_ollama_model(column, table_name, dataset_name, model)
        sample_data = df[column].sample(5)
        data_type = str(df[column].dtype)
        
        metadata.append({
            "table_name": table_name,
            "dataset_name": dataset_name,
            "column_name": column,
            "sample_data": list(sample_data),
            "data_type": data_type,
            "column_description": description
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

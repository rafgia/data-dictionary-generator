import click
import os
from data_dictionary_generator.generator import process_csv

@click.command()
@click.argument('folder_path')
@click.argument('dataset_name')
@click.argument('output_file')
@click.option('--model', default="deepseek-r1:1.5b", help="Ollama model to use for metadata generation.")
def generate_dictionary(folder_path, dataset_name, output_file, model):
    """
    Generates a data dictionary for all CSV files in the FOLDER_PATH and saves it to OUTPUT_FILE.
    """
    import pandas as pd
    from pathlib import Path

    all_metadata = pd.DataFrame()

    # Iterate over all CSV files in the folder
    for file in Path(folder_path).glob("*.csv"):
        metadata = process_csv(str(file), dataset_name, model)
        if metadata is not None:
            all_metadata = pd.concat([all_metadata, metadata], ignore_index=True)

    # Save to CSV
    all_metadata.to_csv(output_file, index=False)
    click.echo(f"Metadata generation complete. Saved to {output_file}")

if __name__ == '__main__':
    generate_dictionary()

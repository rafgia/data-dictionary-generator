import click
from data_dictionary_generator.generator import process_csv
import pandas as pd
from pathlib import Path


@click.command()
@click.argument("folder_path")
@click.argument("dataset_name")
@click.argument("output_file")
@click.option(
    "--model",
    default="llama3.1",
    help="Ollama model to use for metadata generation.",
)
def generate_dictionary(
    folder_path: str, dataset_name: str, output_file: str, model: str
) -> None:
    """
    Generates a data dictionary for all CSV files in the FOLDER_PATH
    and saves it to OUTPUT_FILE.
    """
    all_metadata = pd.DataFrame()
    for file in Path(folder_path).glob("*.csv"):
        metadata = process_csv(str(file), dataset_name, model)
        if metadata is not None:
            all_metadata = pd.concat([all_metadata, metadata], ignore_index=True)
    all_metadata.to_csv(output_file, index=False)

    message = f"Metadata generation complete. Saved to {output_file}"
    click.echo(message)


if __name__ == "__main__":
    generate_dictionary()

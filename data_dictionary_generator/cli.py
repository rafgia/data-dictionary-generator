import click
from data_dictionary_generator.generator import process_csv
import pandas as pd
from pathlib import Path

from fpdf import FPDF  # for PDF output


def save_as_pdf(df: pd.DataFrame, output_file: str) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    col_widths = [30] * len(df.columns)

    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 10, col, border=1)
    pdf.ln()

    for _, row in df.iterrows():
        for i, col in enumerate(df.columns):
            value = str(row[col])[:25]  # Truncate long fields
            pdf.cell(col_widths[i], 10, value, border=1)
        pdf.ln()

    pdf.output(output_file)


def save_as_markdown(df: pd.DataFrame, output_file: str) -> None:
    with open(output_file, "w") as f:
        f.write(df.to_markdown(index=False))


@click.command()
@click.argument("folder_path")
@click.argument("dataset_name")
@click.argument("output_file")
@click.option(
    "--model",
    default="llama3.1",
    help="Ollama model to use for metadata generation.",
)
@click.option(
    "--format",
    default="csv",
    type=click.Choice(["csv", "json", "markdown", "pdf"]),
    help="Output format for the data dictionary.",
)
def generate_dictionary(
    folder_path: str, dataset_name: str, output_file: str, model: str, format: str
) -> None:
    """
    Generates a data dictionary for all CSV files in the FOLDER_PATH
    and saves it to OUTPUT_FILE in the specified FORMAT.
    """
    all_metadata = pd.DataFrame()
    for file in Path(folder_path).glob("*.csv"):
        metadata = process_csv(str(file), dataset_name, model)
        if metadata is not None:
            all_metadata = pd.concat([all_metadata, metadata], ignore_index=True)

    if format == "csv":
        all_metadata.to_csv(output_file, index=False)
    elif format == "json":
        all_metadata.to_json(output_file, orient="records", indent=2)
    elif format == "markdown":
        save_as_markdown(all_metadata, output_file)
    elif format == "pdf":
        save_as_pdf(all_metadata, output_file)

    click.echo(
        f"Metadata generation complete. Saved as {format.upper()} to {output_file}"
    )


if __name__ == "__main__":
    generate_dictionary()

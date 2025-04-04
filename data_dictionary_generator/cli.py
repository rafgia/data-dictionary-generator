import click
from data_dictionary_generator.generator import (
    generate_relationships_between_tables,
    process_csv,
)
import pandas as pd
from pathlib import Path
import logging
from fpdf import FPDF
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            value = str(row[col])[:25]
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
    all_metadata: pd.DataFrame = pd.DataFrame()
    all_quality_reports: pd.DataFrame = pd.DataFrame()
    all_tables_metadata: List[Dict] = []

    for file in Path(folder_path).glob("*.csv"):
        result = process_csv(
            str(file), dataset_name, model, all_tables_metadata=all_tables_metadata
        )

        if result is None:
            logger.warning(f"Skipping {file.name} — process_csv returned None.")
            continue

        metadata, quality_report, all_tables_metadata = result

        if metadata is not None:
            metadata_df = (
                pd.DataFrame(metadata) if isinstance(metadata, list) else metadata
            )
            all_metadata = pd.concat([all_metadata, metadata_df], ignore_index=True)

        if quality_report is not None:
            quality_report_df = (
                pd.DataFrame(quality_report)
                if isinstance(quality_report, list)
                else quality_report
            )
            all_quality_reports = pd.concat(
                [all_quality_reports, quality_report_df], ignore_index=True
            )

    relationships_df = generate_relationships_between_tables(all_tables_metadata, model)

    all_metadata.to_csv(output_file, index=False)
    logger.info(f"Metadata dictionary saved to {output_file}")

    quality_output_path = Path(output_file).with_name("data_quality.csv")
    all_quality_reports.to_csv(quality_output_path, index=False)
    logger.info(f"Data quality report saved to {quality_output_path}")

    relations_output_path = Path(output_file).with_name("relations.csv")
    relationships_df.to_csv(relations_output_path, index=False)
    logger.info(f"Relationships report saved to {relations_output_path}")

    print("Metadata generation complete")


if __name__ == "__main__":
    generate_dictionary()

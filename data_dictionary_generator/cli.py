import os
import subprocess
import pandas as pd
import click
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from data_dictionary_generator.generator import (
    process_csv,
    generate_relationships_between_tables,
    save_relationships_to_markdown,
)
from importlib.metadata import version

VERSION = version("data-dictionary-generator")


def save_metadata_to_pdf(metadata_df: pd.DataFrame, output_file: str) -> None:
    """Save metadata as a formatted PDF document with text layout"""
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["Heading1"],
            fontSize=14,
            leading=16,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableDescription",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            spaceAfter=18,
            leftIndent=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ColumnHeader",
            parent=styles["Heading2"],
            fontSize=12,
            leading=14,
            spaceAfter=6,
            leftIndent=36,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ColumnDetail",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            spaceAfter=6,
            leftIndent=54,
        )
    )

    elements = []

    elements.append(Paragraph("Data dictionary", styles["Title"]))
    elements.append(Spacer(1, 0.25 * inch))

    grouped = metadata_df.groupby("table_name")

    for table_name, group in grouped:
        table_desc = group.iloc[0]["table_description"]
        dataset_name = group.iloc[0]["dataset_name"]

        elements.append(Paragraph(f"Table: {table_name}", styles["TableHeader"]))
        elements.append(
            Paragraph(f"Dataset: {dataset_name}", styles["TableDescription"])
        )
        elements.append(
            Paragraph(f"Description: {table_desc}", styles["TableDescription"])
        )
        elements.append(Spacer(1, 0.1 * inch))

        for _, row in group.iterrows():
            sample_data = ", ".join(str(x) for x in row["sample_data"][:3])

            elements.append(
                Paragraph(f"Column: {row['column_name']}", styles["ColumnHeader"])
            )
            elements.append(
                Paragraph(
                    f"Description: {row['column_description']}", styles["ColumnDetail"]
                )
            )
            elements.append(
                Paragraph(f"Data Type: {row['datatype']}", styles["ColumnDetail"])
            )
            elements.append(
                Paragraph(f"Sample Values: {sample_data}", styles["ColumnDetail"])
            )
            elements.append(Spacer(1, 0.1 * inch))

        elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)


def save_quality_to_pdf(quality_df: pd.DataFrame, output_file: str) -> None:
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["Heading1"],
            fontSize=14,
            leading=16,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="QualityHeader",
            parent=styles["Heading2"],
            fontSize=12,
            leading=14,
            spaceAfter=6,
            leftIndent=36,
        )
    )
    styles.add(
        ParagraphStyle(
            name="QualityDetail",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            spaceAfter=6,
            leftIndent=54,
        )
    )

    elements = []

    elements.append(Paragraph("Data quality report", styles["Title"]))
    elements.append(Spacer(1, 0.25 * inch))

    grouped = quality_df.groupby("table_name")

    for table_name, group in grouped:
        elements.append(Paragraph(f"Table: {table_name}", styles["TableHeader"]))
        elements.append(Spacer(1, 0.1 * inch))

        general_info = group[group["column_name"] == "ALL_COLUMNS"]
        if not general_info.empty:
            info = general_info.iloc[0]
            elements.append(
                Paragraph("General Table Quality:", styles["QualityHeader"])
            )
            if info.get("has_duplicate_rows") == "True":
                elements.append(
                    Paragraph(
                        f"Duplicate Rows: {info.get('num_duplicate_rows', 'N/A')}",
                        styles["QualityDetail"],
                    )
                )
            if info.get("has_duplicate_columns") == "True":
                elements.append(
                    Paragraph(
                        f"Duplicate Columns: {info.get('num_duplicate_columns', 'N/A')}",
                        styles["QualityDetail"],
                    )
                )
            elements.append(Spacer(1, 0.1 * inch))

        columns_info = group[group["column_name"] != "ALL_COLUMNS"]
        for _, row in columns_info.iterrows():
            elements.append(
                Paragraph(f"Column: {row['column_name']}", styles["QualityHeader"])
            )
            elements.append(
                Paragraph(
                    f"Missing Values: {row['missing_percentage']}%",
                    styles["QualityDetail"],
                )
            )
            if row.get("has_outliers") == "True":
                elements.append(
                    Paragraph(
                        f"Outliers: {row.get('num_outliers', 'N/A')}",
                        styles["QualityDetail"],
                    )
                )
            elements.append(Spacer(1, 0.1 * inch))

        elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)


def save_metadata_in_format(
    metadata_df: pd.DataFrame,
    format: str,
    output_file: str,
    is_relationships: bool = False,
) -> None:
    """Save metadata in specified format"""
    if format == "csv":
        metadata_df.to_csv(output_file, index=False)
    elif format == "json":
        metadata_df.to_json(output_file, orient="records", lines=True)
    elif format == "pdf":
        if is_relationships:
            save_relationships_to_pdf(metadata_df, output_file)
        else:
            save_metadata_to_pdf(metadata_df, output_file)
    elif format == "markdown":
        with open(output_file, "w") as f:
            if is_relationships:
                f.write("# Table relationships\n")
                for _, row in metadata_df.iterrows():
                    f.write(
                        f"## {row['table_a']}.{row['column_a']} ↔ {row['table_b']}.{row['column_b']}\n"
                    )
                    f.write(
                        f"- **Semantic similarity**: {row['semantic_similarity']:.2f}\n"
                    )
                    f.write(f"- **Cardinality**: {row['cardinality']}\n")
                    f.write(f"- **Confidence**: {row['confidence']:.2f}\n\n")
            else:
                f.write("# Metadata\n")
                for _, row in metadata_df.iterrows():
                    f.write(f"## {row['table_name']} - {row['column_name']}\n")
                    f.write(f"- **Description**: {row['column_description']}\n")
                    f.write(f"- **Data type**: {row['datatype']}\n")
                    f.write(f"- **Sample data**: {row['sample_data']}\n")
                    f.write("\n")
    else:
        click.echo(f"Unsupported format: {format}")


def save_relationships_to_pdf(relationships_df: pd.DataFrame, output_file: str) -> None:
    """Save relationships as a formatted PDF document"""
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="Relationship",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            spaceAfter=6,
        )
    )

    elements = []

    elements.append(Paragraph("Table relationships", styles["Title"]))
    elements.append(Spacer(1, 0.25 * inch))

    if relationships_df.empty:
        elements.append(
            Paragraph("No relationships detected between tables.", styles["Normal"])
        )
    else:
        table_data = []
        headers = [
            "Table A",
            "Column A",
            "Table B",
            "Column B",
            "Similarity",
            "Cardinality",
            "Confidence",
        ]
        table_data.append(headers)

        for _, row in relationships_df.iterrows():
            table_data.append(
                [
                    row["table_a"],
                    row["column_a"],
                    row["table_b"],
                    row["column_b"],
                    f"{row['semantic_similarity']:.2f}",
                    row["cardinality"],
                    f"{row['confidence']:.2f}",
                ]
            )

        col_widths = [
            1.2 * inch,
            1.5 * inch,
            1.2 * inch,
            1.5 * inch,
            0.8 * inch,
            1 * inch,
            0.8 * inch,
        ]
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)


def validate_ollama_model_available(model: str) -> None:
    """Fail fast if the requested Ollama model is not available locally."""
    try:
        result = subprocess.run(
            ["ollama", "show", model], capture_output=True, text=True
        )
    except FileNotFoundError as e:
        raise click.ClickException(
            "Ollama CLI not found. Please install Ollama and ensure 'ollama' is on your PATH."
        ) from e

    if result.returncode != 0:
        raise click.ClickException(
            f"Ollama model '{model}' not found. Install it with: 'ollama pull {model}'"
        )


@click.command()
@click.argument("data-dir", type=click.Path(exists=True, file_okay=False))
@click.option("--dataset-name", required=True, help="Name of the dataset")
@click.option("--model", default="llama3.1", help="Ollama model to use")
@click.option(
    "--output-dir",
    default="output",
    type=click.Path(file_okay=False),
    help="Output directory for reports",
)
@click.option(
    "--format",
    default="csv",
    type=click.Choice(["csv", "json", "pdf", "markdown"]),
    help="Output format for metadata",
)
@click.version_option(version=VERSION, prog_name="data-dictionary-generator")
def main(
    data_dir: str, dataset_name: str, model: str, output_dir: str, format: str
) -> None:
    """Generate clinical data dictionaries from CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    # Validate model only if user explicitly provided --model (not when default is used)
    ctx = click.get_current_context(silent=True)
    if ctx is not None:
        src = ctx.get_parameter_source("model")
        if src == click.core.ParameterSource.COMMANDLINE:
            validate_ollama_model_available(model)
    all_metadata: list[dict] = []
    all_quality_reports: list[dict] = []

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            result = process_csv(
                file_path, dataset_name, model, output_dir, all_metadata
            )
            if result:
                metadata, quality_report, _ = result
                base_filename = os.path.splitext(file)[0]

                metadata_file = os.path.join(
                    output_dir, f"{base_filename}_metadata.{format}"
                )
                save_metadata_in_format(pd.DataFrame(metadata), format, metadata_file)

                quality_df = pd.DataFrame(quality_report)
                if format == "pdf":
                    quality_file = os.path.join(
                        output_dir, f"{base_filename}_quality.pdf"
                    )
                    save_quality_to_pdf(quality_df, quality_file)
                else:
                    quality_file = os.path.join(
                        output_dir, f"{base_filename}_quality.csv"
                    )
                    quality_df.to_csv(quality_file, index=False)

                all_quality_reports.extend(quality_report)

    if all_metadata:
        relationships_df = generate_relationships_between_tables(all_metadata, data_dir)
        relationships_file = os.path.join(output_dir, f"relationships.{format}")
        relationships_csv = os.path.join(output_dir, "relationships.csv")

        save_metadata_in_format(
            relationships_df, format, relationships_file, is_relationships=True
        )
        relationships_df.to_csv(relationships_csv, index=False)
        save_relationships_to_markdown(
            relationships_df, os.path.join(output_dir, "relationships.md")
        )

    click.echo(f"Success! Files saved in {output_dir}", color=True)


if __name__ == "__main__":
    main()

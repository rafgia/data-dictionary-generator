import os
import pandas as pd
import click
from data_dictionary_generator.generator import (
    process_csv,
    generate_relationships_between_tables,
    save_relationships_to_markdown,
)


def save_metadata_in_format(
    metadata_df: pd.DataFrame, format: str, output_file: str
) -> None:
    """Save metadata in specified format"""
    if format == "csv":
        metadata_df.to_csv(output_file, index=False)
    elif format == "json":
        metadata_df.to_json(output_file, orient="records", lines=True)
    elif format == "pdf":
        click.echo("PDF format is not implemented yet.")
    elif format == "markdown":
        with open(output_file, "w") as f:
            f.write("# Metadata\n")
            for _, row in metadata_df.iterrows():
                f.write(f"## {row['table_name']} - {row['column_name']}\n")
                f.write(f"- **Description**: {row['column_description']}\n")
                f.write(f"- **Data Type**: {row['datatype']}\n")
                f.write(f"- **Sample Data**: {row['sample_data']}\n")
                f.write("\n")
    else:
        click.echo(f"Unsupported format: {format}")


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
def main(
    data_dir: str, dataset_name: str, model: str, output_dir: str, format: str
) -> None:
    """Generate clinical data dictionaries from CSV files."""
    os.makedirs(output_dir, exist_ok=True)
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

                quality_df = pd.DataFrame(
                    [r for r in quality_report if r["column_name"] != "ALL_COLUMNS"]
                )
                quality_file = os.path.join(output_dir, f"{base_filename}_quality.csv")
                quality_df.to_csv(quality_file, index=False)

                all_quality_reports.extend(quality_report)

    if all_metadata:
        relationships_df = generate_relationships_between_tables(all_metadata, data_dir)
        relationships_file = os.path.join(output_dir, f"relationships.{format}")
        relationships_csv = os.path.join(output_dir, "relationships.csv")

        save_metadata_in_format(relationships_df, format, relationships_file)
        relationships_df.to_csv(relationships_csv, index=False)
        save_relationships_to_markdown(
            relationships_df, os.path.join(output_dir, "relationships.md")
        )

    click.echo(f"Success! Files saved in [green]{output_dir}[/green]", color=True)


if __name__ == "__main__":
    main()

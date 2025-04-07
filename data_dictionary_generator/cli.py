import os
import argparse
import pandas as pd
from generator import (
    process_csv,
    generate_relationships_between_tables,
    save_relationships_to_markdown,
)


def save_metadata_in_format(
    metadata_df: pd.DataFrame, format: str, output_file: str
) -> None:
    if format == "csv":
        metadata_df.to_csv(output_file, index=False)
    elif format == "json":
        metadata_df.to_json(output_file, orient="records", lines=True)
    elif format == "pdf":
        print("PDF format is not implemented yet.")
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
        print(f"Unsupported format: {format}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate metadata and relationships from clinical CSV files."
    )
    parser.add_argument("data_dir", type=str, help="Directory with CSV files.")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument(
        "--model", type=str, default="llama3.1", help="LLM model to use."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save metadata and reports.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "json", "pdf", "markdown"],
        help="Output format.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_metadata = []
    all_quality_reports = []

    for file in os.listdir(args.data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(args.data_dir, file)
            result = process_csv(
                file_path, args.dataset_name, args.model, args.output_dir, all_metadata
            )
            if result:
                metadata, quality_report, _ = result
                base_filename = os.path.splitext(file)[0]

                metadata_file = os.path.join(
                    args.output_dir, f"{base_filename}_metadata.{args.format}"
                )
                save_metadata_in_format(
                    pd.DataFrame(metadata), args.format, metadata_file
                )

                quality_df = pd.DataFrame(quality_report)
                quality_df = quality_df[quality_df["column_name"] != "ALL_COLUMNS"]
                quality_file = os.path.join(
                    args.output_dir, f"{base_filename}_quality.csv"
                )
                quality_df.to_csv(quality_file, index=False)

                all_quality_reports.extend(
                    [r for r in quality_report if r["column_name"] != "ALL_COLUMNS"]
                )

    relationships_df = generate_relationships_between_tables(
        all_metadata, args.data_dir
    )
    relationships_file = os.path.join(args.output_dir, f"relationships.{args.format}")
    relationships_csv = os.path.join(args.output_dir, "relationships.csv")
    relationships_md = os.path.join(args.output_dir, "relationships.md")

    save_metadata_in_format(relationships_df, args.format, relationships_file)
    relationships_df.to_csv(relationships_csv, index=False)
    save_relationships_to_markdown(relationships_df, relationships_md)

    print(f"\nDone! All files saved in '{args.output_dir}'")


if __name__ == "__main__":
    main()

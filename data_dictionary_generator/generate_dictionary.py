import argparse
import pathlib
import sys
from metadata_extraction import DatasetMeta, extract_dataset_metadata
from description_generator import (
    generate_dataset_description,
    generate_table_description,
    generate_single_column_description,
    run_llm_dispatcher
)

def generate_full_metadata_and_descriptions(
    dataset_path: pathlib.Path,
    output_dir: pathlib.Path,
    llm_model: str = "llama3.1",
    domain_covered=None
) -> DatasetMeta:

    print(f"Extracting metadata from {dataset_path}...")
    dataset_meta = extract_dataset_metadata(dataset_path, domain_covered, output_dir)

    print("Generating descriptions for dataset...")
    dataset_meta.description = generate_dataset_description(
        dataset_meta,
        llm_model,
        run_llm_dispatcher
    )

    for table_name, table_meta in dataset_meta.tables.items():
        print(f"Generating description for table: {table_name}")
        table_meta.description = generate_table_description(
            table_meta=table_meta,
            dataset_name=dataset_meta.dataset_name,
            dataset_description=dataset_meta.description,
            dataset_domain=dataset_meta.domain_covered or "Unspecified",
            model=llm_model,
            llm_function=run_llm_dispatcher
            )


        for column_name, column_meta in table_meta.columns.items():
            column_meta.description = generate_single_column_description(
                column_meta=column_meta,
                table_name=table_name,
                table_description=table_meta.description,
                dataset_name=dataset_meta.dataset_name,
                dataset_description=dataset_meta.description,
                dataset_domain=dataset_meta.domain_covered or "Unspecified",
                model=llm_model,
                llm_function=run_llm_dispatcher
            )

    print("All descriptions generated successfully.")
    return dataset_meta

def save_metadata_to_json(dataset_meta: DatasetMeta, output_file: pathlib.Path):
    with open(output_file, "w", encoding="utf-8") as f:
        if hasattr(dataset_meta, "model_dump_json"):
            f.write(dataset_meta.model_dump_json(indent=2, exclude_none=True))
        else:
            f.write(dataset_meta.json(indent=2, exclude_none=True))


def save_markdown_summary(dataset_meta: DatasetMeta, output_file: pathlib.Path):
    with open(output_file, "w", encoding="utf-8") as f:

        f.write(f"# Dataset: {dataset_meta.dataset_name}\n\n")
        f.write(f"**Domain:** {dataset_meta.domain_covered or 'N/A'}\n")
        if dataset_meta.time_period_covered:
            start = dataset_meta.time_period_covered["start"]
            end = dataset_meta.time_period_covered["end"]
            f.write(f"**Time coverage:** {start} → {end}\n")
        f.write(f"\n**Overall description:** {dataset_meta.description}\n\n")

        f.write("## Tables\n")
        for table_name, table_meta in dataset_meta.tables.items():
            f.write(f"\n### {table_name}\n")
            f.write(f"- **Rows:** {table_meta.row_count}\n")
            f.write(f"- **Description:** {table_meta.description}\n")
            f.write("- **Columns:**\n")
            for col_name, col_meta in table_meta.columns.items():
                col_type = col_meta.semantic_type or col_meta.data_type
                f.write(f"  - `{col_name}` ({col_type}): {col_meta.description}\n")


def cli():
    parser = argparse.ArgumentParser(
        description="Generate metadata dictionary with descriptions for a dataset folder."
    )

    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the dataset folder containing CSV files"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where output JSON and MD files will be saved"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1",
        help="LLM model to use (default: llama3.1)"
    )

    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Optional domain tag to include in metadata"
    )

    args = parser.parse_args()

    dataset_path = pathlib.Path(args.folder_path)
    output_dir = pathlib.Path(args.output_dir)
    model = args.model
    domain = args.domain

    if not dataset_path.exists():
        print(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_output = output_dir / "dictionary.json"
    md_output = output_dir / "dictionary_summary.md"

    print(f"Using model: {model}")

    dataset_meta = generate_full_metadata_and_descriptions(
        dataset_path=dataset_path,
        output_dir=output_dir,
        llm_model=model,
        domain_covered=domain
    )

    save_metadata_to_json(dataset_meta, json_output)
    save_markdown_summary(dataset_meta, md_output)

    print("\nData dictionary successfully generated.")
    print(f"JSON saved to: {json_output}")
    print(f"Markdown saved to: {md_output}")


if __name__ == "__main__":
    cli()

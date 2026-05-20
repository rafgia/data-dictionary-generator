import argparse
import json
import pathlib
import sys
from data_dictionary_generator.metadata_extraction import (DatasetMeta, extract_dataset_metadata)
from data_dictionary_generator.description_generator import (generate_dataset_description, generate_table_description, generate_single_column_description, run_llm_dispatcher)
from data_dictionary_generator.schema_generation import (infer_schema_with_llm, generate_postgresql_schema)


def generate_full_metadata_and_descriptions(
    dataset_path: pathlib.Path,
    output_dir: pathlib.Path,
    llm_model: str = "llama3.1",
    domain_covered=None
) -> DatasetMeta:
    """
    Extract metadata and generate semantic descriptions.
    """
    print(f"\nExtracting metadata from: {dataset_path}")
    dataset_meta = extract_dataset_metadata(
        dataset_path,
        domain_covered,
        output_dir
    )
    print("\nGenerating dataset description...")
    dataset_meta.description = (
        generate_dataset_description(
            dataset_meta,
            llm_model,
            run_llm_dispatcher
        )
    )
    for table_name, table_meta in (
        dataset_meta.tables.items()
    ):
        print(f"\n[Table] {table_name}")
        table_meta.description = (
            generate_table_description(
                table_meta=table_meta,
                dataset_name=dataset_meta.dataset_name,
                dataset_description=dataset_meta.description,
                dataset_domain=(
                    dataset_meta.domain_covered
                    or "Unspecified"
                ),
                model=llm_model,
                llm_function=run_llm_dispatcher
            )
        )
        for column_name, column_meta in (
            table_meta.columns.items()
        ):
            print(f"   └── {column_name}")
            column_meta.description = (
                generate_single_column_description(
                    column_meta=column_meta,
                    table_name=table_name,
                    table_description=table_meta.description,
                    dataset_name=dataset_meta.dataset_name,
                    dataset_description=dataset_meta.description,
                    dataset_domain=(
                        dataset_meta.domain_covered
                        or "Unspecified"
                    ),
                    model=llm_model,
                    llm_function=run_llm_dispatcher
                )
            )
    print("\nDescriptions generated successfully.")
    return dataset_meta

def save_metadata_to_json(
    dataset_meta: DatasetMeta,
    output_file: pathlib.Path
):
    """
    Save metadata dictionary as JSON.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            dataset_meta.model_dump_json(
                indent=2,
                exclude_none=True
            )
        )

def save_markdown_summary(
    dataset_meta: DatasetMeta,
    output_file: pathlib.Path
):
    """
    Save dataset summary as Markdown.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            f"# Dataset: "
            f"{dataset_meta.dataset_name}\n\n"
        )
        f.write(
            f"**Domain:** "
            f"{dataset_meta.domain_covered or 'N/A'}\n"
        )
        if dataset_meta.time_period_covered:
            start = dataset_meta.time_period_covered.get(
                "start",
                "Unknown"
            )
            end = dataset_meta.time_period_covered.get(
                "end",
                "Unknown"
            )
            f.write(
                f"**Time coverage:** "
                f"{start} → {end}\n"
            )
        f.write("\n## Description\n\n")
        f.write(
            f"{dataset_meta.description}\n\n"
        )
        f.write("## Tables\n")
        for table_name, table_meta in (
            dataset_meta.tables.items()
        ):
            f.write(f"\n### {table_name}\n")
            f.write(
                f"- **Rows:** "
                f"{table_meta.row_count}\n"
            )
            f.write(
                f"- **Description:** "
                f"{table_meta.description}\n"
            )
            f.write("\n#### Columns\n")
            for col_name, col_meta in (
                table_meta.columns.items()
            ):
                col_type = (
                    col_meta.semantic_type
                    or col_meta.data_type
                )
                null_percentage = (
                    (
                        col_meta.null_count
                        / col_meta.total_rows
                    ) * 100
                    if col_meta.total_rows > 0
                    else 0.0
                )
                f.write(
                    f"- `{col_name}` "
                    f"({col_type})"
                    f" — {col_meta.description} "
                    f"[nulls: {null_percentage:.2f}%]\n"
                )

def cli():
    parser = argparse.ArgumentParser(
        description=(
            "Generate semantic metadata dictionaries "
            "and reconstruct relational schemas."
        )
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to dataset folder"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1",
        help="LLM model"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Dataset domain"
    )
    parser.add_argument(
        "--no-schema",
        action="store_false",
        dest="infer_schema",
        help="Skip schema reconstruction"
    )
    args = parser.parse_args()
    dataset_path = pathlib.Path(
        args.folder_path
    )
    output_dir = pathlib.Path(
        args.output_dir
    )
    if not dataset_path.exists():
        print(
            f"ERROR: Dataset path not found: "
            f"{dataset_path}"
        )
        sys.exit(1)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("STEP 1 — Metadata Extraction and Description Generation")

    dataset_meta = (
        generate_full_metadata_and_descriptions(
            dataset_path,
            output_dir,
            args.model,
            args.domain
        )
    )
    schema_meta = None
    if (
        args.infer_schema
        and len(dataset_meta.tables) >= 1
    ):
        print("\nSTEP 2 — Schema Reconstruction")
        schema_meta = infer_schema_with_llm(
            dataset_meta=dataset_meta,
            llm_model=args.model,
            llm_function=run_llm_dispatcher
        )
    print("\nSTEP 3 — Saving Outputs")
    save_markdown_summary(
        dataset_meta,
        output_dir / "dictionary_summary.md"
    )
    if schema_meta:
        with open(
            output_dir / "schema.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                schema_meta.model_dump(),
                f,
                indent=2
            )
        postgres_schema = (
            generate_postgresql_schema(
                dataset_meta,
                schema_meta
            )
        )
        with open(
            output_dir / "schema.sql",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(postgres_schema)
    for table_meta in dataset_meta.tables.values():
        if hasattr(table_meta, "dataframe"):
            table_meta.dataframe = None
    save_metadata_to_json(
        dataset_meta,
        output_dir / "dictionary.json"
    )
    print("\nCOMPLETED")
    print(
        f"\nOutput directory: "
        f"{output_dir}"
    )
    print("\nDone.\n")

if __name__ == "__main__":
    cli()

import argparse
import pandas as pd
import pathlib
import sys
import json
from data_dictionary_generator.metadata_extraction import DatasetMeta, extract_dataset_metadata
from data_dictionary_generator.description_generator import (
    generate_dataset_description,
    generate_table_description,
    generate_single_column_description,
    run_llm_dispatcher)
from data_dictionary_generator.llm_relationships_inference import infer_relationships_with_llm
from data_dictionary_generator.schema_visualization import (generate_er_diagram, save_relationships_summary)

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
        f.write(dataset_meta.model_dump_json(indent=2, exclude_none=True))

def save_markdown_summary(dataset_meta: DatasetMeta, output_file: pathlib.Path):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Dataset: {dataset_meta.dataset_name}\n\n")
        f.write(f"**Domain:** {dataset_meta.domain_covered or 'N/A'}\n")

        if dataset_meta.time_period_covered:
            start = dataset_meta.time_period_covered.get("start", "Unknown")
            end = dataset_meta.time_period_covered.get("end", "Unknown")
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
                null_percentage = (col_meta.null_count / col_meta.total_rows * 100) if col_meta.total_rows > 0 else 0.0
                
                f.write(f"  - `{col_name}` ({col_type}): {col_meta.description} [nulls: {null_percentage:.2f}%]\n")

def cli():
    parser = argparse.ArgumentParser(description="Generate metadata dictionary and schema visualization.")
    parser.add_argument("folder_path", type=str, help="Path to the dataset folder")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="llama3.1")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--no-relationships", action="store_false", dest="infer_relationships")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    
    args = parser.parse_args()
    dataset_path, output_dir = pathlib.Path(args.folder_path), pathlib.Path(args.output_dir)

    if not dataset_path.exists():
        print(f"Error: Path {dataset_path} not found.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate metadata and descriptions
    dataset_meta = generate_full_metadata_and_descriptions(
        dataset_path, output_dir, args.model, args.domain
    )
    
    found_any_rels = False
    if args.infer_relationships and len(dataset_meta.tables) >= 2:
        # Use LLM to infer relationships
        relationships = infer_relationships_with_llm(
            dataset_meta,
            args.model,
            run_llm_dispatcher,
            args.min_confidence
        )
        
        if relationships:
            found_any_rels = True
            
            # Save relationships summary
            rel_summary_path = output_dir / "relationships_summary.md"
            save_relationships_summary(relationships, rel_summary_path)
            
            # Save as JSON
            with open(output_dir / "relationships.json", 'w', encoding='utf-8') as f:
                rels_data = [r.model_dump() for r in relationships]
                json.dump(rels_data, f, indent=2)
            
            # Generate ER diagram
            generate_er_diagram(dataset_meta, relationships, output_dir / "schema_diagram.mmd")
            
            print(f"\nFound {len(relationships)} relationships")
        else:
            print("\nNo relationships detected")

    # Save outputs
    save_markdown_summary(dataset_meta, output_dir / "dictionary_summary.md")
    
    # Clean up dataframes before saving JSON
    for table_meta in dataset_meta.tables.values():
        if hasattr(table_meta, "dataframe"):
            table_meta.dataframe = None
    
    save_metadata_to_json(dataset_meta, output_dir / "dictionary.json")
    
    if found_any_rels:
        print(f"\nDiagram saved to: {output_dir / 'schema_diagram.mmd'}")
        print(f"Summary saved to: {output_dir / 'relationships_summary.md'}")

if __name__ == "__main__":
    cli()

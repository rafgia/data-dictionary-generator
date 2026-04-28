import pathlib
from typing import List
from relationship_inference import RelationshipMeta

def clean_name(name: str) -> str:
    """Ensure table and column names are Mermaid-safe."""
    return name.replace(" ", "_").replace("-", "_").replace(".", "_")

def generate_er_diagram(
    dataset_meta,
    relationships: List[RelationshipMeta],
    output_path: pathlib.Path
) -> pathlib.Path:
    lines = ["erDiagram"]

    for table_name, table_meta in dataset_meta.tables.items():
        safe_table = clean_name(table_name)
        lines.append(f"    {safe_table} {{")

        cols = list(table_meta.columns.items())
        for col_name, col_meta in cols[:8]:
            safe_col = clean_name(col_name)
            col_type = col_meta.data_type
            
            # Stricter PK definition: Must be highly unique AND named like an ID
            is_pk = (col_meta.unique_count / col_meta.total_rows > 0.99) and \
                    ('id' in col_name.lower() or 'pk' in col_name.lower())
            pk_marker = "PK" if is_pk else ""
            
            lines.append(f"        {col_type} {safe_col} {pk_marker}".strip())

        if len(cols) > 8:
            lines.append("        additional_columns hidden")
        
        lines.append("    }")

    for rel in relationships:
        mapping = {
            "1:1": "||--||",
            "1:N": "||--o{",
            "N:1": "}o--||",
            "N:M": "}o--o{"
        }
        card = mapping.get(rel.relationship_type, "}o--o{")
        
        from_t = clean_name(rel.from_table)
        to_t = clean_name(rel.to_table)
        label = clean_name(f"{rel.from_column}_{rel.to_column}")
        
        lines.append(f'    {from_t} {card} {to_t} : "{label}"')

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

def save_relationships_summary(
    relationships: List[RelationshipMeta], 
    output_path: pathlib.Path
) -> pathlib.Path:
    """
    Save a structured Markdown summary of all inferred relationships.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Inferred Entity Relationships\n\n")
        f.write(f"Total potential relationships identified: **{len(relationships)}**\n\n")
        if not relationships:
            f.write("> _No relationships were inferred based on current thresholds._\n")
            return output_path
        
        f.write("## Relationship Details\n\n")
        f.write("| Source Table.Column | Type | Target Table.Column | Confidence | Method |\n")
        f.write("| :--- | :---: | :--- | :---: | :--- |\n")
        
        # Sort by confidence descending so the best matches are at the top
        sorted_rels = sorted(relationships, key=lambda x: x.confidence, reverse=True)
        
        for rel in sorted_rels:            
            f.write(
                f"| `{rel.from_table}.{rel.from_column}` "
                f"| **{rel.relationship_type}** "
                f"| `{rel.to_table}.{rel.to_column}` "
                f"| {rel.confidence:.2f} "
                f"| {rel.inference_method} |\n"
            )    
    return output_path
import pathlib
from typing import List
from data_dictionary_generator.llm_relationships_inference import RelationshipMeta

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
            unique_ratio = col_meta.unique_count / col_meta.total_rows if col_meta.total_rows else 0
            is_pk = unique_ratio > 0.98 and col_meta.data_type in ["integer", "string"]
            pk_marker = "PK" if is_pk else ""
            lines.append(f"        {col_type} {safe_col} {pk_marker}".strip())
        if len(cols) > 8:
            lines.append("        additional_columns hidden")
        lines.append("    }")
    relationships = [r for r in relationships if r.confidence >= 0.85]
    best_rel_per_target = {}
    for r in relationships:
        key = (r.to_table, r.to_column)
        if key not in best_rel_per_target or r.confidence > best_rel_per_target[key].confidence:
            best_rel_per_target[key] = r
    relationships = list(best_rel_per_target.values())
    
    mapping = {
        "1:1": "||--||",
        "1:N": "||--o{",
    }

    for rel in relationships:
        card = mapping.get(rel.relationship_type, "||--o{")
        from_t = clean_name(rel.from_table)
        to_t = clean_name(rel.to_table)
        label = f"{clean_name(rel.from_column)} → {clean_name(rel.to_column)}"
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
        f.write(f"Total relationships identified: **{len(relationships)}**\n\n")
        
        if not relationships:
            f.write("> _No relationships were inferred._\n")
            return output_path
        
        f.write("## Relationship Details\n\n")
        f.write("| From | Type | To | Confidence | Reasoning |\n")
        f.write("| :--- | :---: | :--- | :---: | :--- |\n")
        
        sorted_rels = sorted(relationships, key=lambda x: x.confidence, reverse=True)
        for rel in sorted_rels:
            reasoning = rel.reasoning or ""
            if reasoning.startswith("llm_analysis: "):
                reasoning = reasoning[14:]  # Remove prefix
            
            f.write(
                f"| `{rel.from_table}.{rel.from_column}` "
                f"| **{rel.relationship_type}** "
                f"| `{rel.to_table}.{rel.to_column}` "
                f"| {rel.confidence:.2f} "
                f"| {reasoning} |\n"
            )
    
    return output_path

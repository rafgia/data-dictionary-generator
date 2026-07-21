import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from json_repair import repair_json
from data_dictionary_generator.metadata_extraction import ( DatasetMeta, TableMeta, ColumnMeta)

class ForeignKeyMeta(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    cardinality: str

class TableSchemaMeta(BaseModel):
    table_name: str
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: List[ForeignKeyMeta] = Field(default_factory=list)

class DatabaseSchemaMeta(BaseModel):
    tables: List[TableSchemaMeta] = Field(default_factory=list)

def compute_unique_ratio(column_meta: ColumnMeta) -> float:
    if column_meta.total_rows == 0:
        return 0.0
    return round(
        column_meta.unique_count / column_meta.total_rows,
        4
    )

def compute_null_ratio(column_meta: ColumnMeta) -> float:
    if column_meta.total_rows == 0:
        return 0.0
    return round(
        column_meta.null_count / column_meta.total_rows,
        4
    )

def detect_candidate_key(column_meta: ColumnMeta) -> bool:
    unique_ratio = compute_unique_ratio(column_meta)
    null_ratio = compute_null_ratio(column_meta)
    identifier_keywords = [
        "id",
        "uuid",
        "key"
    ]
    column_name = column_meta.column_name.lower()
    has_identifier_name = any(
        keyword in column_name
        for keyword in identifier_keywords
    )
    return (
        unique_ratio >= 0.98
        and null_ratio <= 0.01
        and has_identifier_name
    )

def summarize_column(column_meta: ColumnMeta) -> str:
    """
    Create a compact structured representation
    of a column for the LLM prompt.
    """
    unique_ratio = compute_unique_ratio(column_meta)
    null_ratio = compute_null_ratio(column_meta)
    candidate_key = detect_candidate_key(column_meta)
    sample_values = column_meta.sample_values[:5]
    sample_str = ", ".join(
        [str(v) for v in sample_values]
    )
    lines = [
        f"column_name: {column_meta.column_name}",
        f"data_type: {column_meta.data_type}",
        f"semantic_type: {column_meta.semantic_type or 'unknown'}",
        f"unique_ratio: {unique_ratio}",
        f"null_ratio: {null_ratio}",
        f"candidate_key: {candidate_key}",
        f"examples: [{sample_str}]"
    ]
    if column_meta.description:
        lines.append(
            f"description: {column_meta.description[:120]}"
        )
    return "\n".join(lines)

def summarize_table(table_name: str,
                    table_meta: TableMeta) -> str:
    """
    Create a structured table summary.
    """
    lines = [
        "=" * 60,
        f"TABLE: {table_name}"
    ]
    if table_meta.description:
        lines.append(
            f"DESCRIPTION: {table_meta.description}"
        )
    lines.append("\nCOLUMNS:\n")
    for _, column_meta in table_meta.columns.items():
        lines.append(
            summarize_column(column_meta)
        )
        lines.append("")
    return "\n".join(lines)

def build_lightweight_table_index(
    dataset_meta: DatasetMeta,
    exclude_table: Optional[str] = None
) -> str:
    lines = []
    for table_name, table_meta in dataset_meta.tables.items():
        if table_name == exclude_table:
            continue
        col_entries = []
        for col_name, column_meta in table_meta.columns.items():
            marker = " [candidate_key]" if detect_candidate_key(column_meta) else ""
            col_entries.append(f"{col_name}{marker}")
        lines.append(f"{table_name}: {', '.join(col_entries)}")
    return "\n".join(lines)

TABLE_SCHEMA_PROMPT_TEMPLATE = """
You are a senior PostgreSQL database architect.

Your task is to reconstruct the primary key and foreign keys for ONE
table, using ONLY the provided metadata. Foreign keys may reference
columns in the other tables listed in the OTHER TABLES index below.

============================================================
CRITICAL RULES
============================================================

- NEVER invent tables
- NEVER invent columns
- NEVER invent relationships
- Use ONLY provided metadata
- If uncertain, omit the relationship
- Prefer precision over recall
- Candidate keys (marked [candidate_key]) are likely primary keys or
  foreign key targets
- Foreign keys usually reference primary keys
- Cardinality must be one of:
    - one-to-one
    - one-to-many
    - many-to-one
    - many-to-many

============================================================
OUTPUT FORMAT
============================================================

Return ONLY STRICT JSON for THIS table only (no wrapping "tables" list).

DO NOT include explanations.
DO NOT include markdown.
DO NOT include comments.

JSON FORMAT:

{{
  "table_name": "{table_name}",
  "primary_keys": [
    "column_name"
  ],
  "foreign_keys": [
    {{
      "from_table": "{table_name}",
      "from_column": "column_name",
      "to_table": "other_table_name",
      "to_column": "column_name",
      "cardinality": "many-to-one"
    }}
  ]
}}

============================================================
TARGET TABLE METADATA
============================================================

{table_text}

============================================================
OTHER TABLES (name: columns, candidate keys marked)
============================================================

{other_tables_index}
"""

def build_table_schema_prompt(
    table_name: str,
    dataset_meta: DatasetMeta
) -> str:
    """
    Build a schema reconstruction prompt scoped to a single table.
    """
    table_meta = dataset_meta.tables[table_name]
    table_text = summarize_table(table_name, table_meta)
    other_tables_index = build_lightweight_table_index(dataset_meta, exclude_table=table_name)
    return TABLE_SCHEMA_PROMPT_TEMPLATE.format(
        table_name=table_name,
        table_text=table_text,
        other_tables_index=other_tables_index or "(no other tables)"
    )

def validate_fk(
    fk: dict,
    dataset_meta: DatasetMeta
) -> bool:
    """
    Validate FK references.
    """
    required_fields = [
        "from_table",
        "from_column",
        "to_table",
        "to_column",
        "cardinality"
    ]
    for field in required_fields:
        if field not in fk:
            return False
    from_table = fk["from_table"]
    to_table = fk["to_table"]
    if from_table not in dataset_meta.tables:
        return False
    if to_table not in dataset_meta.tables:
        return False
    from_column = fk["from_column"]
    to_column = fk["to_column"]
    if from_column not in (
        dataset_meta.tables[from_table].columns
    ):
        return False
    if to_column not in (
        dataset_meta.tables[to_table].columns
    ):
        return False
    valid_cardinalities = {
        "one-to-one",
        "one-to-many",
        "many-to-one",
        "many-to-many"
    }
    if fk["cardinality"] not in valid_cardinalities:
        return False
    return True

def parse_single_table_schema_response(
    response: Optional[str],
    table_name: str,
    dataset_meta: DatasetMeta
) -> Optional[TableSchemaMeta]:
    if not response:
        return None
    try:
        repaired = repair_json(
            response,
            return_objects=False
        )
        data = json.loads(repaired)
    except Exception:
        return None

    if not isinstance(data, dict) or "primary_keys" not in data and "foreign_keys" not in data:
        return None

    valid_columns = set(
        dataset_meta.tables[table_name].columns.keys()
    )
    primary_keys = [
        pk for pk in data.get("primary_keys", [])
        if pk in valid_columns
    ]
    foreign_keys = []
    for fk_data in data.get("foreign_keys", []):
        if not validate_fk(fk_data, dataset_meta):
            continue
        foreign_keys.append(ForeignKeyMeta(**fk_data))

    return TableSchemaMeta(
        table_name=table_name,
        primary_keys=primary_keys,
        foreign_keys=foreign_keys
    )

def remove_duplicate_foreign_keys(
    schema_meta: DatabaseSchemaMeta
) -> DatabaseSchemaMeta:
    """
    Remove duplicated FK definitions.
    """
    for table in schema_meta.tables:
        unique_fks = {}
        for fk in table.foreign_keys:
            key = (
                fk.from_table,
                fk.from_column,
                fk.to_table,
                fk.to_column
            )
            unique_fks[key] = fk
        table.foreign_keys = list(
            unique_fks.values()
        )
    return schema_meta

def infer_schema_with_llm(
    dataset_meta: DatasetMeta,
    llm_model: str,
    llm_function,
    max_output_tokens: int = 1500
) -> DatabaseSchemaMeta:
    print(
        f"\nReconstructing schema for dataset: "
        f"{dataset_meta.dataset_name} "
        f"({len(dataset_meta.tables)} tables, one schema call per table)"
    )
    parsed_tables = []
    failed_tables = []
    for table_name in dataset_meta.tables:
        print(f"   [Schema] {table_name}")
        prompt = build_table_schema_prompt(table_name, dataset_meta)
        try:
            response = llm_function(
                prompt=prompt,
                model=llm_model,
                system_message="Return ONLY valid JSON.",
                is_json=True,
                temperature=0,
                max_output_tokens=max_output_tokens
            )
        except Exception as e:
            print(f"      WARNING: schema call failed for '{table_name}': {e}")
            parsed_tables.append(TableSchemaMeta(table_name=table_name))
            failed_tables.append(table_name)
            continue

        table_schema = parse_single_table_schema_response(
            response, table_name, dataset_meta
        )
        if table_schema is None:
            print(
                f"      WARNING: could not parse a schema for '{table_name}' "
                f"(empty or invalid/truncated JSON response) - this table "
                f"will have no primary/foreign keys in the output schema."
            )
            parsed_tables.append(TableSchemaMeta(table_name=table_name))
            failed_tables.append(table_name)
            continue

        parsed_tables.append(table_schema)

    schema_meta = remove_duplicate_foreign_keys(
        DatabaseSchemaMeta(tables=parsed_tables)
    )

    if failed_tables:
        print(
            f"\nSchema reconstruction completed with {len(failed_tables)} "
            f"of {len(dataset_meta.tables)} table(s) failing to parse: "
            f"{', '.join(failed_tables)}. Re-running with a higher "
            f"--schema-max-tokens value may help if these kept truncating."
        )
    else:
        print("\nSchema reconstruction completed.")

    return schema_meta

def map_postgres_type(
    data_type: str
) -> str:
    """
    Map inferred types to PostgreSQL types.
    """
    data_type = data_type.lower()
    if data_type in ["integer", "int"]:
        return "INTEGER"
    if data_type in ["float", "double"]:
        return "DOUBLE PRECISION"
    if data_type in ["boolean", "bool"]:
        return "BOOLEAN"
    if data_type in ["datetime", "timestamp"]:
        return "TIMESTAMP"
    if data_type in ["date"]:
        return "DATE"
    return "TEXT"

def generate_postgresql_schema(
    dataset_meta: DatasetMeta,
    schema_meta: DatabaseSchemaMeta
) -> str:
    """
    Deterministically generate PostgreSQL DDL.
    """
    ddl_parts = []
    for table_schema in schema_meta.tables:
        table_name = table_schema.table_name
        table_meta = dataset_meta.tables[
            table_name
        ]
        lines = [
            f"CREATE TABLE {table_name} ("
        ]
        column_lines = []
        for column_name, column_meta in (
            table_meta.columns.items()
        ):
            postgres_type = map_postgres_type(
                column_meta.data_type
            )
            column_def = (
                f"    {column_name} "
                f"{postgres_type}"
            )
            if (
                column_name
                in table_schema.primary_keys
            ):
                column_def += " PRIMARY KEY"
            column_lines.append(column_def)
        for fk in table_schema.foreign_keys:
            fk_name = (
                f"fk_"
                f"{fk.from_table}_"
                f"{fk.from_column}"
            )
            fk_line = (
                f"    CONSTRAINT {fk_name} "
                f"FOREIGN KEY ({fk.from_column}) "
                f"REFERENCES "
                f"{fk.to_table}({fk.to_column})"
            )
            column_lines.append(fk_line)
        lines.append(
            ",\n".join(column_lines)
        )
        lines.append(");")
        ddl_parts.append(
            "\n".join(lines)
        )
    return "\n\n".join(ddl_parts)

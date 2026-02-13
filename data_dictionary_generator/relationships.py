from typing import List, TYPE_CHECKING
from rapidfuzz import fuzz
from pydantic import BaseModel
import pandas as pd
import pathlib
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

if TYPE_CHECKING:
    from metadata_extraction import DatasetMeta

class Relationship(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str
    confidence: float
    confidence_name: float
    confidence_values: float

def detect_primary_keys(df: pd.DataFrame) -> List[str]:
    primary_keys = []
    for col in df.columns:
        series = df[col]
        if series.isnull().mean() > 0.1:
            continue
        unique_ratio = series.nunique(dropna=True) / max(1, series.notnull().sum())
        if unique_ratio > 0.95:
            primary_keys.append(col)
    return primary_keys


def mark_primary_keys(dataset: "DatasetMeta") -> None:
    for table_name, table_meta in dataset.tables.items():
        if not hasattr(table_meta, "dataframe"):
            continue
        df = table_meta.dataframe
        pks = detect_primary_keys(df)
        for col in pks:
            if col in table_meta.columns:
                table_meta.columns[col].is_primary_key = True


def jaccard_similarity(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def detect_foreign_keys(
    dataset: "DatasetMeta",
    output_dir: pathlib.Path,
    min_confidence: float = 0.6,
    make_graph: bool = True,
) -> List[Relationship]:

    output_dir.mkdir(parents=True, exist_ok=True)
    relationships: List[Relationship] = []

    pk_values_per_table = {}
    for table_name, table_meta in dataset.tables.items():
        df = table_meta.dataframe
        pk_values = {
            col: set(df[col].dropna().unique().tolist())
            for col, meta in table_meta.columns.items()
            if meta.is_primary_key
        }
        pk_values_per_table[table_name] = pk_values

    for s_table, s_table_meta in dataset.tables.items():
        s_df = s_table_meta.dataframe

        for s_col, s_meta in s_table_meta.columns.items():

            if s_meta.is_primary_key:
                continue

            s_values = set(s_df[s_col].dropna().unique().tolist())
            if not s_values:
                continue

            for t_table, t_pks in pk_values_per_table.items():
                if t_table == s_table:
                    continue

                for t_col, t_vals in t_pks.items():
                    if not t_vals:
                        continue

                    name_sim = fuzz.ratio(s_col.lower(), t_col.lower()) / 100
                    value_sim = jaccard_similarity(s_values, t_vals)

                    final_conf = 0.6 * name_sim + 0.4 * value_sim

                    if final_conf >= min_confidence:
                        relationships.append(
                            Relationship(
                                source_table=s_table,
                                source_column=s_col,
                                target_table=t_table,
                                target_column=t_col,
                                relationship_type="foreign_key",
                                confidence=float(final_conf),
                                confidence_name=float(name_sim),
                                confidence_values=float(value_sim),
                            )
                        )


    if make_graph and relationships:
        G = nx.DiGraph()

        for rel in relationships:
            G.add_edge(rel.source_table, rel.target_table)

        plt.figure(figsize=(16, 12))

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except:
            pos = nx.spring_layout(G, k=0.8)

        node_sizes = []
        for table_name in G.nodes():
            has_pk = any(
                col.is_primary_key
                for col in dataset.tables[table_name].columns.values()
            )
            node_sizes.append(2600 if has_pk else 1500)

        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color="#ADD8E6",
            edgecolors="black",
            node_shape="s",
        )

        nx.draw_networkx_labels(G, pos, font_size=10)

        edge_colors = [rel.confidence for rel in relationships]

        edges = nx.draw_networkx_edges(
            G, pos,
            arrowstyle="->",
            arrowsize=18,
            edge_color=edge_colors,
            edge_cmap=plt.cm.viridis,
            width=2,
        )
        sm = ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array(edge_colors)  # colors used
        plt.colorbar(sm, label="FK confidence score") 
        edge_labels = {
            (rel.source_table, rel.target_table):
                f"{rel.source_column} → {rel.target_column}"
            for rel in relationships
        }

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=8
        )

        plt.title(f"{dataset.dataset_name} – PK/FK Relationship Graph")
        plt.axis("off")

        graph_path = output_dir / f"{dataset.dataset_name}_relationship_graph.png"
        plt.savefig(graph_path, dpi=200, bbox_inches="tight")
        print(f"\nGraph saved to: {graph_path.absolute()}")

    md_path = output_dir / "list_of_relationships.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# List of PK/FK relationships\n\n")
        f.write(f"Dataset: **{dataset.dataset_name}**\n\n")

        if not relationships:
            f.write("No foreign key relationships detected.\n")
        else:
            for rel in relationships:
                f.write(
                    f"- **{rel.source_table}.{rel.source_column} → "
                    f"{rel.target_table}.{rel.target_column}**  "
                    f"(confidence: {rel.confidence:.3f}, "
                    f"name: {rel.confidence_name:.3f}, "
                    f"values: {rel.confidence_values:.3f})\n"
                )

    print(f"Markdown file saved to: {md_path.absolute()}")
    return relationships

def add_relationships_to_metadata(
    dataset: "DatasetMeta",
    relationships: List[Relationship]
):
    for rel in relationships:
        col_meta = dataset.tables[rel.source_table].columns[rel.source_column]
        col_meta.foreign_key_to = f"{rel.target_table}.{rel.target_column}"

def detect_primary_and_foreign_keys(
    dataset: "DatasetMeta",
    output_dir: pathlib.Path
) -> List[Relationship]:
    mark_primary_keys(dataset)
    relationships = detect_foreign_keys(dataset, output_dir=output_dir)
    add_relationships_to_metadata(dataset, relationships)
    return relationships

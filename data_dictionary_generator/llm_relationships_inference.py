import json
from typing import List
import re
from pydantic import BaseModel

class RelationshipMeta(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str
    confidence: float
    reasoning: str 

def metadata_for_llm(dataset_meta) -> str:
    """
    Convert metadata to a format suitable for LLM analysis.
    """
    lines = []
    lines.append(f"Dataset: {dataset_meta.dataset_name}")
    lines.append(f"Domain: {dataset_meta.domain_covered or 'Unspecified'}")
    lines.append(f"Description: {dataset_meta.description}\n")
    
    for table_name, table_meta in dataset_meta.tables.items():
        lines.append(f"\n## Table: {table_name}")
        lines.append(f"Rows: {table_meta.row_count}")
        lines.append(f"Description: {table_meta.description}")
        lines.append("\nColumns:")
        
        for col_name, col_meta in table_meta.columns.items():
            unique_ratio = col_meta.unique_count / col_meta.total_rows if col_meta.total_rows > 0 else 0
            null_pct = (col_meta.null_count / col_meta.total_rows * 100) if col_meta.total_rows > 0 else 0
            
            lines.append(f"\n  {col_name}:")
            lines.append(f"    Type: {col_meta.data_type}")
            if col_meta.semantic_type:
                lines.append(f"    Semantic: {col_meta.semantic_type}")
            lines.append(f"    Description: {col_meta.description}")
            lines.append(f"    Uniqueness: {unique_ratio:.1%}, Nulls: {null_pct:.1f}%")
            if col_meta.sample_values:
                samples = col_meta.sample_values[:10]
                lines.append(f"    Samples: {samples}")
    
    return "\n".join(lines)


def build_relationship_prompt(dataset_meta) -> str:
    """
    Build a prompt asking LLM to identify relationships.
    """
    metadata_text = metadata_for_llm(dataset_meta)
    
    prompt = f"""You are a database schema expert. Analyze the following dataset and identify ALL foreign key relationships between tables.

{metadata_text}

## Task

Identify foreign key relationships by analyzing:
- Column names (e.g., PatientID in one table likely references Patients table)
- Descriptions (what the columns represent)
- Data types (must match)
- Uniqueness ratios (parent should be >90% unique)
- Sample values (should overlap between related columns)
- Domain knowledge (healthcare relationships, common patterns)

For each relationship, specify:
- from_table: Parent table (has the primary key)
- from_column: Parent column (the primary key)
- to_table: Child table (has the foreign key)
- to_column: Child column (the foreign key)
- relationship_type: Must be exactly "1:1", "1:N", or "N:M"
- confidence: Number between 0.6 and 1.0
- reasoning: Brief explanation (one sentence)

## Important Rules

1. Parent column should have uniqueness >90%
2. Only suggest relationships where columns have the same data type
3. Consider table and column descriptions carefully
4. Use domain knowledge (e.g., in healthcare, events often reference patients)
5. One patient can have many events = 1:N relationship
6. If both sides have high uniqueness (>90%) = 1:1 relationship
7. Only include relationships with confidence >= 0.6

## Output Format

You MUST respond with ONLY a JSON array. No explanations, no markdown, no code blocks.
Start with [ and end with ].

Example output:
[
  {{
    "from_table": "Patients",
    "from_column": "PatientID",
    "to_table": "Visits",
    "to_column": "PatientID",
    "relationship_type": "1:N",
    "confidence": 0.95,
    "reasoning": "PatientID in Patients is 98% unique (primary key), and Visits.PatientID references it"
  }}
]

If no valid relationships exist, return: []

RESPOND WITH ONLY VALID JSON.

STRICT JSON RULES:
    1. Every key MUST be in double quotes: "from_table", NOT from_table.
    2. Every string value MUST be in double quotes.
    3. NO trailing commas.
    4. NO comments.
    5. NO markdown formatting blocks (no ```json).
    6. Ensure keys are EXACTLY: "from_table", "from_column", "to_table", "to_column", "relationship_type", "confidence", "reasoning".

VALID example:
{{ "from_table": "Patients" }}"""
    
    return prompt


def parse_llm_response(response: str) -> List[RelationshipMeta]:
    """
    Parse LLM response and convert to RelationshipMeta objects.
    """
    response = response.strip()

    def fix_json(m):
        key = m.group(1).lower()
        mapping = {
            "fromtable": "from_table",
            "fromcolumn": "from_column",
            "totable": "to_table",
            "tocolumn": "to_column",
            "relationshiptype": "relationship_type"
        }
        mapped_key = mapping.get(key, key)
        return f'"{mapped_key}":'
    
    response = re.sub(r'(?<!")(\w+)\s*:', fix_json, response)
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
    elif "```" in response:
        response = response.split("```")[1].split("```")[0]
    
    response = response.strip()
    
    if not response.startswith("["):
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            response = match.group()
        else:
            print(f"[ERROR] Could not find JSON array in response")
            print(f"Response preview: {response[:300]}")
            return []
    
    try:
        data = json.loads(response)
        
        if not isinstance(data, list):
            print(f"[ERROR] Response is not a list: {type(data)}")
            return []
        
        relationships = []
        for item in data:
            try:
                required = ["from_table", "from_column", "to_table", "to_column", 
                           "relationship_type", "confidence", "reasoning"]
                if not all(k in item for k in required):
                    print(f"[WARNING] Missing required fields in: {item}")
                    continue
                if item["relationship_type"] not in ["1:1", "1:N", "N:M"]:
                    print(f"[WARNING] Invalid relationship_type: {item['relationship_type']}")
                    item["relationship_type"] = "1:N"  # Default
                
                rel = RelationshipMeta(
                    from_table=item["from_table"],
                    from_column=item["from_column"],
                    to_table=item["to_table"],
                    to_column=item["to_column"],
                    relationship_type=item["relationship_type"],
                    confidence=float(item["confidence"]),
                    reasoning=item.get("reasoning")
                )
                relationships.append(rel)
                
            except Exception as e:
                print(f"[WARNING] Failed to parse relationship: {e}")
                print(f"Item: {item}")
                continue
        
        return relationships
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode error: {e}")
        print(f"Response preview: {response[:300]}")
        return []


def infer_relationships_with_llm(
    dataset_meta,
    llm_model: str,
    llm_function,
    min_confidence: float = 0.6
) -> List[RelationshipMeta]:
    """
    Use LLM to infer relationships from metadata and descriptions.
    """
    print("\nAnalyzing relationships with LLM...")
    prompt = build_relationship_prompt(dataset_meta)
    
    system_message = (
        "You are a strict JSON generator. "
        "Output MUST be valid JSON. "
        "All keys and string values MUST use double quotes. "
        "Do NOT omit quotes. "
        "Do NOT include explanations."
    )
    
    try:
        response = llm_function(prompt, llm_model, system_message=system_message, is_json=True)
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return []
    relationships = parse_llm_response(response)
    
    print(f"LLM identified {len(relationships)} relationships")
    relationships = [r for r in relationships if r.confidence >= min_confidence]
    validated = []
    for rel in relationships:
        if rel.from_table not in dataset_meta.tables:
            print(f"[WARNING] Unknown table: {rel.from_table}")
            continue
        if rel.to_table not in dataset_meta.tables:
            print(f"[WARNING] Unknown table: {rel.to_table}")
            continue
        from_cols = dataset_meta.tables[rel.from_table].columns
        to_cols = dataset_meta.tables[rel.to_table].columns
        
        if rel.from_column not in from_cols:
            print(f"[WARNING] Unknown column: {rel.from_table}.{rel.from_column}")
            continue
        if rel.to_column not in to_cols:
            print(f"[WARNING] Unknown column: {rel.to_table}.{rel.to_column}")
            continue
        
        validated.append(rel)
        print(f"  {rel.from_table}.{rel.from_column} → {rel.to_table}.{rel.to_column} "
              f"({rel.relationship_type}, conf={rel.confidence:.2f})")
    
    return validated

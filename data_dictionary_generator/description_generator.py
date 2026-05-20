import re
import os
import requests
from typing import Optional
import logging
from pydantic import BaseModel, Field
from data_dictionary_generator.metadata_extraction import DatasetMeta, TableMeta, ColumnMeta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_MODELS = ["gpt-5.5","gpt-5.4","gpt-5.4-mini","gpt-5","gpt-5-mini","gpt-5-nano","gpt-5-pro","gpt-5.5-pro",
    "gpt-4.1","gpt-4o","gpt-4o-mini","gpt-4-turbo","gpt-4","gpt-3.5-turbo"]
OLLAMA_URL = "http://localhost:11434"

DATASET_PROMPT_TEMPLATE = """
You are an expert in dataset documentation. 
Generate a short, precise, and factual description of the dataset.
Do NOT write introductions, explanations, examples, or subjective opinions.
Do NOT repeat the dataset name or make assumptions.

Dataset overview:
- Domain: {{domain}}
- Tables: {{table_list}}
- Row counts per table: {{table_row_counts}}
- Time coverage: {{time_range}}

Output:
- A single sentence (max 15 words) describing the dataset’s content, purpose, and domain.
"""

TABLE_PROMPT_TEMPLATE = """
You are an expert in data documentation. 
Generate a concise and precise description of the table.
Do NOT give examples, assumptions, or repeat the table name.
Use only the provided metadata.

Dataset context:
- Name: {{dataset_name}}
- Domain: {{dataset_domain}}
- Description: {{dataset_description}}

Table metadata:
- Name: {{table_name}}
- Columns (name and type): {{column_list_with_types}}
- Number of rows: {{n_rows}}
- Sample rows: {{sample_rows}}

Output:
- One short sentence (max 15 words) describing what the table represents and its content.
- Include factual details only.
"""

COLUMN_PROMPT_TEMPLATE = """
You are an expert in data documentation and metadata analysis.
Generate a precise, factual, single-sentence description of the column.

Constraints:
1. Do NOT speculate, give examples, or repeat column/table names.
2. Do NOT repeat null percentage, distinct counts, or stats summary in the text.
3. Do NOT start descriptions with "Represents", "Indicates", "Contains", or "Column". Start directly with the subject (e.g., "Patient age..." instead of "Indicates patient age...").

Goals:
- Capture the unit of measurements (e.g., hours, kg, ml) if implied by sample values.
- Use the stats only to infer if the data is categorical or continuous, then write the description accordingly.

Dataset context:
- Name: {{dataset_name}}
- Domain: {{dataset_domain}}
- Description: {{dataset_description}}

Table context:
- Name: {{table_name}}
- Description: {{table_description}}

Column metadata:
- Name: {{column_name}}
- Type: {{type}}
- Sample Values: {{samples}}
- Unique Values: {{unique}}
- Null Percentage: {{null_percent}}
- Summary Statistics: {{stats_summary}}


Output:
- One factual sentence (max 15 words).
"""

def _calculate_null_percent(col_meta: ColumnMeta) -> str:
    if col_meta.total_rows == 0:
        return "N/A"
    percent = (col_meta.null_count / col_meta.total_rows) * 100
    return f"{percent:.1f}%"

def _format_column_stats(col_meta: ColumnMeta) -> str:
    stats = col_meta.stats
    if not stats or stats.get("type") is None:
        return f"Unique values: {stats.get('unique_count', 'N/A')}" if stats else "N/A"

    if stats["type"] == "numeric":
        return (f"numerical: [min: {stats.get('min', 'N/A')}, max: {stats.get('max', 'N/A')}, "
                f"mean: {stats.get('mean', 'N/A')}, median: {stats.get('median', 'N/A')}]")
    elif stats["type"] == "datetime":
        return f"datetime range: [{stats.get('min_date', 'N/A')} to {stats.get('max_date', 'N/A')}]"
    elif stats["type"] == "categorical":
        top_vals = stats.get('top_values', [])
        if top_vals:
            top_5 = ", ".join([str(v[0]) for v in top_vals[:5]])
            return f"categorical: top_5_values={{{top_5}}}"
        return f"categorical: unique_count={stats.get('unique_count', 'N/A')}"
    return "N/A"

def _format_sample_rows(table_meta: TableMeta) -> str:
    if not table_meta.sample_rows:
        return "N/A"
    return "\n".join([str(row) for row in table_meta.sample_rows])

def generate_dataset_description(dataset_meta: DatasetMeta, model: str, llm_function, **kwargs) -> str:
    table_counts = {t_name: t_meta.row_count for t_name, t_meta in dataset_meta.tables.items()}
    time_range = (f"{dataset_meta.time_period_covered.get('start')} to "
                  f"{dataset_meta.time_period_covered.get('end')}") if dataset_meta.time_period_covered else "N/A"

    prompt = DATASET_PROMPT_TEMPLATE.replace("{{domain}}", dataset_meta.domain_covered or "Unspecified")\
                                    .replace("{{table_list}}", ", ".join(dataset_meta.list_of_tables))\
                                    .replace("{{table_row_counts}}", str(table_counts))\
                                    .replace("{{time_range}}", time_range)

    # Pass the kwargs through here
    description = llm_function(prompt, model, **kwargs)
    return description or f"Dataset containing {len(dataset_meta.list_of_tables)} tables covering {dataset_meta.domain_covered or 'unspecified domain'}."


def generate_table_description(table_meta: TableMeta, dataset_name: str, dataset_description: str, dataset_domain: str, model: str, llm_function, **kwargs) -> str:
    col_list_types = ", ".join([f"{col_name} ({meta.data_type})" for col_name, meta in table_meta.columns.items()])
    sample_rows = _format_sample_rows(table_meta)

    prompt = TABLE_PROMPT_TEMPLATE.replace("{{dataset_name}}", dataset_name)\
                                  .replace("{{dataset_domain}}", dataset_domain)\
                                  .replace("{{dataset_description}}", dataset_description)\
                                  .replace("{{table_name}}", table_meta.table_name)\
                                  .replace("{{column_list_with_types}}", col_list_types)\
                                  .replace("{{n_rows}}", str(table_meta.row_count))\
                                  .replace("{{sample_rows}}", sample_rows)

    # Pass the kwargs through here
    description = llm_function(prompt, model, **kwargs)
    return description or f"Table {table_meta.table_name} with {table_meta.number_of_columns} columns and {table_meta.row_count} rows."


def generate_single_column_description(column_meta: ColumnMeta, table_name: str, table_description: str, dataset_name: str, dataset_description: str, dataset_domain: str, model: str, llm_function, **kwargs) -> str:
    null_percent = _calculate_null_percent(column_meta)
    stats_summary = _format_column_stats(column_meta)
    samples_str = ", ".join([str(s) for s in column_meta.sample_values]) if column_meta.sample_values else "N/A"

    type_str = column_meta.semantic_type if column_meta.semantic_type else column_meta.data_type

    prompt = COLUMN_PROMPT_TEMPLATE.replace("{{dataset_name}}", dataset_name)\
                                   .replace("{{dataset_domain}}", dataset_domain)\
                                   .replace("{{dataset_description}}", dataset_description)\
                                   .replace("{{table_name}}", table_name)\
                                   .replace("{{table_description}}", table_description)\
                                   .replace("{{column_name}}", column_meta.column_name)\
                                   .replace("{{type}}", type_str)\
                                   .replace("{{samples}}", samples_str)\
                                   .replace("{{unique}}", str(column_meta.unique_count))\
                                   .replace("{{null_percent}}", null_percent)\
                                   .replace("{{stats_summary}}", stats_summary)

    # Pass the kwargs through here
    response = llm_function(prompt, model, **kwargs)
    return response if response else f"Column of type {type_str}."

def _post_process_response(response_text: str) -> str:
    """Cleans LLM response and strictly enforces word count limit."""
    if not response_text:
        return ""
        
    clean_response = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    clean_response = re.sub(r"```.*?```", "", clean_response, flags=re.DOTALL)
    clean_response = re.sub(r"[*_#>`\"']", "", clean_response)
    clean_response = re.sub(r"\s+", " ", clean_response).strip()
    words = clean_response.split()
    if len(words) > 15:
        clean_response = " ".join(words[:15])
        
    return clean_response

COST_TRACKER = {
    "total_cost": 0.0,
    "models": {}
}

def update_cost(model: str, cost: float):
    COST_TRACKER["total_cost"] += cost
    if model not in COST_TRACKER["models"]:
        COST_TRACKER["models"][model] = {"calls": 0, "cost": 0.0}
    COST_TRACKER["models"][model]["calls"] += 1
    COST_TRACKER["models"][model]["cost"] += cost

def get_cost_summary() -> dict:
    """
    Returns the entire cost summary.
    """
    return COST_TRACKER

def _run_openai_model(prompt: str, model: str, system_message: str = None, temperature: float = 0.1) -> Optional[str]:
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY not set")

        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Use provided system message OR fallback to description prompt
        default_sys = ("You are a precise metadata expert. "
                       "Write extremely concise, factual descriptions (max 15 words).")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message or default_sys},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_completion_tokens=1000 if system_message else 50,
        )

        response_text = response.choices[0].message.content.strip()

        # Token usage
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
        else:
            prompt_tokens = completion_tokens = 0

        # Pricing table (per 1k tokens)
        model_costs = {
            "gpt-5.5": (0.005, 0.03),
            "gpt-5.4": (0.0025, 0.015),
            "gpt-5.4-mini": (0.00075, 0.0045),
            "gpt-5": (0.00125, 0.01),
            "gpt-5-mini": (0.00025, 0.002),
            "gpt-5-nano": (0.00005, 0.0004),
            "gpt-5-pro": (0.015, 0.12),
            "gpt-5.5-pro": (0.03, 0.18),
            "gpt-4.1": (0.002, 0.008),
            "gpt-4o": (0.0025, 0.01),
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-4": (0.03, 0.06),
            "gpt-3.5-turbo": (0.0005, 0.0015),
        }
        prompt_cost_per_1k, completion_cost_per_1k = model_costs.get(model, (0.005, 0.015))

        # Cost of this single call
        cost = (
            (prompt_tokens / 1000) * prompt_cost_per_1k +
            (completion_tokens / 1000) * completion_cost_per_1k
        )

        # Incremental cost update
        update_cost(model, cost)

        logger.info(
            f"[COST] Model={model} | Prompt={prompt_tokens} | Completion={completion_tokens} "
            f"| This call=${cost:.6f} | Total=${COST_TRACKER['total_cost']:.6f}"
        )

        return response_text

    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return None
    
def _run_gemini_model(prompt: str, model: str, system_message: str = None) -> Optional[str]:
    try:
        api_key = os.environ.get("GEMINI_API_COST_KEY")
        if not api_key: raise ValueError("GEMINI_API_KEY not set")

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Pass system_instruction here
        gem_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_message
        )
        
        response = gem_model.generate_content(prompt)
        return response.text.strip() if hasattr(response, "text") else ""
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return None

def run_ollama_model(prompt: str, model: str, system_message: str = None, temperature: float = 0.1) -> Optional[str]:
    """
    Call local Ollama API for concise descriptions or JSON relationships.
    """
    API_ENDPOINT = f"{OLLAMA_URL}/api/generate"
    try:
        num_predict = 1000 if system_message else 50
        
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_message,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict
            }
        }

        response = requests.post(API_ENDPOINT, json=payload)
        response.raise_for_status()
        
        response_data = response.json()
        raw_text = response_data.get("response", "").strip()
        return raw_text

    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API call failed for model {model}: {e}")
        return None


def run_llm_dispatcher(prompt: str, model: str, system_message: str = None, is_json: bool = False, temperature=0.1) -> Optional[str]:
    model_lower = model.lower()
    
    if model_lower in [m.lower() for m in OPENAI_MODELS] or model_lower.startswith("gpt-"):
        response_text = _run_openai_model(prompt, model, system_message, temperature)
    elif model_lower.startswith("gemini"):
        response_text = _run_gemini_model(prompt, model, system_message)
    else:
        response_text = run_ollama_model(prompt, model, system_message, temperature)

    if not response_text:
        return None

    if is_json:
        return response_text
    
    return _post_process_response(response_text)

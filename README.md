# DataAtlas: Automatic Generation of Data Dictionaries

DataAtlas is a Python package to automatically generate structured data dictionaries and schema insights from tabular datasets using large language models (LLMs).

It supports CSV, Excel, and JSON datasets, and produces rich metadata including column descriptions, table summaries, and inferred relationships.

---

## Features

### Metadata Generation

- **Column Descriptions**: AI-generated semantic meaning for each field
- **Table Summaries**: Context-aware descriptions of each table
- **Smart Sampling**: Automatic data type detection with representative values
- **Pattern Recognition**: Identification of IDs, timestamps, clinical codes, etc.

---

### Relationship Inference

- Automatic detection of relationships across tables
- Confidence scoring based on name similarity and value overlap
- Cardinality estimation (1:1, 1:N, N:M)
- Deduplication of redundant relationships

---

### Schema Visualization

- Generates **Mermaid ER diagrams** (`.mmd`)
- No system dependencies (no Graphviz required)
- Compatible with GitHub and https://mermaid.live/

---

### Multi-format Outputs

- `dictionary.json` → full structured metadata
- `dictionary_summary.md` → human-readable documentation
- `relationships.json` → structured relationships
- `relationships_summary.md` → readable relationships
- `schema_diagram.mmd` → ER diagram

---

### Flexible LLM Support

Supports multiple backends through a unified interface:

- OpenAI models (e.g., GPT-4o, GPT-4o-mini)
- Google Gemini models
- Local models via **Ollama** (recommended for privacy)

---

## Installation

```bash
pip install data-atlas
```

---

## LLM Setup (Important)

### Option 1 — Local models (recommended)

Install :contentReference[oaicite:0]{index=0}:

```bash
# install ollama (see official website)
ollama pull llama3.1
```

---

### Option 2 — API-based models

Set your API keys for:
- OpenAI
- Google Gemini

---

## Usage

### Command Line (main usage)

After installation, a command is automatically available:

```bash
generate-dictionary <folder_path> --output-dir <output_dir>
```

### Example

```bash
generate-dictionary data/MIMIC --output-dir output --model llama3.1
```

---

### ⚙️ Parameters

- `<folder_path>`: Folder containing dataset files (CSV, Excel, JSON)
- `--output-dir`: Output directory
- `--model`: LLM model (default: `llama3.1`)
- `--domain`: Optional domain (e.g., `clinical`)
- `--no-relationships`: Disable relationship inference
- `--min-overlap`: Minimum overlap % (default: 80)
- `--min-confidence`: Minimum confidence (default: 0.7)

---

## Alternative Execution (if CLI not found)

If `generate-dictionary` is not recognized, run:

```bash
python -m data_dictionary_generator.generate_dictionary <folder_path> --output-dir <output_dir>
```

This avoids PATH issues.

---

## Output Structure

After execution, the output directory will contain:

### Data Dictionary

- `dictionary.json`
- `dictionary_summary.md`

### Relationships

- `relationships.json`
- `relationships_summary.md`

### Schema

- `schema_diagram.mmd`

---

## Example Output

### Table-level

- Table name
- Number of rows
- Table description

### Column-level

- Column name
- Data type
- Sample values
- AI-generated description
- Null percentage

### Relationships

- Source and target columns
- Relationship type (1:1, 1:N, N:M)
- Confidence score

---

## Visualization

To visualize schema diagrams:

1. Open https://mermaid.live/
2. Paste the `.mmd` file content
3. Export as PNG or SVG

---

## Design Principles

- **Augmentation, not replacement**: supports expert workflows
- **Scalable**: handles large datasets via sampling
- **Portable**: no heavy system dependencies
- **Privacy-aware**: supports fully local execution

---

## Troubleshooting

### 1. Command not found

If this fails:

```bash
generate-dictionary
```

Try:

```bash
python -m data_dictionary_generator.generate_dictionary ...
```

Or ensure your Python `Scripts/` folder is in PATH.

---

### 2. Model errors

#### Ollama

```bash
ollama pull llama3.1
```

#### API models

Ensure API keys are set correctly.

---

### 3. Missing dependencies

Reinstall:

```bash
pip install --upgrade data-atlas
```

---

## Contributing

1. Fork the repository  
2. Create a branch  
3. Commit changes  
4. Push  
5. Open a pull request  

---

## License

MIT License — see `LICENSE` file.

---

## Author

Raffaele Giancotti

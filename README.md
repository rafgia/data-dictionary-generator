# DataAtlas: Automatic Generation of Data Dictionaries

A Python package to automatically generate data dictionaries for clinical datasets using large language models (LLMs). This tool takes in dataset files (CSV, Excel, or JSON format), processes them, and generates descriptions for each column in the dataset, as well as other metadata like data types, sample data, and table descriptions.

---

## Features

### Metadata Generation

* **Column Descriptions**: AI-generated clinical context for each field
* **Table Summaries**: Dataset-level documentation
* **Smart Sampling**: Automatic data type detection with sample values
* **Pattern Recognition**: Structural pattern analysis to identify identifiers, clinical codes, timestamps, and other semantically meaningful data types

### Relationship Inference

* **Automatic Detection** of relationships across tables
* **Confidence Scoring** based on name matching and value overlap
* **Cardinality Estimation** (1:1, 1:N, N:M)
* **Deduplication** of redundant relationships

### Schema Visualization

* **Mermaid ER Diagrams** (text-based, portable, no dependencies)
* Easily visualizable via: https://mermaid.live/
* Compatible with GitHub Markdown and documentation tools

### Multi-Format Outputs

* JSON (machine-readable)
* Markdown (human-readable)
* Mermaid diagrams (`.mmd`) for schema visualization

### Flexible LLM Integration

Supports multiple model backends through a unified dispatcher interface:

* **OpenAI**: GPT-4, GPT-4o-mini, and other OpenAI models
* **Google Gemini**: Gemini-2.5-Flash and other Gemini models
* **Local models via Ollama**: Llama 3.1 and any other Ollama-compatible model

---

## Requirements

Make sure you have Python 3.8+ installed. All dependencies are automatically installed when you install the package.

---

## Installation

```bash
pip install data-atlas
```

---

## Usage

### Command-line Interface

Once the package is installed, you can use the command line to generate metadata and schema information for your dataset(s).

```bash
python generate_dictionary.py <folder_path> --output-dir <output_dir> --model <model_name>
```

---

### Parameters

* `<folder_path>`: Path to the folder containing dataset files (CSV, Excel, or JSON)
* `<output-dir>`: Directory where outputs will be saved
* `--model`: Model to use (e.g., `llama3.1`, `gpt-4o-mini`, `gemini-2.5-flash`)
* `--domain`: Optional domain tag (e.g., "clinical")
* `--no-relationships`: Disable relationship inference
* `--min-overlap`: Minimum value overlap (%) for relationship detection (default: 70)
* `--min-confidence`: Minimum confidence threshold (default: 0.5)

---

## Example

1. **Prepare your data files**
   Place dataset files in a folder (e.g., `data/MIMIC`)

2. **Run the generator**

```bash
python -m generate_dictionary.py data/MIMIC --output-dir output --model gpt-4o-mini
```

---

## Outputs

### Data Dictionary

* `dictionary.json`: Full structured metadata
* `dictionary_summary.md`: Human-readable documentation

### Relationships

* `relationships.txt`: Summary of inferred relationships
* `relationships.json`: Structured relationship data

### Schema Diagram

* `schema_diagram.mmd`: Mermaid ER diagram

👉 To visualize:

* Open https://mermaid.live/
* Paste the `.mmd` content
* Export as PNG or SVG

---

## Sample Output

For each **table**:

* Table name
* Number of rows and columns
* Table description

For each **column**:

* Column name
* Sample data
* Data type
* AI-generated description

For **relationships**:

* Source and target columns
* Relationship type (1:1, 1:N, N:M)
* Confidence score
* Inference method

---

## Design Principles

* **Augmentation, not replacement**: DataAtlas is intended to support expert curation
* **Scalable**: Uses sampling and chunk-based processing for large datasets
* **Portable**: No system dependencies (Mermaid diagrams instead of Graphviz)
* **Privacy-aware**: Supports fully local deployment

---

## Troubleshooting

### 1. Model not found errors

* **Ollama**: Ensure the model is available locally

  ```bash
  ollama run llama3.1
  ```
* **OpenAI / Gemini**: Ensure API keys are correctly set

---

### 2. Diagram visualization

Mermaid diagrams are saved as `.mmd` files.

To view:

* Go to https://mermaid.live/
* Paste the file content

---

### 3. Dependency issues

Ensure:

* Python version is correct
* Dependencies in `pyproject.toml` are installed

---

## Contributing

If you would like to contribute:

1. Fork the repository
2. Create a branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push (`git push origin feature-name`)
5. Open a pull request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

Raffaele Giancotti


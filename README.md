# Data Dictionary Generator

A Python package to automatically generate data dictionaries for clinical datasets using a large language model (LLM) via Ollama. This tool takes in dataset files (CSV format), processes them, and generates descriptions for each column in the dataset, as well as other metadata like data types, sample data, table descriptions and some quality information such as missing values, outliers and redundant values. It is also able to find relationships between tables and columns.

## Features
### Metadata Generation
- **Column Descriptions**: AI-generated clinical context for each field
- **Table Summaries**: Dataset-level documentation
- **Smart Sampling**: Automatic data type detection with sample values

### Quality Analysis
- Missing value statistics
- Outlier detection (numeric fields)
- Duplicate row/column identification

### Advanced Capabilities
- **Semantic Relationship Detection**: Finds connected columns across tables
- **Multi-Format Outputs**: CSV, JSON, Markdown, and PDF
- **Custom LLM Integration**: Supports any Ollama model

## Requirements

Make sure you have Python 3.8+ installed, along with the following dependencies:

- **pandas**: For handling and processing the dataset.
- **requests**: For HTTP requests (if needed).
- **ollama**: For generating metadata descriptions using Ollama's LLM.
- **torch**: PyTorch for deep learning operations (used with Ollama).
- **transformers**: Hugging Face Transformers library, if you're using other LLMs.

Install dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/rafgia/data-dictionary-generator.git
cd data-dictionary-generator
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Command-line Interface

Once the package is installed, you can use the command line to generate metadata for your dataset(s).

To run the tool, use the following command:

```bash
python cli.py <folder_path> --dataset-name <dataset_name> --output-dir <output_dir> --model <ollama_model> --format <format>
```

#### Parameters:
- `<folder_path>`: The path to the folder containing your CSV files.
- `<dataset-name>`: The name of your dataset (e.g., `MIMIC`).
- `<output-dir>`: The name of the path where the metadata will be saved.
- `--model`: (Optional) Specify the Ollama model to use for generating metadata (default is `deepseek-r1:1.5b`).
- `--format`: where <format> can be one of csv, json, pdf, markdown (default is csv).

### Example

1. **Prepare your data files**:
   Place all your CSV files (representing tables in your dataset) in a folder, e.g., `data/MIMIC`.

2. **Run the generator**:

   ```bash
   python cli.py data/MIMIC --dataset-name MIMIC --output-dir output --model deepseek-r1:1.5b --format csv
   ```

This will generate metadata for each column in the dataset and save it to a CSV file (`metadata_output.csv`).

### Sample Output

For each **table** in your dataset, the following metadata will be generated:
- **Table Name**: The name of the table (CSV filename).
- **Number of Rows**: The total number of rows in the table.
- **Number of Columns**: The total number of columns in the table.
- **Table Description**: A generated description of what the table contains.

For each **column** in your dataset:
- **Column Name**: The name of the column.
- **Sample Data**: A sample of 5 data points from the column.
- **Data Type**: The inferred data type (e.g., integer, float, string).
- **Column Description**: A description generated by the model for the column.

### Example of generated metadata:

# Sample data dictionary output

## Table metadata

| table_name | dataset_name | table_description                     | number_of_rows | number_of_columns |
|------------|--------------|---------------------------------------|----------------|-------------------|
| patients   | MIMIC-IV     | Contains core patient demographics    | 50,000         | 12                |
| lab_results| MIMIC-IV     | Laboratory test measurements          | 250,000        | 8                 |

## Column metadata

| table_name | column_name    | datatype | sample_data                  | column_description                          |
|------------|----------------|----------|------------------------------|---------------------------------------------|
| patients   | patient_id     | integer  | [1001, 1002, 1003]           | Unique hospital patient identifier          |
| patients   | gender         | string   | ["M", "F", "M"]              | Biological sex (M/F)                        |
| patients   | age            | integer  | [45, 72, 38]                 | Patient age at admission                    |
| lab_results| test_name      | string   | ["WBC", "HbA1c", "Creatinine"]| Laboratory test performed                   |
| lab_results| result_value   | float    | [12.5, 6.2, 0.9]             | Numeric result of the laboratory test       |

## Data quality report

| table_name | column_name    | missing_percentage | has_outliers | num_duplicates |
|------------|----------------|--------------------|--------------|----------------|
| patients   | patient_id     | 0.0%               | No           | 0              |
| patients   | gender         | 0.2%               | No           | 0              |
| lab_results| test_name      | 0.0%               | No           | 0              |
| lab_results| result_value   | 1.8%               | Yes          | 12             |

## Table relationships

### patients.patient_id ↔ admissions.patient_id
- **Confidence Score**: 0.98
- **Relationship Type**: One-to-Many
- **Semantic Similarity**: 0.95
- **Mutual Information**: 0.97

### lab_results.test_name ↔ procedures.procedure_name
- **Confidence Score**: 0.82
- **Relationship Type**: Many-to-Many
- **Semantic Similarity**: 0.78
- **Mutual Information**: 0.85

## Sample markdown output structure

output/
├── patients_metadata.md
├── lab_results_metadata.md
├── data_quality_report.md
└── relationships.md

## Troubleshooting

### 1. If you encounter an error such as `model not found`, make sure you have set up Ollama correctly and the model is available.
   - Ensure that you can manually run the model using `ollama run deepseek-r1:1.5b` from the command line before using it in the Python script.

### 2. If the dependencies are not installing, make sure you're using the correct Python version and have all required libraries listed in `requirements.txt`.

### 3. If the dataset is very large, consider breaking it down into smaller CSV files for more efficient processing.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request. Make sure to add tests and document any new features.

### To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

Raffaele Giancotti

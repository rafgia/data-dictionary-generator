import pandas as pd
from data_dictionary_generator.generator import (
    run_ollama_model,
    generate_metadata_for_table,
)


def test_run_ollama_model():
    """
    Test the Ollama model integration. This test should be adapted based on your model's real response.
    """
    prompt = "What is the result of 2+2?"
    response = run_ollama_model(prompt)
    assert response is not None
    assert "4" in response


def test_generate_metadata_for_table():
    """
    Test generating metadata for a given table.
    """
    df = pd.DataFrame({"column1": [1, 2, 3], "column2": ["a", "b", "c"]})
    table_name = "test_table"
    dataset_name = "test_dataset"
    metadata = generate_metadata_for_table(df, table_name, dataset_name)

    assert isinstance(metadata, pd.DataFrame)
    assert len(metadata) == len(df.columns)
    assert metadata["table_name"].iloc[0] == table_name
    assert metadata["dataset_name"].iloc[0] == dataset_name

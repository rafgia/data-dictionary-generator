import pathlib
import pandas as pd
from typing import Union

def read_dataset(path: pathlib.Path) -> pd.DataFrame:
    """
    Attempt to load a table from a file or a dataset from a folder.
    Raises an exception if the file cannot be read, the format is unsupported,
    or the resulting DataFrame is empty.
    """
    suffix = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")

    try:
        if suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
        elif suffix in {".xls", ".xlsx"}:
            df = pd.read_excel(path)
        elif suffix == ".json":
            try:
                df = pd.read_json(path, orient="records", lines=True)
            except ValueError:
                df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
            
        if df.empty:
            raise ValueError(f"The dataset loaded from {path} is empty.")
            
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to read dataset from {path}: {e}") from e
import pathlib
import pandas as pd
from typing import Optional, Union


def _count_data_rows(path: pathlib.Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return max(count - 1, 0)  # minus the header row


def _downcast_low_cardinality_strings(
    df: pd.DataFrame,
    max_unique: int = 10_000,
    max_ratio: float = 0.1,
) -> pd.DataFrame:
    
    if len(df) == 0:
        return df

    for col in df.select_dtypes(include="object").columns:
        try:
            n_unique = df[col].nunique(dropna=False)
        except TypeError:
            # Unhashable values (rare for CSV/Excel/JSON-sourced columns,
            # but possible) - leave the column as-is rather than error out.
            continue

        if n_unique <= max_unique or (n_unique / len(df)) <= max_ratio:
            df[col] = df[col].astype("category")

    return df


def read_dataset(
    path: pathlib.Path,
    max_rows: Optional[int] = 20_000_000,
) -> pd.DataFrame:

    suffix = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")

    try:
        row_cap_hit = False
        total_rows_in_file = None

        if suffix == ".csv":
            if max_rows is not None:
                total_rows_in_file = _count_data_rows(path)
                row_cap_hit = total_rows_in_file > max_rows

            if row_cap_hit:
                df = pd.read_csv(path, low_memory=False, nrows=max_rows)
            else:
                try:
                    df = pd.read_csv(path, engine="pyarrow")
                except Exception:
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

        df = _downcast_low_cardinality_strings(df)

        df.attrs["is_row_capped"] = row_cap_hit
        df.attrs["rows_processed"] = len(df)
        df.attrs["total_rows_in_file"] = (
            total_rows_in_file if total_rows_in_file is not None else len(df)
        )

        if row_cap_hit:
            print(
                f"Warning: {path.name} has {total_rows_in_file:,} rows, more than "
                f"the {max_rows:,}-row cap; only the first {max_rows:,} rows were "
                f"read. row_count, null/unique counts, and time-period coverage "
                f"for this table reflect that partial read, not the full file."
            )

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to read dataset from {path}: {e}") from e

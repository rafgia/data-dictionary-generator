from click.testing import CliRunner
from data_dictionary_generator.cli import main
import pytest


def test_generate_dictionary_cli(tmp_path: pytest.TempPathFactory) -> None:
    """
    Test the CLI command for generating the dictionary.
    """
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()
    (test_data_dir / "patients.csv").write_text("patient_id,age,gender\n1,45,M\n2,32,F")

    output_dir = tmp_path / "output"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            str(test_data_dir),
            "--dataset-name",
            "test_dataset",
            "--output-dir",
            str(output_dir),
            "--format",
            "csv",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "patients_metadata.csv").exists()

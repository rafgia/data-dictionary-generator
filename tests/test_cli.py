from click.testing import CliRunner
from cli import generate_dictionary


def test_generate_dictionary_cli():
    """
    Test the CLI command for generating the dictionary.
    """
    runner = CliRunner()
    result = runner.invoke(
        generate_dictionary, ["tests/test_data", "test_dataset", "output.csv"]
    )
    assert result.exit_code == 0
    assert "Metadata generation complete" in result.output

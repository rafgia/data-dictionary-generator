from click.testing import CliRunner
from data_dictionary_generator.cli import main


def test_generate_dictionary_cli():
    """
    Test the CLI command for generating the dictionary.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["tests/test_data", "test_dataset", "output.csv"])
    print(result.output)
    assert result.exit_code == 0
    assert "Metadata generation complete" in result.output

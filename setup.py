from setuptools import setup, find_packages

setup(
    name="data_dictionary_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["click", "pandas"],
    entry_points={
        "console_scripts": [
            "generate-dictionary=data_dictionary_generator.cli:generate_dictionary",
        ],
    },
    author="Raffaele Giancotti",
    description="A package to generate data dictionaries using Ollama.",
    url="https://github.com/rafgia/data-dictionary-generator",
)

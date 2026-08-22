# Reads the Small World of Words dataset
# Acknowledgement: https://smallworldofwords.org/en/project/research

from pathlib import Path

from GenerateKeywordCards2.get_root_directory import get_root_directory


def get_swow_data() -> str:
    """
    Unzips and caches the English SWOW dataset file.

    Returns:
        str: Path to the cached dataset file.
    """
    swow_zip_path = "/run/local-cache/SWOW-EN18.zip"

    download_instructions = """In order to use this dataset, you must:
1. Visit https://smallworldofwords.org/en/project/research#SWOW-EN18
2. Click the link to download SWOW-EN18.zip
3. Complete the access form
4. Place the downloaded file at Development\\SoCloverAI\\SWOW-EN18.zip."""

    zip_file = Path(swow_zip_path)
    if not zip_file.exists():
        raise FileNotFoundError(
            f"SWOW dataset not found at {swow_zip_path}.\n{download_instructions}"
        )

    # TODO: Implement caching/unzipping

    return str(zip_file)

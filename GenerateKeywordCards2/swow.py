# Reads the Small World of Words dataset
# Acknowledgement: https://smallworldofwords.org/en/project/research

from pathlib import Path
from zipfile import ZipFile

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

    # Ensure zip file is mounted.
    zip_file = Path(swow_zip_path)
    if not zip_file.exists():
        raise FileNotFoundError(
            f"SWOW dataset not found at {swow_zip_path}.\n{download_instructions}"
        )

    # Unzip the complete data file to a cache directory
    cache_dir = get_root_directory() / "cache"
    complete_data_filename = "SWOW-EN.complete.20180827.csv"
    complete_data_path = cache_dir / complete_data_filename
    if complete_data_path.exists():
        print(f"Using cached SWOW data{complete_data_path}")
    else:
        print(
            f"Extracting {complete_data_filename} from {swow_zip_path} to {complete_data_path}"
        )
        with ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extract(complete_data_filename, cache_dir)
        assert complete_data_path.exists(), (
            f"Failed to extract {complete_data_filename} from {swow_zip_path}"
        )

    return str(complete_data_path)

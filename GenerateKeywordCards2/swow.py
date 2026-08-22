# Reads the Small World of Words dataset
# Acknowledgement: https://smallworldofwords.org/en/project/research

from collections import defaultdict
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from GenerateKeywordCards2.get_root_directory import get_root_directory


def get_swow_datafile() -> str:
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


def get_swow_rows() -> list[dict[str, str]]:
    """
    Reads the SWOW dataset and returns it as a list of dictionaries.

    Returns:
        list[dict[str, str]]: List of rows from the SWOW dataset.
    """
    datafile_path = get_swow_datafile()
    with open(datafile_path, newline="", encoding="utf-8") as csvfile:
        reader = DictReader(csvfile)
        data = [row for row in reader]
    return data


@dataclass(frozen=True)
class WordAssociations:
    # Count of the number of times words occurred as a response to this word as a cue.
    forward: dict[str, int]
    # Count of the number of times words were a cue that produced this word as a response.
    backward: dict[str, int]


class SWOWAssociations:
    def __init__(self) -> None:
        associations: defaultdict[str, WordAssociations] = defaultdict(
            lambda: WordAssociations(defaultdict(int), defaultdict(int))
        )
        rows = get_swow_rows()
        for row in rows:
            cue = row["cue"]

            responses = []
            non_response = "No more responses"
            if row["R1"] != non_response:
                responses.append(row["R1"])
            if row["R2"] != non_response:
                responses.append(row["R2"])
            if row["R3"] != non_response:
                responses.append(row["R3"])

            for response in responses:
                associations[cue].forward[response] += 1
                associations[response].backward[cue] += 1

        # sort associations alphabetically and, within forward/backward, by count descending
        self.associations: dict[str, WordAssociations] = {}
        for word, word_associations in associations.items():
            sorted_forward = dict(
                sorted(
                    word_associations.forward.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            sorted_backward = dict(
                sorted(
                    word_associations.backward.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            self.associations[word] = WordAssociations(sorted_forward, sorted_backward)

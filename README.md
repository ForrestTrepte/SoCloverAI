# SoCloverAI

Experiments to see how well an LLM can play the party game So Clover!

Which then transitioned into creating an expansion set of cards for the game.

# Setup

Install from VSCode using Dev Containers:
* Open a new vscode window
* Ctrl+Shift+P, Dev Containers: Clone Repository in Named Container Volume
  * enter this repository: https://github.com/ForrestTrepte/SoCloverAI.git
  * and name the volume that will contain your repository files
  * wait for container to build and connect

Alternatively, instead of using containers, it should also work to install and run locally by installing Python and uv.

# Developing

* Set vscode interpreter: Ctrl+Shift+P, Python: Select Interpreter > Enter interpreter path > /opt/venvs/SoCloverAI
* Create .env file with:
  * `OPENAI_API_KEY=sk-...` (required for embeddings-based features)
  * `ANTHROPIC_API_KEY=sk-ant-...` (required for LLM candidate generation and rating)
* Type checking (from terminal): `uv run mypy .`
* Run tests from vscode testing pane
  * Or from terminal: `uv run pytest`
* Open Jupyter notebook .ipynb files in vscode
  * Select kernel (upper right) > Select another kernel > Python environments > /opt/venvs/SoCloverAI
  * Sometimes the Python kernel seems to hang in vscode, particularly when restarting the kernel. Not sure if the is a vscode, jupyter, or python bug. When this happens, you can recover vis Ctrl+Shift+P > Developer: Reload Window.

# GenerateKeywords

Workflow for generating, rating, and selecting a set of expansion keywords for So Clover!
All scripts are run from inside the `GenerateKeywords/` directory.

## Data files

| File | Description |
|------|-------------|
| `CloverExistingKeywords.csv` | 880 keywords from the base game — used as an exclusion list |
| `candidates.csv` | Growing pool of candidate keywords (word, category, source, notes) |
| `candidates_rating.csv` | Same pool with a `human_rating` column (1–5) for manual review |
| `candidates_llm_ratings.csv` | Adds `llm_rating` and `llm_notes` columns from Claude |

## Pipeline

TODO: Consider whether we could combine this entire workflow into a single command. It would use a file in place and addditively fill in columns instead of generating a new file for each step. Would that be simpler and easier to use?

### 1. Generate candidates

Run any combination of these to build up `candidates.csv`:

```
uv run generate_candidates.py               # LLM brainstorm by category (uses Anthropic API)
uv run discover_via_embeddings.py           # Find novel+versatile words from 60k vocabulary
# or edit candidates.csv directly to add seed words
```

Both scripts deduplicate against the base game's 880 words and any existing candidates.
Run them multiple times safely — they only append new words.

### 2. Create / update the rating file

```
uv run create_rating_file.py
```

Creates `candidates_rating.csv` on first run. On subsequent runs, merges in any new candidates
from `candidates.csv` without overwriting existing human ratings.

### 3. Human rating

Open `candidates_rating.csv` and fill in the `human_rating` column for each word:

| Rating | Meaning |
|--------|---------|
| 1 | Veto — exclude |
| 2 | Weak |
| 3 | Ok |
| 4 | Good |
| 5 | Must include |

### 4. LLM rating

```
python rate_candidates.py
```

Claude rates each candidate 1–5 for fun and versatility, writing results to
`candidates_llm_ratings.csv`. Safe to re-run after adding new words — only unrated words
are sent to the API; existing ratings are preserved.

### 5. Compare ratings

```
python compare_ratings.py
```

Prints correlation statistics and highlights the biggest agreements and divergences between
human and LLM ratings. Useful for catching words you underrated or overrated.

### 6. Select shortlist

```
python select_shortlist.py
```

Weighted selection (~110 words) respecting vetoes (human=1), force-includes (human=5),
and minimum category representation. Produces `candidates_shortlist.csv`.

### 7. Cull near-duplicates

`cull_similar.py` — greedy similarity cull using embeddings to remove the most redundant
words from the shortlist, producing `candidates_final.csv`.

### 8. Generate cards *(see GenerateCards)*

Feed the final word list into `GenerateCards/generate_cards.py` to produce a printable PDF.

# GenerateCards

The GenerateCards folder contains code for generating a printable pdf with a supplied set of keywords on cards suitable for printing and playing with So Clover!

# GenerateClues

The GenerateClues folder experiments with multiple methods of using an LLM or word embeddings to generate clue words. A clue word is an attept to find a connection to a given pair of keywords.

## Evaluation rubric

Score:

0. No solid connection to either word
1. Solid connection to one word, but not the other
2. Solid connection to one word, tentative to the other
3. Strong connection to both words
4. Home run connection

Legal:

0. Invalid, not clever
1. Questionably valid -or- clever
2. Valid

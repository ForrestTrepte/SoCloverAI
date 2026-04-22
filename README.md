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
* Create .env file with `OPENAI_API_KEY=sk-...`
* Type checking (from terminal): `uv run mypy .`
* Run tests from vscode testing pane
  * Or from terminal: `uv run pytest`
* Open Jupyter notebook .ipynb files in vscode
  * Select kernel (upper right) > Select another kernel > Python environments > /opt/venvs/SoCloverAI
  * Sometimes the Python kernel seems to hang in vscode, particularly when restarting the kernel. Not sure if the is a vscode, jupyter, or python bug. When this happens, you can recover vis Ctrl+Shift+P > Developer: Reload Window.

# GenerateCards

The GenerateCards folder contains code for generating a printable pdf with a supplied set of keywords on cards suitable for printing and playing with So Clover!

# GenerateKeywords

The GenerateKeywords folder contains a workflow for generating candidate keywords (manually, via LLM, or using embeddings), rating candidate keywords (manually via LLM), selecting a set of words, and allocating them to cards.

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

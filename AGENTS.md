# Agent Notes

## Fixing Security Vulnerabilities

### Overview

Dependencies are managed with `uv`. The source of truth is `pyproject.toml` and `uv.lock`.
Never use `pip install` directly to fix vulnerabilities — changes won't persist through a
container rebuild. All fixes must go through `pyproject.toml` + `uv sync`.

### Checking for vulnerabilities

Run the helper script (permitted in `.claude/settings.json`):

```
python3 GenerateKeywordCards2/pip_audit_with_urls.py
```

This prints one line per vulnerable package: `name  current_version -> fix_version(s)`.
The same file's `pip_audit_with_urls()` function renders a full table for the notebook.

### Always raise the floor to the fix version

When a scan flags a vulnerable version, bump the `pyproject.toml` floor to the fix version
itself (e.g. `"nltk>=3.10.2"`), even if the existing constraint is already loose enough to
technically permit it (e.g. `"nltk>=3.10.0"`). Don't stop at re-locking to get the vulnerable
version out of `uv.lock` — a lockfile-only fix leaves no record in the source of truth that
the old version is specifically disallowed, so a lockfile regeneration (or another tool
reading `pyproject.toml` directly) can silently land back on it. The constraint documents
*why* the floor is where it is; the lockfile just pins what's currently resolved. Edit the
constraint in `pyproject.toml`, then run `uv sync` to regenerate `uv.lock` from it — don't
hand-edit the lockfile as a substitute.

### Identifying direct vs. transitive dependencies

- **Direct deps** are listed in the `[project] dependencies` section of `pyproject.toml`.
  Add a version floor here: `"langchain>=1.3.9"`.
- **Transitive deps** (pulled in by direct deps) go in `[tool.uv] constraint-dependencies`.
  This tells uv to enforce a minimum version without making the package a direct dependency:
  `"langchain-core>=1.3.3"`.

Upgrading a direct dep often pulls its transitive deps up automatically. Check what's still
vulnerable after each `uv sync` before adding explicit transitive constraints.

### Cluster-based upgrade strategy

Group vulnerable packages by their root dependency and upgrade one cluster at a time.
This makes it easier to verify each step and revert if something breaks. Typical clusters
in this project:

1. **litellm** — upgrade first; it controls the openai version pin and gates langchain-openai
2. **LangChain ecosystem** — langchain, langchain-openai, and their transitive deps
   (langchain-core, langchain-classic, langgraph-*, langsmith)
3. **Jupyter ecosystem** — jupyterlab pulls jupyter-server, bleach, mistune, pillow,
   soupsieve, tornado
4. **litellm transitives** — aiohttp, click, msgpack (not dragged up by litellm itself)
5. **Standalones** — nltk, pydantic-settings, etc.

After each cluster: run the audit script and confirm only the expected packages were changed.

### Reverting a cluster upgrade

If `uv sync` was run with the wrong constraints, reverting `pyproject.toml` alone is not
enough — the lockfile already has the new versions pinned. To revert:

```
git restore uv.lock
uv sync
```

This resets to the committed lockfile state and re-applies only the current `pyproject.toml`
constraints, effectively replaying just the approved upgrades.

### Packages with no fix available

Some vulnerabilities have no fix version. Document them in the notebook's PIP Audit cell
with a comment explaining:
- The CVE/GHSA identifier
- Why the risk is acceptable (or not) for this project
- Any conditions that would change the risk assessment

Current known unfixable: `diskcache` (GHSA-w8v5-vhqr-4h9v, pickle deserialization).
See the comment in `RatingExperiments.ipynb` for the full risk assessment.

### Permission setup

The audit script is permitted in `.claude/settings.json`:

```json
"Bash(python3 GenerateKeywordCards2/pip_audit_with_urls.py)"
```

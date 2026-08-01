import html
import json
import subprocess

from IPython.display import Markdown, display
from tabulate import tabulate  # type: ignore


def get_advisory_url(vulnerability_id: str) -> str:
    if vulnerability_id.startswith("CVE-"):
        return f"https://www.cve.org/CVERecord?id={vulnerability_id}"
    if vulnerability_id.startswith("GHSA-"):
        return f"https://github.com/advisories/{vulnerability_id}"
    if vulnerability_id.startswith("PYSEC-"):
        return f"https://osv.dev/vulnerability/{vulnerability_id}"
    if vulnerability_id.startswith("BIT-"):
        return f"https://osv.dev/vulnerability/{vulnerability_id}"
    assert False, f"Unexpected vulnerability ID format: {vulnerability_id}"


def get_advisory_markdown_link(vulnerability_id: str) -> str:
    url = get_advisory_url(vulnerability_id)
    return f"[{vulnerability_id}]({url})"


def markdown_with_tooltip(text: str, tooltip: str) -> str:
    # Using HTML to create a tooltip in Markdown
    result = f'<span title="{html.escape(tooltip)}">{html.escape(text)}</span>'
    return result


def pip_audit_with_urls() -> None:
    """Helper function to display the information from pip-audit with clickable vulnerability advisory URLs"""
    cmd = [
        "pip-audit",
        "--format=json",
        "--aliases=on",
        "--desc=on",
        "--progress-spinner=off",
    ]
    json_result = subprocess.run(cmd, capture_output=True, text=True)
    if json_result.returncode not in [0, 1]:
        raise RuntimeError(
            f"pip-audit failed with exit code {json_result.returncode}:\n{json_result.stderr}"
        )
    dict_result = json.loads(json_result.stdout)

    rows = []
    for dependency in dict_result["dependencies"]:
        for vulnerability in dependency["vulns"]:
            aliases = [
                get_advisory_markdown_link(alias) for alias in vulnerability["aliases"]
            ]
            aliases_str = ", ".join(aliases) if aliases else None

            fix_versions = vulnerability["fix_versions"]
            fix_versions_str = ", ".join(fix_versions) if fix_versions else None

            maximum_description_length = 80
            full_description = vulnerability["description"]
            description_str = full_description
            if len(full_description) > maximum_description_length:
                description_str = markdown_with_tooltip(
                    full_description[:maximum_description_length] + "...",
                    full_description,
                )

            rows.append(
                [
                    dependency["name"],
                    get_advisory_markdown_link(vulnerability["id"]),
                    aliases_str,
                    dependency["version"],
                    fix_versions_str,
                    description_str,
                ]
            )

    table_md = tabulate(
        rows,
        headers=["Name", "ID", "Aliases", "Version", "Fix Versions", "Description"],
        tablefmt="pipe",  # markdown-compatible table format
    )
    display(Markdown(table_md))  # type: ignore

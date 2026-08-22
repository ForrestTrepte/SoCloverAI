import html
import json
import subprocess
from typing import Any

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
    all_dependencies = set()
    vulnerable_dependencies = set()
    for dependency in dict_result["dependencies"]:
        all_dependencies.add(dependency["name"])
        for vulnerability in dependency["vulns"]:
            vulnerable_dependencies.add(dependency["name"])
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
    print(
        f"Found {len(rows)} vulnerabilities in {len(vulnerable_dependencies)}/{len(all_dependencies)} dependencies."
    )
    display(Markdown(table_md))  # type: ignore


def pip_audit_summary() -> None:
    """Print a compact vulnerability summary for Claude's use: one line per package."""
    cmd = [
        "pip-audit",
        "--format=json",
        "--aliases=on",
        "--progress-spinner=off",
    ]
    json_result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if json_result.returncode not in [0, 1]:
        raise RuntimeError(
            f"pip-audit failed with exit code {json_result.returncode}:\n{json_result.stderr}"
        )
    dict_result = json.loads(json_result.stdout)

    pkgs: dict[str, dict[Any, Any]] = {}
    for dep in dict_result["dependencies"]:
        for vuln in dep["vulns"]:
            name = dep["name"]
            if name not in pkgs:
                pkgs[name] = {"version": dep["version"], "fixes": set()}
            for fv in vuln["fix_versions"]:
                pkgs[name]["fixes"].add(fv)

    if not pkgs:
        print("No vulnerabilities found.")
        return

    for name in sorted(pkgs):
        fixes = ", ".join(sorted(pkgs[name]["fixes"])) or "(no fix)"
        print(f"{name:25} {pkgs[name]['version']:10} -> {fixes}")


if __name__ == "__main__":
    pip_audit_summary()

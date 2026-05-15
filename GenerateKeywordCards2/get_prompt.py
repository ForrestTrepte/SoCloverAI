from GenerateKeywordCards2.get_root_directory import get_root_directory


def get_prompt(prompt_name: str, prompt_directory: str, args: dict[str, str]) -> str:
    prompt_file = (
        get_root_directory() / "prompts" / prompt_directory / f"{prompt_name}.md"
    )
    prompt_template = prompt_file.read_text()
    formatted_prompt = prompt_template.format(**args)
    return formatted_prompt

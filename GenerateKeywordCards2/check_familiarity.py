from get_prompt import get_prompt
from litellm import completion
from llm_metadata import LlmMetadata


def check_familiarity(model: str) -> tuple[str, LlmMetadata]:
    prompt = get_prompt("check_familiarity", "check_familiarity", {})
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="low",
    )
    return response.choices[0].message.content, LlmMetadata.from_response(response)

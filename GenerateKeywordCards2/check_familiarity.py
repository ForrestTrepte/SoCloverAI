from get_prompt import get_prompt
from llm import generate_async
from llm_metadata import LlmMetadata


async def check_familiarity_async(model: str) -> tuple[str, LlmMetadata]:
    prompt = get_prompt("check_familiarity", "check_familiarity", {})
    result, metadata = await generate_async(
        model=model,
        user_message=prompt,
        reasoning_effort="low",
        trial=0,
    )
    return result, metadata

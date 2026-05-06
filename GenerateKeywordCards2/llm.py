from typing import Literal

from litellm import acompletion
from llm_metadata import LlmMetadata


async def generate_async(
    model: str,
    user_message: str,
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "default"
    ],
    trial: int,
) -> tuple[str, LlmMetadata]:
    response = await acompletion(
        model=model,
        messages=[{"role": "user", "content": user_message}],
        reasoning_effort=reasoning_effort,
        # set user to trial number so requests from different trials will be treated separately in the cache
        user=f"trial_{trial}",
    )
    return response.choices[0].message.content, LlmMetadata.from_response(response)

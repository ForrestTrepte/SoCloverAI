from asyncio import Semaphore
from typing import Literal

from litellm import acompletion
from llm_metadata import LlmMetadata

try:
    import notes
except ImportError:
    pass


small_models = [
    "openai/gpt-5.4-mini-2026-03-17",
    "anthropic/claude-haiku-4-5-20251001",
    "gemini/gemini-3-flash-preview",
    "deepseek/deepseek-v4-flash",
]


maximum_concurrent_requests = 25
llm_semaphore = Semaphore(maximum_concurrent_requests)
log_llm_concurrency_was_in_use = False


def log_llm_concurrency() -> None:
    global log_llm_concurrency_was_in_use

    in_use = maximum_concurrent_requests - llm_semaphore._value
    if in_use <= 1 and not log_llm_concurrency_was_in_use:
        if in_use == 0:
            log_llm_concurrency_was_in_use = False
        return

    log_llm_concurrency_was_in_use = True
    waiting = len(llm_semaphore._waiters or [])
    print(f"LLMs {in_use} in use, {waiting} waiting")


async def generate_async(
    model: str,
    user_message: str,
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "default"
    ],
    trial: int,
) -> tuple[str, LlmMetadata]:
    async with llm_semaphore:
        log_llm_concurrency()
        # print(f"> acompletion {model}")
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": user_message}],
            reasoning_effort=reasoning_effort,
            # set user to trial number so requests from different trials will be treated separately in the cache
            user=f"trial_{trial}",
        )
        # print(f"< acompletion {model}")
    log_llm_concurrency()
    return response.choices[0].message.content, LlmMetadata.from_response(response)

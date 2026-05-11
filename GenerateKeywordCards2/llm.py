from asyncio import Semaphore
from typing import Literal, TypeVar

from litellm import acompletion
from pydantic import BaseModel

from .llm_metadata import LlmMetadata

try:
    from . import notes
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


async def generate_structured_async[T: BaseModel](
    model: str,
    system_message: str,
    user_message: str,
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "default"
    ],
    trial: int,
    response_format: type[T],
    response_format_fallback_description: str,
) -> tuple[T, LlmMetadata]:
    async with llm_semaphore:
        log_llm_concurrency()
        # print(f"> acompletion {model}")
        response_format_param: dict[str, str] | type[T]
        if model.startswith("deepseek/"):
            response_format_param = {"type": "json_object"}
            system_message += response_format_fallback_description
        else:
            response_format_param = response_format
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            reasoning_effort=reasoning_effort,
            # set user to trial number so requests from different trials will be treated separately in the cache
            user=f"trial_{trial}",
            response_format=response_format_param,
        )
        # print(f"< acompletion {model}")
    log_llm_concurrency()
    result = response_format.model_validate_json(response.choices[0].message.content)
    return result, LlmMetadata.from_response(response)

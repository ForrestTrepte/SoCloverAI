import asyncio
from asyncio import Semaphore
from contextlib import nullcontext
from collections.abc import Awaitable, Callable
from typing import Literal, TypeAlias, cast

from litellm import ModelResponse, acompletion
from litellm.exceptions import RateLimitError
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

maximum_concurrent_requests_anthropic = 3
llm_semaphore_anthropic = Semaphore(maximum_concurrent_requests_anthropic)

log_llm_concurrency_was_in_use = False


def log_llm_concurrency() -> None:
    global log_llm_concurrency_was_in_use

    in_use = maximum_concurrent_requests - llm_semaphore._value
    if in_use <= 1 and not log_llm_concurrency_was_in_use:
        if in_use == 0:
            log_llm_concurrency_was_in_use = False
        return

    log_llm_concurrency_was_in_use = True
    anthropic_in_use = (
        maximum_concurrent_requests_anthropic - llm_semaphore_anthropic._value
    )
    waiting = len(llm_semaphore._waiters or [])
    anthropic_waiting = len(llm_semaphore_anthropic._waiters or [])
    print(
        f"LLMs {in_use} in use, {waiting + anthropic_waiting} waiting (anthropic {anthropic_in_use} in use, {anthropic_waiting} waiting)"
    )


ReasoningEffort: TypeAlias = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "default"
]


_with_rate_limit_retry_id = 0


async def with_rate_limit_retry[R](
    request_info: str,
    op: Callable[[], Awaitable[R]],
    max_tries: int = 5,
    extra_time_seconds: float = 0.2,
) -> R:
    global _with_rate_limit_retry_id
    _with_rate_limit_retry_id += 1
    id = _with_rate_limit_retry_id

    for i in range(max_tries):
        try:
            return await op()
        except RateLimitError as e:
            now_str = f"{asyncio.get_running_loop().time():,.1f}"
            if i == max_tries - 1:
                print(
                    f"request {id} {now_str}s {request_info} rate limit error: giving up after {max_tries} tries"
                )
                raise

            assert hasattr(e, "litellm_response_headers")
            retry_after = float(e.litellm_response_headers.get("retry-after"))
            retry_after += extra_time_seconds
            print(
                f"request {id} {now_str}s {request_info} rate limit: retrying after {retry_after:.1f} seconds"
            )
            await asyncio.sleep(retry_after)

            now_str = f"{asyncio.get_running_loop().time():,.1f}"
            print(f"request {id} {now_str}s {request_info} rate limit: resuming")

    assert False


async def generate_async(
    model: str,
    user_message: str,
    reasoning_effort: ReasoningEffort,
    trial: int,
) -> tuple[str, LlmMetadata]:
    anthropic_lock = (
        llm_semaphore_anthropic if model.startswith("anthropic/") else nullcontext()
    )

    async def complete() -> ModelResponse:
        result = await acompletion(
            model=model,
            messages=[{"role": "user", "content": user_message}],
            reasoning_effort=reasoning_effort,
            # set user to trial number so requests from different trials will be treated separately in the cache
            user=f"trial_{trial}",
        )
        assert isinstance(result, ModelResponse)
        return result

    async with anthropic_lock:
        async with llm_semaphore:
            log_llm_concurrency()
            # print(f"> acompletion {model}")
            response = await with_rate_limit_retry(request_info=model, op=complete)
            # print(f"< acompletion {model}")

    log_llm_concurrency()
    content = response.choices[0].message.content
    assert content is not None
    return content, LlmMetadata.from_response(response)


async def generate_structured_async[T: BaseModel](
    model: str,
    system_message: str,
    user_message: str,
    reasoning_effort: ReasoningEffort,
    trial: int,
    response_format: type[T],
    response_format_fallback_description: str,
) -> tuple[T, LlmMetadata]:
    anthropic_lock = (
        llm_semaphore_anthropic if model.startswith("anthropic/") else nullcontext()
    )

    response_format_param: dict[str, str] | type[T]
    if model.startswith("deepseek/"):
        response_format_param = {"type": "json_object"}
        system_message += response_format_fallback_description
    else:
        response_format_param = response_format

    async def complete() -> ModelResponse:
        result = await acompletion(
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
        assert isinstance(result, ModelResponse)
        return result

    async with anthropic_lock:
        async with llm_semaphore:
            log_llm_concurrency()
            # print(f"> acompletion {model}")
            response = await with_rate_limit_retry(request_info=model, op=complete)
            # print(f"< acompletion {model}")
    log_llm_concurrency()
    content = response.choices[0].message.content
    assert content is not None
    result = response_format.model_validate_json(content)
    return result, LlmMetadata.from_response(response)

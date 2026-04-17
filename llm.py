import logging
import re
from typing import Any, List, Optional, cast

from langchain_core.globals import get_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import llm_cache_stats_wrapper
import simple_llm_cache
from init_openai import init_openai

logger = logging.getLogger("SoCloverAI")
init_openai()
model_name = "gpt-4-1106-preview"


def set_trial(trial: int) -> None:
    cache = cast(llm_cache_stats_wrapper.LlmCacheStatsWrapper, get_llm_cache())
    cast(simple_llm_cache.SimpleLlmCache, cache.inner_cache).set_trial(trial)


def dump_cache_stats_since_last_call() -> None:
    cache = cast(llm_cache_stats_wrapper.LlmCacheStatsWrapper, get_llm_cache())
    logger.info(cache.get_cache_stats_summary())
    cache.clear_cache_stats()


def create_llm_model(temperature: float, model_name: str) -> ChatOpenAI:
    result = ChatOpenAI(temperature=temperature, model_name=model_name)
    return result


async def predict(temperature: float, template: str, **kwargs: Any) -> List[str]:
    prompt = PromptTemplate(
        template=template.strip(), input_variables=["word0", "word1"]
    )
    llm = create_llm_model(temperature, model_name)
    chain = prompt | llm | StrOutputParser()
    output = await chain.ainvoke(kwargs)
    logger.debug(output)
    predictions = parse_candidates(output)
    best = parse_best(output)
    if best:
        predictions = [best] + predictions

    strip_chars = ' \t"'
    predictions = [prediction.strip(strip_chars) for prediction in predictions]
    predictions = [prediction for prediction in predictions if prediction]

    # remove duplicates while preserving order
    seen = set()
    unique_predictions = list()
    for prediction in predictions:
        if prediction not in seen:
            unique_predictions.append(prediction)
            seen.add(prediction)
    predictions = unique_predictions
    return predictions


def parse_candidates(output: str) -> List[str]:
    result = []
    for line in output.splitlines():
        if not line.startswith("Candidates:"):
            continue
        candidates_str = line[len("Candidates: "):]
        candidates = candidates_str.split(",")
        candidates = [candidate.strip() for candidate in candidates]
        result += candidates
    return result


pattern = re.compile(r"Best: (.*)")


def parse_best(output: str) -> Optional[str]:
    match = pattern.search(output)
    if match:
        return match.group(1)
    split_output = output.split()
    if len(split_output) == 1:
        logger.info(f"Invalid output format: {output}")
        return split_output[0]
    logger.info(f"Invalid output: {output}")
    return None

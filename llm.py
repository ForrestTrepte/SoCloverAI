import logging
import re
from typing import Any, List, Optional

import langchain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from init_openai import init_openai

logger = logging.getLogger("SoCloverAI")
init_openai()
model_name = "gpt-4-1106-preview"


def set_trial(trial: int) -> None:
    langchain.llm_cache.inner_cache.set_trial(trial)


def dump_cache_stats_since_last_call() -> None:
    logger.info(langchain.llm_cache.get_cache_stats_summary())
    langchain.llm_cache.clear_cache_stats()


def create_llm_model(temperature: float, model_name: str) -> ChatOpenAI:
    # mypy seems confused about the model_name parameter:
    #   Unexpected keyword argument "model_name" for "ChatOpenAI"
    result = ChatOpenAI(temperature=temperature, model_name=model_name)  # type: ignore
    return result


def predict(temperature: float, template: str, **kwargs: Any) -> List[str]:
    prompt = PromptTemplate(
        template=template.strip(), input_variables=["word0", "word1"]
    )
    llm = create_llm_model(temperature, model_name)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    output = chain.predict(**kwargs)
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
        candidates_str = line[len("Candidates: ") :]
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

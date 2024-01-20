import logging
import os
import re

import langchain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import llm_cache_stats_wrapper
import simple_llm_cache

logger = logging.getLogger("SoCloverAI")

# In order to make it easy to run work projects and personal AI experiments, override OPENAI_API_KEY with the value of OPENAI_API_KEY_PERSONAL if it is set.
if "OPENAI_API_KEY_PERSONAL" in os.environ:
    logger.info("Using key from OPENAI_API_KEY_PERSONAL environment variable")
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_PERSONAL"]

langchain.llm_cache = llm_cache_stats_wrapper.LlmCacheStatsWrapper(
    simple_llm_cache.SimpleLlmCache("llm-cache.json")
)

model_name = "gpt-4-1106-preview"


def set_trial(trial):
    langchain.llm_cache.inner_cache.set_trial(trial)


def dump_cache_stats_since_last_call():
    logger.info(langchain.llm_cache.get_cache_stats_summary())
    langchain.llm_cache.clear_cache_stats()


def predict(temperature, template, **kwargs):
    prompt = PromptTemplate(
        template=template.strip(), input_variables=["word0", "word1"]
    )
    llm = ChatOpenAI(temperature=temperature, model_name=model_name)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    output = chain.predict(**kwargs)
    logger.debug(output)
    candidates = parse_candidates(output)
    best = parse_best(output)
    try:
        candidates.remove(best)
    except ValueError:
        pass  # OK if candidates doesn't contain best
    result = [best] + candidates
    return result


def parse_candidates(output):
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


def parse_best(output):
    match = pattern.search(output)
    if match:
        return match.group(1)
    split_output = output.split()
    if len(split_output) == 1:
        logger.info(f"Invalid output format: {output}")
        return split_output[0]
    logger.info(f"Invalid output: {output}")
    return None

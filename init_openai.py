import logging
import os

import langchain

import llm_cache_stats_wrapper
import simple_llm_cache

logger = logging.getLogger("SoCloverAI")


def init_openai():
    # In order to make it easy to run work projects and personal AI experiments, override OPENAI_API_KEY with the value of OPENAI_API_KEY_PERSONAL if it is set.
    if "OPENAI_API_KEY_PERSONAL" in os.environ:
        logger.info("Using key from OPENAI_API_KEY_PERSONAL environment variable")
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_PERSONAL"]

    langchain.llm_cache = llm_cache_stats_wrapper.LlmCacheStatsWrapper(
        simple_llm_cache.SimpleLlmCache("llm-cache.json")
    )

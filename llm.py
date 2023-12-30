import langchain
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
import simple_llm_cache
import llm_cache_stats_wrapper
import os
import re

# In order to make it easy to run work projects and personal AI experiments, override OPENAI_API_KEY with the value of OPENAI_API_KEY_PERSONAL if it is set.
if "OPENAI_API_KEY_PERSONAL" in os.environ:
    print("Using key from OPENAI_API_KEY_PERSONAL environment variable")
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_PERSONAL"]

langchain.llm_cache = llm_cache_stats_wrapper.LlmCacheStatsWrapper(
    simple_llm_cache.SimpleLlmCache("llm-cache.json")
)

model_name = "gpt-4-1106-preview"


def set_trial(trial):
    langchain.llm_cache.inner_cache.set_trial(trial)


def dump_cache_stats_since_last_call():
    print(langchain.llm_cache.get_cache_stats_summary())
    langchain.llm_cache.clear_cache_stats()


def predict(temperature, template, **kwargs):
    prompt = PromptTemplate(
        template=template.strip(), input_variables=["word0", "word1"]
    )
    llm = ChatOpenAI(temperature=temperature, model_name=model_name)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    output = chain.predict(**kwargs)
    result = parse_output(output)
    return result


pattern = re.compile(r"Best: (.*)")


def parse_output(output):
    match = pattern.search(output)
    if match:
        return match.group(1)
    split_output = output.split()
    if len(split_output) == 1:
        print(f"Invalid output format: {output}")
        return split_output[0]
    print(f"Invalid output: {output}")
    return None

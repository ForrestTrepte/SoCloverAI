from litellm import completion
from litellm.cost_calculator import completion_cost

from get_prompt import get_prompt


def check_familiarity(model: str) -> tuple[str, float]:
    prompt = get_prompt("check_familiarity", "check_familiarity", {})
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    cost = completion_cost(response)
    return response.choices[0].message.content, cost

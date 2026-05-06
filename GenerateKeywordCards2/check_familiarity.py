from litellm import completion

from get_prompt import get_prompt


def check_familiarity(model: str) -> str:
    prompt = get_prompt("check_familiarity", "check_familiarity", {})
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

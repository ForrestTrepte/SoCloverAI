from evaluate import evaluate
from generate import generate

if __name__ == "__main__":
    results = generate()
    evaluate(results)

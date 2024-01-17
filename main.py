from evaluate import evaluate
import generate import generate_with_standard_settings, clear_logs

if __name__ == "__main__":
    clear_logs()
    results = generate_with_standard_settings()
    evaluate(results)

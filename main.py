import asyncio

from evaluate import evaluate
from generate import clear_logs, generate_with_standard_settings


async def main() -> None:
    clear_logs()
    results = await generate_with_standard_settings()
    evaluate(results)


if __name__ == "__main__":
    asyncio.run(main())

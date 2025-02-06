from pathlib import Path


def main() -> None:
    path = Path(__file__).parent.parent / ".fdua-competition/evaluation/src/evaluator.py"
    path.write_text(
        path.read_text()
        .replace("from openai import OpenAI", "from openai import AzureOpenAI")
        .replace("OpenAI()", "AzureOpenAI()")
    )


if __name__ == "__main__":
    main()

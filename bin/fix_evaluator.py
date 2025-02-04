from pathlib import Path


def main() -> None:
    path = Path(__file__).parent.parent / ".fdua-competition/evaluation/src/evaluator.py"

    path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "from dotenv import load_dotenv",
                "",
                "load_dotenv(Path(__file__).parent.parent.parent.parent / 'secrets/.env')",
                "",
                (
                    path.read_text()
                    .replace("from openai import OpenAI", "from openai import AzureOpenAI")
                    .replace("OpenAI()", "AzureOpenAI()")
                ),
            ]
        )
    )


if __name__ == "__main__":
    main()

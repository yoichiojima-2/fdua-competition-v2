from pathlib import Path


def main() -> None:
    path = Path(__file__).parent.parent / ".fdua-competition/evaluation/src/evaluator.py"
    original_code = path.read_text()
    modified_code = "\n".join(
        [
            "from pathlib import Path",
            "from dotenv import load_dotenv",
            "",
            "",
            "load_dotenv(Path(__file__).parent.parent.parent.parent / 'secrets/.env')",
            "",
            original_code.replace("OpenAI", "AzureOpenAI"),
        ]
    )

    path.write_text(modified_code)


if __name__ == "__main__":
    main()

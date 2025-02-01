import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from main import get_root


def main() -> None:
    path = get_root() / "evaluation/src/evaluator.py"
    original_code = path.read_text()
    modified_code = "\n".join(
        [
            "from pathlib import Path",
            "from dotenv import load_dotenv",
            "",
            "",
            "load_dotenv(Path(__file__).parent.parent.parent.parent / 'secrets/.env')",
            "",
            (
                original_code
                .replace("from openai import OpenAI", "from openai import AzureOpenAI")
                .replace("OpenAI()", "AzureOpenAI()")
            ),
        ]
    )
    path.write_text(modified_code)


if __name__ == "__main__":
    main()

import argparse
import sys
import warnings
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from langsmith import traceable
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from fdua_competition.chat import get_chat_model, get_prompt_template, invoke_chain_with_retry
from fdua_competition.enums import ChatModelOption, Mode, VectorStoreOption
from fdua_competition.parser import get_output_parser
from fdua_competition.utils import get_queries, write_result
from fdua_competition.vectorstore import build_context, build_vectorstore

load_dotenv("secrets/.env")


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser()
    opt = parser.add_argument
    opt("--output-name", "-o", type=str)
    opt("--mode", "-m", type=str, choices=[choice.value for choice in Mode], default=Mode.TEST.value)
    opt("--vectorstore", "-v", type=str, choices=[choice.value for choice in VectorStoreOption], default=VectorStoreOption.CHROMA.value)
    # fmt: on
    return parser.parse_args()


@traceable
def main(output_name: str, mode: Mode, vectorstore_option: VectorStoreOption) -> None:
    vectorstore = build_vectorstore(output_name, mode, vectorstore_option)

    prompt_template = get_prompt_template()
    chat_model = get_chat_model(ChatModelOption.AZURE)
    parser = get_output_parser()
    chain = prompt_template | chat_model | parser

    system_prompt = (Path(__file__).parent / "prompts/system_prompt.txt").read_text()

    responses = []
    for query in tqdm(get_queries(mode=mode), desc="querying.."):
        res = invoke_chain_with_retry(
            chain=chain,
            payload={
                "system_prompt": system_prompt,
                "query": query,
                "context": build_context(vectorstore=vectorstore, query=query),
                "language": "japanese",
            },
        )
        pprint(res)
        print()
        responses.append(res)

    write_result(output_name=output_name, responses=responses)
    print("[main] :)  done")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    main(output_name=args.output_name, mode=Mode(args.mode), vectorstore_option=VectorStoreOption(args.vectorstore))

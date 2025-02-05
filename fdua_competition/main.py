import argparse
import sys
import warnings
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from langsmith import traceable
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from fdua_competition.chat import get_chat_model, get_prompt_template, get_queries, invoke_chain_with_retry
from fdua_competition.enums import ChatModelOption, Mode, VectorStoreOption
from fdua_competition.parser import get_output_parser
from fdua_competition.utils import write_result
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

    system_prompt = " ".join(
        [
            "answer the question using only the provided context in {language}.",
            "return a json object that contains the following keys: query, response, reason, organization_name, contexts.",
            "make sure that the response field is under 54 tokens and contains no commas. other fields do not have token limits.",
            "do not include honorifics or polite expressions; use plain, assertive language in the response field.",
            "the response field must be based only from given context.",
            "do not include any special characters that can cause json parsing errors across all fields. this must be satisfied regardless of the language.",
        ]
    )

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

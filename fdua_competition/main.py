from argparse import ArgumentParser, Namespace

from fdua_competition.answer_query import answer_queries_concurrently
from fdua_competition.enums import EmbeddingOpt, Mode
from fdua_competition.logging_config import logger
from fdua_competition.models import create_embeddings
from fdua_competition.utils import read_queries, write_result
from fdua_competition.vectorstore import FduaVectorStore


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--mode", "-m", type=str, default=Mode.TEST.value, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings = create_embeddings(EmbeddingOpt.AZURE)
    vs = FduaVectorStore(mode=Mode(args.mode), embeddings=embeddings)
    queries = read_queries(Mode(args.mode))
    responses = answer_queries_concurrently(queries, vs, mode=Mode(args.mode))
    write_result(responses=responses, mode=Mode(args.mode))
    logger.info("[main] :)  done")


if __name__ == "__main__":
    main()

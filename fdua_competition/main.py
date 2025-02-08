from argparse import ArgumentParser

from tqdm import tqdm

from fdua_competition.answer_query import answer_query
from fdua_competition.enums import EmbeddingOpt
from fdua_competition.index_document import extract_organization_name
from fdua_competition.models import create_embeddings
from fdua_competition.utils import read_queries, write_result
from fdua_competition.vectorstore import FduaVectorStore
from fdua_competition.enums import Mode


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--output_name", "-o", type=str, required=True)
    opt("--mode", "-m", type=str, default=Mode.TEST.value, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    embeddings = create_embeddings(EmbeddingOpt.AZURE)
    vs = FduaVectorStore(args.output_name, embeddings)

    responses = []
    for query in tqdm(read_queries(Mode(args.mode))):
        response = answer_query(query=query, vectorstore=vs)
        responses.append(response)

    write_result(output_name=args.output_name, responses=responses)
    print("[main] :)  done")


if __name__ == "__main__":
    main()

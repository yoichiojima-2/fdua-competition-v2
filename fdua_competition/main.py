from argparse import ArgumentParser, Namespace

from tqdm import tqdm

from fdua_competition.answer_query import answer_query
from fdua_competition.enums import EmbeddingOpt, Mode
from fdua_competition.models import create_embeddings
from fdua_competition.utils import read_queries, write_result
from fdua_competition.vectorstore import FduaVectorStore
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--mode", "-m", type=str, default=Mode.TEST.value, required=True)
    return parser.parse_args()


def process_queries_concurrently(queries: list[str], vectorstore: FduaVectorStore) -> list[str]:
    responses = [None] * len(queries)
    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(answer_query, query=query, vectorstore=vectorstore): i for i, query in enumerate(queries)}
        for future in tqdm(as_completed(future_to_index), total=len(queries), desc="processing queries.."):
            index = future_to_index[future]
            try:
                response = future.result()
                print(response)
                responses[index] = response
            except Exception as exc:
                print(f"[process_queries_concurrently] query at index {index} generated an exception: {exc}")
    return responses


def main():
    args = parse_args()

    embeddings = create_embeddings(EmbeddingOpt.AZURE)
    vs = FduaVectorStore(embeddings)

    queries = read_queries(Mode(args.mode))
    responses = process_queries_concurrently(queries, vs)
    write_result(responses=responses)
    print("[main] :)  done")


if __name__ == "__main__":
    main()

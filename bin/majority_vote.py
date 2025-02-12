import os
import textwrap
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

from fdua_competition.enums import LogLevel
from fdua_competition.get_version import get_version
from fdua_competition.logging_config import logger, set_log_level
from fdua_competition.models import create_chat_model
from fdua_competition.utils import before_sleep_hook, dict_to_yaml


class MajorityVoteOutput(BaseModel):
    query: str = Field(description="Query text")
    candidates: list[str] = Field(description="[Candidate responses]")
    output: str = Field(description="Final answer selected by majority vote")
    reason: str = Field(description="Justification for the final answer")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def majority_vote(query: str, candidates: list[str]):
    logger.info(f"[majority_vote]\nquery: {query}\ncandidates: {candidates}\n")
    role = textwrap.dedent(
        """
        You are an advanced language model that has received several candidate responses, each generated using different parameter settings. Your task is to compare these responses and determine a final answer by taking a majority vote.

        Instructions:
        1. Review all provided candidate responses carefully.
        2. Identify the core information and key details in each response.
        3. Compare the responses to determine which answer, or elements thereof, appears most frequently.
        4. Select the final answer that best represents the consensus among the candidate responses.
        5. In case of ambiguity or a tie, choose the response that preserves the essential details most accurately.
        6. Provide the final answer along with a brief explanation of your majority vote decision, ensuring clarity and conciseness.
        7. Answer based only on the information provided in the candidate responses. Do not conduct additional research or use external sources.
        8. If there is only one candidate has a answer and else are わかりません, return that response as the final answer.
        9. outputは日本語で50文字いないに収めてください.

        Return only the final answer and the justification for your choice.
        """
    )

    chat_model = create_chat_model().with_structured_output(MajorityVoteOutput)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", role), ("user", "query: {query}"), ("user", "candidates: {candidates}")],
    )
    chain = prompt_template | chat_model

    res = chain.invoke({"query": query, "candidates": "\n".join([f"- {c}" for c in candidates])})
    logger.info(f"[majority_vote]\n{dict_to_yaml(res.model_dump())}\n")
    return res


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--log-level", "-l", type=str, default=LogLevel.INFO.value)
    return parser.parse_args()


def main():
    args = parse_args()
    set_log_level(LogLevel(args.log_level))

    candidates_dir = Path(os.environ["FDUA_DIR"]) / ".fdua-competition/candidates"
    csv_paths = list(candidates_dir.glob("*.csv"))

    if len(csv_paths) == 0:
        logger.error(f"no candidate CSV files found in {candidates_dir}")
        return

    candidates_df = pd.concat([pd.read_csv(csv, header=None)[1] for csv in csv_paths], axis=1)
    candidates_list = [row.to_list() for _, row in candidates_df.iterrows()]

    queries_df = pd.read_csv(Path(os.environ["FDUA_DIR"]) / ".fdua-competition/query.csv")
    queries_df.columns = ["index", "query"]
    queries = queries_df["query"].to_list()

    responses = [None] * len(queries)
    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(majority_vote, query, candidates): i
            for i, (query, candidates) in enumerate(zip(queries, candidates_list))
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="majority_vote"):
            idx = future_to_index[future]
            responses[idx] = future.result()

    df = pd.DataFrame([{"response": res.output} for res in responses])

    output_path = Path(os.environ["FDUA_DIR"]) / f".fdua-competition/majority_vote/v{get_version()}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, header=False)
    print(f"done: {output_path}")


if __name__ == "__main__":
    main()

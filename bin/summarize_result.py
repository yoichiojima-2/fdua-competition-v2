import os
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from fdua_competition.enums import Mode
from fdua_competition.get_version import get_version
from fdua_competition.utils import read_queries


def summarize():
    print(" score ".center(88, "="))
    print()
    calc_score()
    print(" detail ".center(88, "="))
    print()
    show_detail()


def calc_score():
    df = pd.read_csv(Path(os.environ["FDUA_DIR"]) / ".fdua-competition/evaluation/result/scoring.csv", header=None)
    df.columns = ["index", "evaluation", "crag_score"]

    evaluation_to_score = {"Perfect": 1, "Acceptable": 0.5, "Missing": 0, "Incorrect": -1}
    df["unit_score"] = df["evaluation"].apply(lambda x: evaluation_to_score[x])

    print(f"score: {df['unit_score'].mean()}\n")

    df = (
        df[["index", "evaluation", "unit_score"]]
        .groupby(["evaluation", "unit_score"])
        .count()
        .rename(columns={"index": "count"})
        .reset_index()
        .sort_values("unit_score", ascending=False)
    )
    print(tabulate(df, headers=df.columns, tablefmt="grid", showindex=False))
    print()


def show_detail():
    answer_df = pd.read_csv(Path(os.environ["FDUA_DIR"]) / ".fdua-competition/evaluation/data/ans_txt.csv", header=None)
    answer_df.columns = ["index", "answer"]

    output_df = pd.read_csv(Path(os.environ["FDUA_DIR"]) / f".fdua-competition/results/v{get_version()}.csv", header=None)
    output_df.columns = ["index", "output"]

    score_df = pd.read_csv(Path(os.environ["FDUA_DIR"]) / ".fdua-competition/evaluation/result/scoring.csv", header=None)
    score_df.columns = ["index", "evaluation", "score"]

    for question, ansewr, output, score in zip(
        read_queries(Mode.TEST), answer_df["answer"], output_df["output"], score_df["evaluation"]
    ):
        print(f"question: {question}")
        print(f"answer: {ansewr}")
        print(f"output: {output}")
        print(f"evaluation: {score}")
        print()


def main():
    summarize()


if __name__ == "__main__":
    main()

import os

import pandas as pd
from tabulate import tabulate

from fdua_competition.enums import Mode
from fdua_competition.utils import get_queries, get_root


def main():
    print(" score ".center(88, "="))
    print()
    calc_score()
    print(" detail ".center(88, "="))
    print()
    show_detail()


def calc_score():
    df = pd.read_csv(get_root() / "evaluation/result/scoring.csv", header=None)
    df.columns = ["index", "evaluation", "crag_score"]

    evaluation_to_score = {"Perfect": 1, "Acceptable": 0.5, "Missing": 0, "Incorrect": -1}
    df["unit_score"] = df["evaluation"].apply(lambda x: evaluation_to_score[x])

    print(f"score: {df['unit_score'].mean()}\n")

    df = (
        df[["index", "evaluation", "unit_score"]]
        .groupby(["evaluation", "unit_score"])
        .count()
        .reset_index()
        .sort_values("unit_score", ascending=False)
    )
    print(tabulate(df, headers=df.columns, tablefmt="grid", showindex=False))
    print()


def show_detail():
    answer_df = pd.read_csv(get_root() / "evaluation/data/ans_txt.csv", header=None)
    answer_df.columns = ["index", "answer"]

    output_df = pd.read_csv(get_root() / f"results/{os.getenv("OUTPUT_NAME")}.csv", header=None)
    output_df.columns = ["index", "output"]

    score_df = pd.read_csv(get_root() / "evaluation/result/scoring.csv", header=None)
    score_df.columns = ["index", "evaluation", "score"]

    for question, ansewr, output, score in zip(
        get_queries(Mode.TEST), answer_df["answer"], output_df["output"], score_df["evaluation"]
    ):
        print(f"question: {question}")
        print(f"answer: {ansewr}")
        print(f"output: {output}")
        print(f"evaluation: {score}")
        print()


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate

sys.path.append(str(Path(__file__).parent.parent))

from main import get_root


def main():
    print(" score ".center(88, "="))
    print()
    calc_score()
    print(" detail ".center(88, "="))
    print()
    show_detail()


def show_detail():
    answer_df = pd.read_csv(get_root() / "evaluation/data/ans_txt.csv", header=None)
    answer_df.columns = ["index", "answer"]

    output_df = pd.read_csv(get_root() / "results/output_simple_test.csv", header=None)
    output_df.columns = ["index", "output"]

    score_df = pd.read_csv(get_root() / "evaluation/result/scoring.csv", header=None)
    score_df.columns = ["index", "evaluation", "score"]

    for ansewr, output, score in zip(answer_df["answer"], output_df["output"], score_df["evaluation"]):
        print(f"answer: {ansewr}")
        print(f"output: {output}")
        print(f"evaluation: {score}")
        print()


def calc_score():
    df = pd.read_csv(get_root() / "evaluation/result/scoring.csv", header = None)
    df.columns = ["index", "evaluation", "score"]

    evaluation_to_score = {"Perfect": 1, "Acceptable": 0.5, "Missing": 0, "Incorrect": -1}
    converted_scores = df["evaluation"].map(lambda x: evaluation_to_score[x])

    print(f"score: {converted_scores.mean()}\n")
    print(tabulate(df.groupby("evaluation").count()))
    print()
    

if __name__ == "__main__":
    main()
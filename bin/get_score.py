import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from main import get_root


def main():
    df = pd.read_csv(get_root() / "evaluation/result/scoring.csv", header=0)

    evaluation_to_score = {"Perfect": 1, "Acceptable": 0.5, "Missing": 0, "Incorrect": -1}

    data = []
    for index, row in df.iterrows():
        data.append({"evaluation": row[1], "score": evaluation_to_score[row[1]]})

    score_df = pd.DataFrame(data)
    print(score_df["score"].mean())


if __name__ == "__main__":
    main()

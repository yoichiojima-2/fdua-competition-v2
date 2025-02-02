import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

sys.path.append(str(Path(__file__).parent.parent))
from main import get_root


def main():
    df = (
        pd.read_csv(get_root() / "evaluation/data/ans_txt.csv")
        .merge(pd.read_csv(get_root() / "results/output_simple_test.csv"))
        .merge(pd.read_csv(get_root() / "evaluation/result/scoring.csv"))
    )

    print(tabulate(df))


main()

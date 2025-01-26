from pathlib import Path

import pandas as pd


def main():
    output_md = Path(__file__).parent.parent / "docs/query.md"

    with output_md.open("w") as f:
        f.write("# Query\n\n")

        df = pd.read_csv(Path().home() / ".fdua-competition/downloads/query.csv")
        for i in df["problem"]:
            print(i)
            f.write(f"- {i}\n")

    print(f"[{Path(__file__).stem}] done: {output_md}")


main()

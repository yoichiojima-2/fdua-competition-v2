"""
example:
    uv run python fdua_competition/main.py -o example -m test -v chroma
"""


import argparse
import warnings
from pprint import pprint

from langsmith import traceable
from tqdm import tqdm

from fdua_competition.enums import Mode, VectorStoreOption
from fdua_competition.rag import ResearchAssistant
from fdua_competition.utils import get_queries, write_result
from fdua_competition.vectorstore import build_vectorstore


def parse_args() -> argparse.Namespace:
    """
    returns:
        argparse.Namespace
    """
    # fmt: off
    parser = argparse.ArgumentParser()
    opt = parser.add_argument
    opt(
        "--output-name",
        "-o",
        type=str,
        help="出力ファイル名の指定"
    )
    opt(
        "--mode",
        "-m",
        type=str,
        help="動作モードの指定",
        choices=[choice.value for choice in Mode],
        default=Mode.TEST.value
    )
    opt(
        "--vectorstore",
        "-v",
        help="使用するvectorstoreの指定",
        type=str,
        choices=[choice.value for choice in VectorStoreOption],
        default=VectorStoreOption.CHROMA.value
    )
    # fmt: on
    return parser.parse_args()


@traceable
def main(output_name: str, mode: Mode, vectorstore_option: VectorStoreOption) -> None:
    """
    vectorstoreを構築し, 各クエリに対してRAGに回答させ, 結果をCSVファイルに保存
    args:
        output_name (str): 出力ファイル名（拡張子なし）
        mode (Mode): 動作モード (TEST または SUBMIT)
        vectorstore_option (VectorStoreOption): 使用するvectorstoreのオプション
    """
    # vectorstoreを構築
    vectorstore = build_vectorstore(output_name, mode, vectorstore_option)

    # ResearchAssistant のインスタンスを生成
    research_assistant = ResearchAssistant(vectorstore=vectorstore)

    responses = []

    # 各クエリに対して処理を実行（tqdm で進捗を表示）
    for query in tqdm(get_queries(mode=mode), desc="querying.."):
        res = research_assistant.invoke(query)
        pprint(res)
        print()
        responses.append(res)

    # 結果を CSV ファイルとして書き出す
    write_result(output_name=output_name, responses=responses)
    print("[main] :)  done")


if __name__ == "__main__":
    # ユーザー警告（UserWarning）を無視する設定
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()

    # 引数を元に main 関数を実行
    main(output_name=args.output_name, mode=Mode(args.mode), vectorstore_option=VectorStoreOption(args.vectorstore))

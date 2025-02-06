"""メイン処理を定義するモジュール。

このモジュールは、コマンドライン引数の解析、ベクトルストアの構築、
ResearchAssistant の呼び出し、及び結果の書き出しを行います。
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
    """コマンドライン引数を解析する関数。

    Returns:
        argparse.Namespace: 解析されたコマンドライン引数のオブジェクト。
    """
    # fmt: off
    parser = argparse.ArgumentParser()
    opt = parser.add_argument

    # 出力ファイル名の指定
    opt("--output-name", "-o", type=str)

    # 動作モードの指定（"submit" または "test"）
    opt("--mode", "-m", type=str, choices=[choice.value for choice in Mode], default=Mode.TEST.value)

    # 使用するベクトルストアの指定
    opt("--vectorstore", "-v", type=str, choices=[choice.value for choice in VectorStoreOption], default=VectorStoreOption.CHROMA.value)
    # fmt: on

    return parser.parse_args()



@traceable
def main(output_name: str, mode: Mode, vectorstore_option: VectorStoreOption) -> None:
    """メインの処理を実行する関数。

    ベクトルストアを構築し、各クエリに対して ResearchAssistant の処理を呼び出し、
    その結果を CSV ファイルに保存します。

    Args:
        output_name (str): 出力ファイル名（拡張子なし）。
        mode (Mode): 動作モード（TEST または SUBMIT）。
        vectorstore_option (VectorStoreOption): 使用するベクトルストアのオプション。
    """
    # ベクトルストアを構築
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
    main(
        output_name=args.output_name,
        mode=Mode(args.mode),
        vectorstore_option=VectorStoreOption(args.vectorstore)
    )

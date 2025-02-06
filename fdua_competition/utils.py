"""ユーティリティ関数を定義するモジュール。

このモジュールには、ルートディレクトリの取得、クエリの取得、結果の書き出し、
及びリトライ時のコールバック関数などが含まれます。
"""

import os
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from fdua_competition.enums import Mode


def get_root() -> Path:
    """プロジェクトのルートディレクトリを取得する。

    Returns:
        Path: プロジェクトのルートディレクトリのパス。
    """
    return Path(os.getenv("FDUA_DIR")) / ".fdua-competition"



def get_queries(mode: Mode) -> list[str]:
    """指定されたモードに基づいてクエリ（問題）のリストを取得する。

    Args:
        mode (Mode): 動作モード（TEST または SUBMIT）。

    Returns:
        list[str]: クエリのリスト。

    Raises:
        ValueError: 未知のモードが指定された場合。
    """
    match mode:
        case Mode.TEST:
            # テストモードの場合、validation ディレクトリの CSV を読み込む
            df = pd.read_csv(get_root() / "validation/ans_txt.csv")
            return df["problem"].tolist()

        case Mode.SUBMIT:
            # 提出モードの場合、query.csv を読み込む
            df = pd.read_csv(get_root() / "query.csv")
            return df["problem"].tolist()

        case _:
            raise ValueError(f"): unknown mode: {mode}")



def print_before_retry(retry_state):
    """リトライ前に呼び出されるコールバック関数。

    Args:
        retry_state: リトライ状態を示すオブジェクト。
    """
    print(
        f":( retrying attempt {retry_state.attempt_number} after exception: {retry_state.outcome.exception()}",
        file=sys.stderr
    )



def write_result(output_name: str, responses: list[BaseModel]) -> None:
    """回答結果を CSV ファイルに書き出す。

    Args:
        output_name (str): 出力ファイル名（拡張子なし）。
        responses (list[BaseModel]): 各クエリに対する応答結果のリスト。
    """
    # 出力に必須なフィールドが存在することを確認
    assert responses[0].response, "response field is missing"

    output_path = get_root() / f"results/{output_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 各レスポンスの response フィールドのみを CSV として保存
    df = pd.DataFrame([{"response": res.response} for res in responses])
    df.to_csv(output_path, header=False)

    print(f"[write_result] done: {output_path}")

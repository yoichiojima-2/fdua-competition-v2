# fdua-competition

## インストール

リポジトリのルートにある secrets ディレクトリに、以下のファイルを配置する

```bash
secrets
├── .env
└── google-application-credentials.json
```

.env ファイルには、以下の環境変数を設定する

```bash
OUTPUT_NAME=結果csvファイル名
AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT
OPENAI_API_VERSION=YOUR_OPENAI_API_VERSION
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=YOUR_LANGCHAIN_API_KEY
```

## 開発コンテナのビルド

以下のコマンドを実行してください

*`make`が実行するコマンドは`Makefile`を参照*

```bash
make up # コンテナを起動
```

```bash
make in # コンテナに入る
```

`make up` はjupyter labも起動する. (`http://localhost:8888`)


## 使い方

*`make`が実行するコマンドは`Makefile`を参照*

### メインスクリプトを実行する

```bash
make run  # 初回実行時に必要なデータを収集する
```

### テストを実行する

```bash
make test
```

### 結果を評価する

```bash
make evaluate
```

### 結果とスコアを見る

```bash
make summary
```

出力例

```bash
======================================== score =========================================

score: 0.05

+--------------+--------------+---------+
| evaluation   |   unit_score |   index |
+==============+==============+=========+
| Perfect      |          1   |      14 |
+--------------+--------------+---------+
| Acceptable   |          0.5 |      11 |
+--------------+--------------+---------+
| Missing      |          0   |       8 |
+--------------+--------------+---------+
| Incorrect    |         -1   |      17 |
+--------------+--------------+---------+

======================================== detail ========================================

question: 大成温調が積極的に資源配分を行うとしている高付加価値セグメントを全てあげてください。
answer: 改修セグメント、医療用・産業用セグメント、官公庁セグメント
output: 情報は利用できません。
evaluation: Missing

question: 花王の生産拠点数は何拠点ですか？
answer: 36拠点
output: 生産拠点数は36拠点です
evaluation: Perfect

question: 電通グループPurposeは何ですか？
answer: an invitation to the never before.
output: 社会にポジティブな動力を生み出すことです
evaluation: Incorrect
...
```


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

```bash
make up # コンテナを起動
```

```bash
make in # コンテナに入る
```

`make up` はjupyter labも起動する. (`http://localhost:8888`)


## 使い方

メインスクリプトを実行する

```bash
make run  # 初回実行時に必要なデータを収集する
```

テストを実行する

```bash
make test
```

結果を評価する

```bash
make evaluate
```

結果とスコアを見る

```bash
make summary
```

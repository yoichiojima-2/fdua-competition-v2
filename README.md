
## installation

place the following files in the `assets` directory at the repository root:

```bash
assets
├── documents.zip
├── evaluation.zip
├── query.csv
├── readme.md
├── sample_submit.zip
└── validation.zip
```

place the following files in the `secrets` directory at the repository root:

```bash
secrets
├── .env
└── google-application-credentials.json
```

the `.env` file should contain the following environment variables:
```bash
FDUA_DIR="path/to/fdua-competition"
AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
OPENAI_API_VERSION="YOUR_AZURE_OPENAI_API_VERSION"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="YOUR_LANGCHAIN_API_KEY"
```

to build dev container, run:
```bash
# build the container
docker build -t fdua:latest .
# run the container
docker run -it --name fdua -v ${PWD}:/app fdua:latest bash
```

## usage
to run the main script:
```bash
make run
```

to evaluate the results:
```bash
make evaluate
```

to summarize the result:
```bash
make summarize-result
```

to test:
```bash
make test
```
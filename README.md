
## installation

place the following files in the `secrets` directory at the repository root:

```bash
secrets
├── .env
└── google-application-credentials.json
```

the `.env` file should contain the following environment variables:
```bash
FDUA_DIR=/fdua-competition
AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
OPENAI_API_VERSION="YOUR_AZURE_OPENAI_API_VERSION"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="YOUR_LANGCHAIN_API_KEY"
```

to build dev container, run:
```bash
# start the container
make up  # this also starts jupyter lab. open http://localhost:8888 on your browser

# enter the container
make in
```

## usage
to run the main script:
```bash
make run  # this collects required data on first execution
```

to test:
```bash
make test
```

to evaluate the results:
```bash
make evaluate
```

to summarize the result:
```bash
make summary
```

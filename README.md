## requirements
- python@3.12
- google-cloud-sdk
- make

```bash
sudo apt update
sudo apt install make google-cloud-sdk

curl -fsSL https://pyenv.run | bash
pyenv install 3.12
```

## installation

`fdua-competition/secrets/.env`

```bash
FDUA_DIR="path/to/fdua-competition"
AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT
OPENAI_API_VERSION=YOUR_AZURE_OPENAI_API_VERSION
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=YOUR_LANGCHAIN_API_KEY
```

```bash
export GOOGLE_APPLICATION_CREDENTIALS="${PWD}/secrets/google-application-credentials.json"

make install
```

## run 
```bash
make run
```

## evaluate
```bash
make evaluate
```

## summarize results
```bash
make summarize-reults
```

UV = uv run
PYTHON = ${UV} python
PYTEST = ${UV} pytest
GS_PATH = "gs://yo-personal/fdua-competition"

install:
	-mkdir .fdua-competition
	find assets -type f -name "*.zip" -print -exec unzip -o {} -d .fdua-competition/ \;
	cp assets/query.csv .fdua-competition/
	cp assets/readme.md .fdua-competition/
	make clean

run:
	${PYTHON} -m main

test:
	${PYTEST} -vvv

pre-commit: lint clean

lint:
	${UV} isort .
	${UV} ruff check . --fix
	${UV} ruff format .

clean:
	-rm -rf .venv
	-rm uv.lock
	find . -type d -name "__pycache__" -print -exec rm -r {} +
	find . -type d -name ".pytest_cache" -print -exec rm -r {} +
	find . -type d -name ".ruff_cache" -print -exec rm -r {} +
	find . -type d -name "__MACOSX" -print -exec rm -r {} +
	find . -type f -name ".DS_Store" -print -exec rm -r {} +
	find . -type f -name "*.Identifier" -print -exec rm -r {} +

download-assets:
	-mkdir assets
	gsutil -m cp -r ${GS_PATH}/assets/* assets/
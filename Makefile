PYTHON = uv run python
PYTEST = uv run pytest

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
	uv run isort .
	uv run ruff check . --fix
	uv run ruff format .

clean:
	-rm -rf .venv
	-rm uv.lock
	find . -type d -name "__pycache__" -print -exec rm -r {} +
	find . -type d -name ".pytest_cache" -print -exec rm -r {} +
	find . -type d -name ".ruff_cache" -print -exec rm -r {} +
	find . -type d -name "__MACOSX" -print -exec rm -r {} +
	find . -type f -name ".DS_Store" -print -exec rm -r {} +
	find . -type f -name "*.Identifier" -print -exec rm -r {} +


DATA_DIR = ~/.fdua-competition

download:
	mkdir -p ${DATA_DIR}/downloads
	gsutil -m cp -r gs://yo-personal/fdua/downloads/* ${DATA_DIR}/downloads

download-secret:
	gsutil cp gs://yo-personal/fdua/secret/.env .

upload: clean
	gsutil -m cp -r ${DATA_DIR}/downloads gs://yo-personal/fdua/

unzip:
	cd ${DATA_DIR}/downloads && \
	find . -name "*.zip" -print -exec unzip {} \;

clean:
	find ${DATA_DIR} -type d -name "*__MACOS*" -print -exec rm -r {} +
	find ${DATA_DIR} -type f -name ".DS_Store" -print -exec rm -r {} +
	find ${DATA_DIR} -type d -name "__pycache__" -print -exec rm -r {} +
	find ${DATA_DIR} -type f -name "*.Identifier" -print -exec rm -r {} +

lint:
	ruff check . --fix
	ruff format .

pre-commit: lint clean

test:
	uv run pytest -vvv
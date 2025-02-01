UV = uv run
GS_PATH = gs://fdua-competition
ASSETS_DIR = assets
INSTALL_DIR = .fdua-competition

run: install
	${UV} python -m main --mode test

test: install
	${UV} pytest -vvv

evaluate: install
	${UV} python ${INSTALL_DIR}/evaluation/crag.py

install: ${INSTALL_DIR}/.installed
${INSTALL_DIR}/.installed: ${ASSETS_DIR}/.success
	echo ${INSTALL_DIR}
	-mkdir -p ${INSTALL_DIR}
	find assets -type f -name "*.zip" -print -exec unzip -o {} -d ${INSTALL_DIR} \;
	perl -pi -e 's/from openai import OpenAI/from openai import AzureOpenAI/' ${INSTALL_DIR}/evaluation/src/evaluator.py
	perl -pi -e 's/client = OpenAI()/client = AzureOpenAI()/' ${INSTALL_DIR}/evaluation/src/evaluator.py
	cp assets/query.csv ${INSTALL_DIR}/
	cp assets/readme.md ${INSTALL_DIR}/
	make clean
	touch ${INSTALL_DIR}/.installed

download-assets: ${ASSETS_DIR}/.success
${ASSETS_DIR}/.success:
	-mkdir assets
	gsutil -m cp -r ${GS_PATH}/assets/* assets/
	touch ${ASSETS_DIR}/.success

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

upload-results:
	gsutil -m cp -r .fdua-competition/results ${GS_PATH}/

download-results:
	gsutil -m cp -r ${GS_PATH}/results .fdua-competition/

upload-secrets:
	gsutil -m cp -r secrets ${GS_PATH}/

download-secrets:
	gsutil -m cp -r ${GS_PATH}/secrets .

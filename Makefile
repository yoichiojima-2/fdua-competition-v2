PWD = $(shell pwd)
UV = uv run
GS_PATH = gs://fdua-competition
ASSETS_DIR = assets
INSTALL_DIR = .fdua-competition
OUTPUT_NAME = output_simple_test

run: install
	${UV} python -m main -o ${OUTPUT_NAME}

test: install
	${UV} pytest -vvv

test-evaluate: install
	uv run python ${INSTALL_DIR}/evaluation/crag.py \
		--model-name 4omini \
		--result-dir ${PWD}/${INSTALL_DIR}/evaluation/submit \
		--result-name predictions.csv \
		--ans-dir ${PWD}/${INSTALL_DIR}/evaluation/data \
		--ans-txt ans_txt.csv \
		--eval-result-dir ${PWD}/${INSTALL_DIR}/evaluation/result

evaluate: install
	uv run python ${INSTALL_DIR}/evaluation/crag.py \
		--model-name 4omini \
		--result-dir ${PWD}/${INSTALL_DIR}/results \
		--result-name output_simple_test.csv \
		--ans-dir ${PWD}/${INSTALL_DIR}/evaluation/data \
		--ans-txt ans_txt.csv \
		--eval-result-dir ${PWD}/${INSTALL_DIR}/evaluation/result

install: ${INSTALL_DIR}/.installed
${INSTALL_DIR}/.installed: ${ASSETS_DIR}/.success
	-mkdir -p ${INSTALL_DIR}
	find assets -type f -name "*.zip" -print -exec unzip -o {} -d ${INSTALL_DIR} \;
	cp assets/query.csv ${INSTALL_DIR}/
	cp assets/readme.md ${INSTALL_DIR}/
	${UV} python bin/fix_evaluator.py
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

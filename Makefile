include secrets/.env
export $(shell sed 's/=.*//' secrets/.env)

PWD = $(shell pwd)
UV = uv run
GS_PATH = gs://fdua-competition
ASSETS_DIR = assets
SECRETS_DIR = secrets
INSTALL_DIR = .fdua-competition
OUTPUT_NAME = v$(shell uv run python -m bin.print_version)
MODE = test
CSV_PATH = ${INSTALL_DIR}/results/${MODE}/${OUTPUT_NAME}.csv

up:
	@echo "\starting container..."
	docker compose up -d

down:
	@echo "\nstopping container..."
	docker compose down

in:
	@echo "\nentering container..."
	docker compose run fdua-competition-v2

run: ${CSV_PATH}
${CSV_PATH}: ${INSTALL_DIR}/index/.success
	@echo "\nrunning..."
	${UV} python -m fdua_competition.main -m ${MODE}
	@echo "done"

vectorstore: ${INSTALL_DIR}/vectorstores/chroma/${OUTPUT_NAME}/.success
${INSTALL_DIR}/vectorstores/chroma/${OUTPUT_NAME}/.success: ${INSTALL_DIR}/.installed
	@echo "\npreparing vectorstore..."
	${UV} python -m fdua_competition.vectorstore 
	touch ${INSTALL_DIR}/vectorstores/chroma/${OUTPUT_NAME}/.success
	@echo "done"

index: ${INSTALL_DIR}/index/.success
${INSTALL_DIR}/index/.success: ${INSTALL_DIR}/vectorstores/chroma/${OUTPUT_NAME}/.success
	@echo "\npreparing index..."
	${UV} python -m fdua_competition.index_documents -m ${MODE}
	${UV} python -m fdua_competition.index_pages -m ${MODE}
	touch ${INSTALL_DIR}/index/.success
	@echo "done"

evaluate: ${PWD}/${INSTALL_DIR}/evaluation/result/scoring.csv
${PWD}/${INSTALL_DIR}/evaluation/result/scoring.csv: ${CSV_PATH}
	@echo "\nevaluating..."
	${UV} python ${INSTALL_DIR}/evaluation/crag.py \
		--model-name 4omini \
		--result-dir ${PWD}/${INSTALL_DIR}/results/${MODE} \
		--result-name ${OUTPUT_NAME}.csv \
		--ans-dir ${PWD}/${INSTALL_DIR}/evaluation/data \
		--ans-txt ans_txt.csv \
		--eval-result-dir ${PWD}/${INSTALL_DIR}/evaluation/result \
		--max-num-tokens 150  # should be removed
	@echo "done"

summary: evaluate
	@echo "\nsummarizing..."
	-mkdir -p ${INSTALL_DIR}/summary
	${UV} python bin/summarize_result.py -m ${MODE} | tee ${INSTALL_DIR}/summary/${OUTPUT_NAME}.txt
	@echo "done"

test: install
	@echo "\ntesting..."
	${UV} pytest -vvv
	@echo "done"

install: ${INSTALL_DIR}/.installed
${INSTALL_DIR}/.installed: ${ASSETS_DIR}/.success
	@echo "\ninstalling..."
	-mkdir -p ${INSTALL_DIR}
	find assets -type f -name "*.zip" -print -exec unzip -o {} -d ${INSTALL_DIR} \;
	cp assets/query.csv ${INSTALL_DIR}/
	cp assets/readme.md ${INSTALL_DIR}/
	${UV} python bin/fix_evaluator.py
	pip install --upgrade pip && pip install uv
	make clean
	touch ${INSTALL_DIR}/.installed
	@echo "done"

download-assets: ${ASSETS_DIR}/.success
${ASSETS_DIR}/.success:
	@echo "\ndownloading assets..."
	-mkdir -p ${ASSETS_DIR}
	gsutil -m cp -r ${GS_PATH}/assets/* assets/
	touch ${ASSETS_DIR}/.success
	@echo "done"

pre-commit: lint clean

lint:
	@echo "\nlinting..."
	${UV} isort .
	${UV} ruff check . --fix
	${UV} ruff format .
	@echo "done"

clean:
	@echo "\ncleaning..."
	find . -type d -name "__pycache__" -print -exec rm -r {} +
	find . -type d -name ".pytest_cache" -print -exec rm -r {} +
	find . -type d -name ".ruff_cache" -print -exec rm -r {} +
	find . -type d -name "__MACOSX" -print -exec rm -r {} +
	find . -type f -name ".DS_Store" -print -exec rm -r {} +
	find . -type f -name "*.Identifier" -print -exec rm -r {} +
	find . -type d -name ".ipynb_checkpoints" -print -exec rm -r {} +
	find . -type f -name "*.ipynb" -print -exec ${UV} jupyter nbconvert --clear-output {} \;
	@echo "done"

clear-result:
	-rm ${CSV_PATH}
	-rm ${PWD}/${INSTALL_DIR}/evaluation/result/scoring.csv

clear-index:
	-rm -r ${INSTALL_DIR}/index

clear-container:
	@echo "\ncleaning container..."
	docker ps -qa | xargs docker rm -f && docker images -q | xargs docker rmi -f
	@echo "done"

clear-vectorstore:
	-rm -r ${INSTALL_DIR}/vectorstores

clear-log:
	-rm -r ${INSTALL_DIR}/log

uninstall: clean clear-container
	@echo "\nuninstalling..."
	-rm -rf .venv
	-rm uv.lock
	-rm -rf ${INSTALL_DIR}
	-rm -rf ${ASSETS_DIR}
	-rm -rf ${SECRETS_DIR}
	@echo "done"

upload-secrets:
	gsutil -m cp -r secrets ${GS_PATH}/

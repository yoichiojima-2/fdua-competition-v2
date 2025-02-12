import json
import os
import textwrap
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

from fdua_competition.enums import LogLevel, Mode
from fdua_competition.get_version import get_version
from fdua_competition.logging_config import logger, set_log_level
from fdua_competition.models import create_chat_model, create_embeddings
from fdua_competition.pdf_handler import get_document_dir
from fdua_competition.utils import before_sleep_hook, dict_to_yaml
from fdua_competition.vectorstore import FduaVectorStore
from fdua_competition.utils import read_queries

OUTPUT_DIR = Path(os.environ["FDUA_DIR"]) / ".fdua-competition/index"


class IndexDocumentOutput(BaseModel):
    organizations: list[str] = Field(description="[Organization names extracted from the document]")
    relevant_queries: list[str] = Field(description="[Relevant queries used to extract the organization names]")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def extract_organization_name(source: Path, vectorstore: VectorStore, mode: Mode):
    role = textwrap.dedent(
        """
        あなたは情報抽出に特化した高度な言語モデルです。提供されたテキストから組織の正式名称を正確に特定し、抽出することが任務です。
        関連するクエリが見つかった場合は、それも出力に含めてください。

        - 組織の正式名称のみを抽出してください（例：「トヨタ自動車株式会社」、「Google Japan」）。
        - 「株式会社」、「支社」、「部門」などの一般的な単語は、正式名称の一部でない限り無視してください。
        - 「会社」、「支店」、「オフィス」などの一般的な用語が単独で出現する場合は除外してください。
        - 個人名、場所、一般的な部門名は含めないでください。
        - 有効な組織名が見つからない場合は、「None」と返してください。
        - 抽出された組織に関連するクエリがない場合は、関連クエリにクエリを含めないでください。

        抽出の正確性と完全性を確保してください。
        """
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 30})  # Increased from 20
    context = retriever.invoke(role, filter={"source": str(source)})

    chat_model = create_chat_model().with_structured_output(IndexDocumentOutput)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", role), ("system", "context:\n{context}"), ("system", "queries: {queries}")],
    )
    chain = prompt_template | chat_model

    res = chain.invoke({"context": "\n---\n".join([i.page_content for i in context]), "queries": read_queries(Mode(mode))})
    logger.info(f"[extract_organization_name]\n{dict_to_yaml(res.model_dump())}\n")
    return res


def extract_organization_name_concurrently(pdfs: list[Path], vectorstore: VectorStore, mode: Mode) -> list[dict]:
    responses = []
    with ThreadPoolExecutor() as executor:
        future_to_pdf = {executor.submit(extract_organization_name, pdf, vectorstore, mode): pdf for pdf in pdfs}
        for future in tqdm(as_completed(future_to_pdf), total=len(pdfs), desc="extracting organization names.."):
            pdf = future_to_pdf[future]
            names = future.result()
            responses.append({**names.model_dump(), "source": str(pdf)})
    return responses


def write_document_index(vectorstore: VectorStore, mode: Mode = Mode.TEST):
    logger.info("[write_document_index] creating document index..")

    output_path = OUTPUT_DIR / f"v{get_version()}/document/{mode.value}.json"

    pdfs = list(get_document_dir(mode).rglob("*.pdf"))
    organization_names = extract_organization_name_concurrently(pdfs, vectorstore, mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(organization_names, f, ensure_ascii=False, indent=2)

    logger.info(f"[write_document_index] done: {output_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--mode", "-m", type=str, default=Mode.TEST.value, required=True)
    opt("--log-level", "-l", type=str, default=LogLevel.INFO.value)
    return parser.parse_args()


def main():
    args = parse_args()
    set_log_level(LogLevel(args.log_level))
    embeddings = create_embeddings()
    vs = FduaVectorStore(mode=Mode(args.mode), embeddings=embeddings)
    write_document_index(vectorstore=vs, mode=Mode(args.mode))


def read_document_index(mode: Mode) -> str:
    index_path = OUTPUT_DIR / f"v{get_version()}/document/{mode.value}.json"
    index = json.load(index_path.open())
    return dict_to_yaml(index)


if __name__ == "__main__":
    main()

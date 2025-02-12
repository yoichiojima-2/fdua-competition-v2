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
        You are an advanced language model specializing in information extraction. Your task is to accurately identify and extract the full names of organizations from the provided text.
        if you find relevant queries in given queries, please include them in the output as well.

        - Extract only full organization names (e.g., "Toyota Motor Corporation", "Google Japan").
        - Ignore common words like "株式会社", "支社", "部門" unless they are part of a full organization name.
        - Exclude general terms like "company", "branch", "office" if they appear alone.
        - Do not include personal names, locations, or generic department names.
        - If no valid organization name is found, return "None".

        Ensure accuracy and completeness in your extraction.
        """
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
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

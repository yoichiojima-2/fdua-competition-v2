import json
import os
import textwrap
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field
from tqdm import tqdm

from fdua_competition.enums import Mode
from fdua_competition.get_version import get_version
from fdua_competition.logger import get_logger
from fdua_competition.models import create_chat_model, create_embeddings
from fdua_competition.pdf_handler import get_document_dir
from fdua_competition.vectorstore import FduaVectorStore

OUTPUT_DIR = Path(os.environ["FDUA_DIR"]) / ".fdua-competition/index/documents"
logger = get_logger()


class IndexDocumentOutput(BaseModel):
    organizations: list[str] = Field(description="[Organization names extracted from the document]")


def extract_organization_name(source: Path, vectorstore: VectorStore):
    role = textwrap.dedent(
        """
        You are an advanced language model specializing in information extraction. Your task is to accurately identify and extract the full names of organizations from the provided text.

        - Extract only full organization names (e.g., "Toyota Motor Corporation", "Google Japan").
        - Ignore common words like "株式会社", "支社", "部門" unless they are part of a full organization name.
        - Exclude general terms like "company", "branch", "office" if they appear alone.
        - Do not include personal names, locations, or generic department names.
        - If no valid organization name is found, return "None".

        Ensure accuracy and completeness in your extraction.
        """
    )

    user_query = "extract organization names from this document"

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    context = retriever.invoke(user_query, filter={"source": str(source)})

    chat_model = create_chat_model().with_structured_output(IndexDocumentOutput)
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", role), ("system", "context:\n{context}"), ("user", user_query)]
    )
    chain = prompt_template | chat_model

    return chain.invoke({"context": "\n---\n".join([i.page_content for i in context])})


def extract_organization_name_concurrently(pdfs: list[Path], vectorstore: VectorStore) -> list[dict]:
    organization_names = []
    with ThreadPoolExecutor() as executor:
        future_to_pdf = {executor.submit(extract_organization_name, pdf, vectorstore): pdf for pdf in pdfs}
        for future in tqdm(as_completed(future_to_pdf), total=len(pdfs), desc="extracting organization names.."):
            pdf = future_to_pdf[future]
            names = future.result()
            organization_names.append({"organizations": names.organizations, "source": str(pdf)})
    return organization_names


def write_document_index(vectorstore: VectorStore, mode: Mode = Mode.TEST):
    logger.info("[write_document_index] creating document index..")

    output_path = OUTPUT_DIR / f"v{get_version()}.json"

    pdfs = list(get_document_dir(mode).rglob("*.pdf"))
    organization_names = extract_organization_name_concurrently(pdfs, vectorstore)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(organization_names, f, ensure_ascii=False, indent=2)

    logger.info(f"[write_document_index] done: {output_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--mode", "-m", type=str, default=Mode.TEST.value, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings = create_embeddings()
    vs = FduaVectorStore(embeddings=embeddings)
    write_document_index(vectorstore=vs, mode=Mode(args.mode))


def read_document_index() -> str:
    index_path = OUTPUT_DIR / f"v{get_version()}.json"
    index = json.load(index_path.open())
    return yaml.dump(index, allow_unicode=True, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()

import json
import os
import textwrap
from argparse import ArgumentParser
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field
from tqdm import tqdm

from fdua_competition.enums import Mode
from fdua_competition.models import create_chat_model, create_embeddings
from fdua_competition.pdf_handler import get_document_dir
from fdua_competition.vectorstore import FduaVectorStore

OUTPUT_DIR = Path(os.environ["FDUA_DIR"]) / ".fdua-competition/index"


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

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", role), ("system", "context:\n{context}"), ("user", user_query)]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    context = retriever.invoke(user_query, filter={"source": str(source)})

    chat_model = create_chat_model().with_structured_output(IndexDocumentOutput)
    chain = prompt_template | chat_model

    return chain.invoke({"context": "\n---\n".join([i.page_content for i in context])})


def write_index(output_name: str, vectorstore: VectorStore, mode: Mode = Mode.TEST):
    print("[write_index] creating index..")

    organization_names = []
    pdfs = list(get_document_dir(mode).rglob("*.pdf"))
    for pdf_path in tqdm(pdfs, desc="extracting organization names.."):
        names = extract_organization_name(source=pdf_path, vectorstore=vectorstore)
        organization_names.append({"organizations": names.organizations, "source": str(pdf_path)})

    output_path = OUTPUT_DIR / f"{output_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(organization_names, f, ensure_ascii=False, indent=2)

    print(f"[write_index] done: {output_path}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_name", "-o", type=str)
    return parser.parse_args()


def main(output_name: str):
    embeddings = create_embeddings()
    vs = FduaVectorStore(output_name=output_name, embeddings=embeddings)
    write_index(output_name=output_name, vectorstore=vs)


def read_index(output_name: str) -> str:
    index_path = OUTPUT_DIR / f"{output_name}.json"
    index = json.load(index_path.open())
    return yaml.dump(index, allow_unicode=True, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    args = parse_args()
    main(output_name=args.output_name)

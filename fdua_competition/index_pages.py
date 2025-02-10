import json
import os
import textwrap
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field
from tqdm import tqdm

from fdua_competition.enums import Mode
from fdua_competition.get_version import get_version
from fdua_competition.logging_config import logger
from fdua_competition.models import create_chat_model, create_embeddings
from fdua_competition.pdf_handler import get_document_dir
from fdua_competition.utils import dict_to_yaml
from fdua_competition.vectorstore import FduaVectorStore

OUTPUT_DIR = Path(os.environ["FDUA_DIR"]) / ".fdua-competition/index/pages"


def get_document(source: Path, vectorstore: VectorStore) -> list[Document]:
    docs = vectorstore.get(where={"source": str(source)})
    return [Document(page_content=doc, metadata=metadata) for doc, metadata in zip(docs["documents"], docs["metadatas"])]


class SummarizePageOutput(BaseModel):
    topics: list[str] = Field(description="topics or themes covered in the document page.")
    summary: str = Field(description="A concise summary of the document page.")


def summarize_page(document: Document) -> SummarizePageOutput:
    role = textwrap.dedent(
        """
        You are an advanced language model specializing in text summarization.
        Your task is to generate a **precise and conservative summary** of the provided page in Japanese.
        This summary will serve as a page index for later document retrieval.

        - Focus on capturing only the essential factual details and main topics that clearly identify the page content.
        - Avoid interpretative analysis, creative embellishments, or any additional details that are not directly present in the text.
        - The summary should be **concise and strictly factual**, ideally in **2-3 sentences**.
        - Include only the necessary elements such as names, dates, and key concepts relevant for retrieval.
        - If the page lacks substantial content or is irrelevant, return "None".

        Ensure that your summary remains as close to the original content as possible, without adding extra interpretation.
        """
    )
    chat_model = create_chat_model().with_structured_output(SummarizePageOutput)
    prompt_template = ChatPromptTemplate.from_messages([("system", role), ("system", "page_content:\n{page_content}")])
    chain = prompt_template | chat_model
    return chain.invoke({"page_content": document.page_content})


def summarize_page_concurrently(docs: list[Document]) -> list[dict]:
    summaries = []
    with ThreadPoolExecutor() as executor:
        future_to_doc = {executor.submit(summarize_page, doc): doc for doc in docs}
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            summary = future.result()
            summaries.append({**summary.model_dump(), "metadata": doc.metadata})

    summary_sorted = sorted(summaries, key=lambda x: x["metadata"]["page"])
    logger.info(f"[summarize_page_concurrently]\n{dict_to_yaml(summary_sorted)}\n")
    return summary_sorted


def write_page_index(source: Path, vectorstore: VectorStore) -> None:
    output_path = OUTPUT_DIR / f"v{get_version()}/{source.stem}.json"

    docs = get_document(source, vectorstore=vectorstore)
    page_index = summarize_page_concurrently(docs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(page_index, f, ensure_ascii=False, indent=2)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--mode", "-m", type=str, default=Mode.TEST.value, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = create_embeddings()
    vs = FduaVectorStore(embeddings=embeddings)

    pdfs = list(get_document_dir(mode=Mode(args.mode)).rglob("*.pdf"))

    with ThreadPoolExecutor() as executor:
        future_to_pdf = {executor.submit(write_page_index, source=pdf, vectorstore=vs): pdf for pdf in pdfs}
        for future in tqdm(as_completed(future_to_pdf), total=len(pdfs), desc="indexing pages.."):
            pdf = future_to_pdf[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"error processing {pdf}: {e}")


def read_page_index(source: Path) -> str:
    index_path = OUTPUT_DIR / f"v{get_version()}/{source.stem}.json"
    index = json.load(index_path.open())
    return dict_to_yaml(index)


if __name__ == "__main__":
    main()

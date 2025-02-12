import json
import typing as t
import os
import textwrap
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document
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

OUTPUT_DIR = Path(os.environ["FDUA_DIR"]) / ".fdua-competition/index"


def get_document(source: Path, vectorstore: VectorStore) -> list[Document]:
    docs = vectorstore.get(where={"source": str(source)})
    return [Document(page_content=doc, metadata=metadata) for doc, metadata in zip(docs["documents"], docs["metadatas"])]


class SummarizePageOutput(BaseModel):
    topics: list[str] = Field(description="topics or themes covered in the document page.")
    summary: str = Field(description="A concise summary of the document page.")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def summarize_page(document: Document) -> SummarizePageOutput:
    role = textwrap.dedent(
        """
        あなたはテキスト要約に特化した高度な言語モデルです。
        提供されたページの包括的かつ包括的な要約を日本語で生成することが任務です。
        この要約は後のドキュメント検索のためのページインデックスとして使用されます。

        - 主なトピック、重要な詳細、およびページ内容を特定するのに役立つ関連するコンテキストを捉えることに焦点を当ててください。
        - 名前、日付、主要な概念などの重要な事実の詳細を含め、検索に役立つ追加のコンテキストも含めてください。
        - テキストの意味を明確にするのに役立つ場合は、簡単な解釈分析を取り入れてもかまいません。
        - 要約はページ内容の明確な概要を提供し、理想的には**3〜5文**であるべきです。
        - ページに実質的な内容がない場合や関連性がない場合は、「None」と返してください。

        要約がページ内容を正確に反映し、明示的な詳細と有用な暗黙のコンテキストの両方を含むようにしてください。
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

    summary_sorted: list[dict] = sorted(summaries, key=lambda x: x["metadata"]["page"])
    logger.info(f"[summarize_page_concurrently]\n{dict_to_yaml(summary_sorted)}\n")
    return summary_sorted


def write_page_index(source: Path, vectorstore: VectorStore, mode: Mode) -> None:
    output_path = OUTPUT_DIR / f"v{get_version()}/page/{mode.value}/{source.stem}.json"

    docs = get_document(source, vectorstore=vectorstore)
    page_index = summarize_page_concurrently(docs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(page_index, f, ensure_ascii=False, indent=2)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--mode", "-m", type=str, default=Mode.TEST.value, required=True)
    opt("--log-level", "-l", type=str, default=LogLevel.INFO.value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_log_level(LogLevel(args.log_level))

    embeddings = create_embeddings()
    vs = FduaVectorStore(mode=Mode(args.mode), embeddings=embeddings)

    pdfs = list(get_document_dir(mode=Mode(args.mode)).rglob("*.pdf"))

    with ThreadPoolExecutor() as executor:
        future_to_pdf = {
            executor.submit(write_page_index, source=pdf, vectorstore=vs, mode=Mode(args.mode)): pdf for pdf in pdfs
        }
        for future in tqdm(as_completed(future_to_pdf), total=len(pdfs), desc="indexing pages.."):
            pdf = future_to_pdf[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"error processing {pdf}: {e}")


def read_page_index(source: Path, mode: Mode) -> str:
    index_path = OUTPUT_DIR / f"v{get_version()}/page/{mode.value}/{source.stem}.json"
    index = json.load(index_path.open())
    return dict_to_yaml(index)


if __name__ == "__main__":
    main()

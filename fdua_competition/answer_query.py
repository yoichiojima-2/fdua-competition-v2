import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

from fdua_competition.baes_models import AnswerQueryOutput
from fdua_competition.cleanse import CleanseResponseOutput, cleanse_response
from fdua_competition.enums import Mode
from fdua_competition.logging_config import logger
from fdua_competition.merge_results import merge_results
from fdua_competition.models import create_chat_model
from fdua_competition.reference import search_reference_doc, search_reference_pages
from fdua_competition.refine_query import refine_query
from fdua_competition.tools import divide_number, round_number
from fdua_competition.utils import before_sleep_hook, dict_to_yaml
from fdua_competition.vectorstore import FduaVectorStore

MAX_RETRIEVES = 20  # Increased from 10


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def get_relevant_docs_with_index(query: str, vectorstore: FduaVectorStore, mode: Mode) -> list[Document]:
    docs: list[Document] = []
    ref_doc = search_reference_doc(query, mode)

    if ref_doc.source:  # if reference document is found, search for reference pages
        ref_pages = search_reference_pages(query, Path(ref_doc.source), mode)
        if ref_pages.pages:  # if reference pages are found, add pages
            logger.info(f"reference pages found for query: {query}")
            for page in ref_pages.pages:
                try:
                    docs.extend(
                        (
                            vectorstore.as_retriever(
                                search_kwargs={"filter": {"$and": [{"source": ref_doc.source}, {"page": page}]}}
                            ).invoke(query)
                        )
                    )
                except Exception as e:
                    logger.warning(f"error fetching reference page: {page} - {e}")
        else:  # if no reference pages are found, add pages that are most likely to contain the answer
            logger.info(f"no reference pages found for query: {query}")
            docs.extend(
                (
                    vectorstore
                    .as_retriever(search_kwargs={"k": MAX_RETRIEVES, "filter": {"source": ref_doc.source}})
                    .invoke(query)
                )
            )
    else:  # if no reference document is found, add pages that are most likely to contain the answer
        logger.info(f"no reference document found for query: {query}")
        retriever = vectorstore.as_retriever(search_kwargs={"k": MAX_RETRIEVES})
        docs.extend(retriever.invoke(query))

    logger.debug(f"[get_relevant_docs] {docs}")

    return docs


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def answer_query(query: str, vectorstore: FduaVectorStore, mode: Mode) -> CleanseResponseOutput:
    role = textwrap.dedent(
        """ 
        あなたはリサーチアシスタントです。ユーザーのクエリと「コンテキスト」と呼ばれる一連のドキュメントにアクセスできます。
        コンテキストからの情報のみを使用してクエリに回答する必要があります。コンテキストから回答が見つからない場合は、
        単に情報が利用できないか不明であると述べてください。

        出力はこの正確なJSON構造に従う必要があります：
        {
            "query": "元のユーザーの質問",
            "response": "簡潔な回答。情報が不明な場合はnullを設定",
            "reason": "コンテキストから回答を導き出した方法の簡単な説明",
            "organization_name": "言及されている関連組織名",
            "contexts": ["使用された関連コンテキストのリスト、各コンテキストは文字列"]
        }

        ガイドライン:
        1. 出力に追加のフィールドやテキストを含めないでください。
        2. "response"は50トークン以下にし、JSON解析を壊す可能性のあるカンマや特殊文字を使用しないでください。
        3. 提供されたコンテキスト以外の情報を使用しないでください。コンテキストが不十分な場合は、「不明」または「情報が利用できません」と書いてください。
        4. 事実に基づき、明確かつ簡潔にしてください。
        5. 回答全体（必要に応じて説明を含む）を日本語で記述してください。
        """
    )

    # [start: building chain]
    chat_model = create_chat_model().bind_tools([round_number, divide_number]).with_structured_output(AnswerQueryOutput)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{role}"),
            ("system", "context:\n{context}"),
            ("user", "query:\n{query}"),
        ]
    )
    chain = prompt_template | chat_model
    # [end: building chain]

    refined_query = refine_query(query, vectorstore=vectorstore).output

    # [start: prep two types of payload]
    payload_base = {"role": role, "query": refined_query, "language": "japanese"}
    payload_simple = {
        **payload_base,
        "context": "\n---\n".join(
            [
                f"{i.page_content}\n{i.metadata}"
                for i in vectorstore.as_retriever(search_kwargs={"k": MAX_RETRIEVES}).invoke(refined_query)
            ]
        ),
    }
    payload_index = {
        **payload_base,
        "context": "\n---\n".join(
            [f"{i.page_content}\n{i.metadata}" for i in get_relevant_docs_with_index(refined_query, vectorstore, mode)]
        ),
    }
    # [end: prep two types of payload]

    # [start: invoke chain with two payloads]
    with ThreadPoolExecutor() as executor:
        future_simple = executor.submit(chain.invoke, payload_simple)
        future_index = executor.submit(chain.invoke, payload_index)
        res_simple = future_simple.result()
        res_index = future_index.result()

    logger.debug(f"payload_simple: {dict_to_yaml(res_simple.model_dump())}")
    logger.debug(f"payload_index: {dict_to_yaml(res_index.model_dump())}")

    logger.debug(f"[answer_query] res_simple\n{dict_to_yaml(res_simple.model_dump())}\n")
    logger.debug(f"[answer_query] res_index\n{dict_to_yaml(res_index.model_dump())}\n")
    # [end: invoke chain with two payloads]

    res = merge_results(res_index=res_index, res_simple=res_simple, vectorstore=vectorstore, query=query)
    logger.info(f"[answer_query]\n{dict_to_yaml(res.model_dump())}\n")
    return cleanse_response(res)


def answer_queries_concurrently(queries: list[str], vectorstore: FduaVectorStore, mode: Mode) -> list[AnswerQueryOutput]:
    results: dict[int, AnswerQueryOutput] = {}
    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(answer_query, query=query, vectorstore=vectorstore, mode=mode): i
            for i, query in enumerate(queries)
        }
        for future in tqdm(as_completed(future_to_index), total=len(queries), desc="processing queries.."):
            index = future_to_index[future]
            response = future.result()
            results[index] = response

    return [results[i] for i in range(len(queries))]

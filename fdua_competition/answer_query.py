import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_random
from tqdm import tqdm

from fdua_competition.base_models import AnswerQueryOutput
from fdua_competition.cleanse import CleanseResponseOutput, cleanse_response
from fdua_competition.enums import Mode
from fdua_competition.logging_config import logger
from fdua_competition.merge_results import merge_results
from fdua_competition.models import create_chat_model
from fdua_competition.reference import ReferenceDocOutput, ReferencePageOutput, search_reference_doc, search_reference_pages
from fdua_competition.refine_query import refine_query
from fdua_competition.tools import divide_number, round_number
from fdua_competition.utils import before_sleep_hook, dict_to_yaml
from fdua_competition.vectorstore import FduaVectorStore

MAX_RETRIEVES = 10


def add_specific_pages(
    docs: list[Document],
    reference_doc: ReferenceDocOutput,
    reference_pages: ReferencePageOutput,
    query: str,
    vectorstore: FduaVectorStore,
) -> None:
    for page in reference_pages.pages:
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": {"$and": [{"source": reference_doc.source}, {"page": page}]}}
        )
        try:
            docs.extend(retriever.invoke(query))
        except Exception as e:
            logger.warning(f"error fetching reference page: {page} - {e}")


def add_retrieved_pages(
    docs: list[Document], refecence_doc: ReferenceDocOutput, query: str, vectorstore: FduaVectorStore
) -> None:
    retriever = vectorstore.as_retriever(search_kwargs={"filter": {"source": refecence_doc.source}})
    try:
        docs.extend(retriever.invoke(query))
    except Exception as e:
        logger.warning(f"error fetching reference pages: {e}")


def add_retrived_docs(docs: list[Document], query: str, vectorstore: FduaVectorStore) -> None:
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": MAX_RETRIEVES})
        docs.extend(retriever.invoke(query))
    except Exception as e:
        logger.warning(f"error fetching reference docs: {e}")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def get_relevant_docs_with_index(query: str, vectorstore: FduaVectorStore, mode: Mode) -> list[Document]:
    docs: list[Document] = []
    ref_doc = search_reference_doc(query, mode)
    if ref_doc.source:
        ref_pages = search_reference_pages(query, Path(ref_doc.source), mode)
        if ref_pages.pages:
            logger.info(f"found reference pages found for query: {query}")
            add_specific_pages(docs=docs, reference_pages=ref_pages, query=query, vectorstore=vectorstore)
        else:
            logger.info(f"no reference pages found for query: {query}")
            add_retrieved_pages(docs=docs, refecence_doc=ref_doc, query=query, vectorstore=vectorstore)
    else:
        logger.info(f"no reference document found for query: {query}")
        add_retrived_docs(docs=docs, query=query, vectorstore=vectorstore)
    logger.debug(f"[get_relevant_docs] {docs}")
    return docs


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def answer_query(query: str, vectorstore: FduaVectorStore, mode: Mode) -> CleanseResponseOutput:
    role = textwrap.dedent(
        """ 
        You are a research assistant. You have access to the user's query and a set of documents referred to as “context.”
        You must answer the query using only the information from the context. If the answer cannot be found in the context, 
        simply state that the information is unavailable or unknown.

        Your output must follow this exact JSON structure:
        {
            "query": "the original user question",
            "response": "a concise answer. set null if the answer is unknown",
            "reason": "a brief explanation of how you derived the answer from the context",
            "organization_name": "the relevant organization name if it is mentioned",
            "contexts": ["list of relevant context passages used, each as a string"]
        }

        Guidelines:
        1. Do not include any additional fields or text in the output.
        2. keep "response" under 50 tokens and do not use commas or special characters that may break json parsing.
        3. Do not use information outside of the provided context. If the context is insufficient, write “unknown” or “no information available.”
        4. Stay factual, clear, and concise.
        5. Make sure the entire response (including explanation, if needed) is written in {language}.
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
    return cleanse_response(res, mode=mode)


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

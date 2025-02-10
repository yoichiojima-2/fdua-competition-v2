import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from fdua_competition.baes_models import AnswerQueryOutput
from fdua_competition.cleanse import cleanse_response
from fdua_competition.logging_config import logger
from fdua_competition.models import create_chat_model
from fdua_competition.reference import search_reference_doc, search_reference_pages
from fdua_competition.utils import before_sleep_hook, dict_to_yaml
from fdua_competition.vectorstore import FduaVectorStore


@tool
def divide_number(a: str, b: str) -> str:
    """
    divides two numbers.
    args:
        a: the dividend.
        b: the divisor.
    """
    return str(float(a) / float(b))


@tool
def round_number(number: str, decimals: str) -> str:
    """
    rounds a number to a specified number of decimals.
    args:
        number: the number to round.
        decimals: the number of decimals to round to.
    """
    return str(round(float(number), int(decimals - 1)))


@retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=before_sleep_hook)
def answer_query(query: str, vectorstore: FduaVectorStore) -> AnswerQueryOutput:
    role = textwrap.dedent(
        """ 
        You are a research assistant. You have access to the user's query and a set of documents referred to as “context.”
        You must answer the query using only the information from the context. If the answer cannot be found in the context, 
        simply state that the information is unavailable or unknown.

        Your output must follow this exact JSON structure:
        {
            "query": "the original user question",
            "response": "a concise answer",
            "reason": "a brief explanation of how you derived the answer from the context",
            "organization_name": "the relevant organization name if it is mentioned",
            "contexts": ["list of relevant context passages used, each as a string"]
        }

        Guidelines:
        1. Do not include any additional fields or text in the output.
        2. Keep "response" under 50 tokens and do not use commas or special characters that may break JSON parsing.
        3. Do not use information outside of the provided context. If the context is insufficient, write “unknown” or “no information available.”
        4. Stay factual, clear, and concise.
        5. Make sure the entire response (including explanation, if needed) is written in {language}.
        """
    )

    ref_doc = search_reference_doc(query)
    ref_pages = search_reference_pages(query, Path(ref_doc.source))

    contexts = []
    for page in ref_pages.pages:
        retriever = vectorstore.as_retriever(search_kwargs={"filter": {"$and": [{"source": ref_doc.source}, {"page": page}]}})
        page_contexts = retriever.invoke(query)
        for i in page_contexts:
            contexts.append(i)

    parsed_context = dict_to_yaml([{"content": i.page_content, "metadata": i.metadata} for i in contexts])

    chat_model = create_chat_model().bind_tools([round_number, divide_number]).with_structured_output(AnswerQueryOutput)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{role}"),
            ("system", "context:\n{context}"),
            ("user", "query:\n{query}"),
        ]
    )
    chain = prompt_template | chat_model
    payload = {"role": role, "context": parsed_context, "query": query, "language": "japanese"}
    res = chain.invoke(payload)
    logger.info(f"[answer_query]\n{dict_to_yaml(res.model_dump())}\n")
    return res


def answer_queries_concurrently(queries: list[str], vectorstore: FduaVectorStore) -> list[AnswerQueryOutput]:
    results: dict[int, AnswerQueryOutput] = {}
    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(answer_query, query=query, vectorstore=vectorstore): i for i, query in enumerate(queries)
        }
        for future in tqdm(as_completed(future_to_index), total=len(queries), desc="processing queries.."):
            index = future_to_index[future]
            response = future.result()
            results[index] = cleanse_response(response)
    return [results[i] for i in range(len(queries))]

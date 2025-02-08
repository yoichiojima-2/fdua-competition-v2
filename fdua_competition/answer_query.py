import textwrap

import yaml
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from fdua_competition.models import create_chat_model
from fdua_competition.reference import search_source_to_refer
from fdua_competition.utils import log_retry
from fdua_competition.vectorstore import FduaVectorStore


class AnswerQueryOutput(BaseModel):
    query: str = Field(description="the query that was asked.")
    response: str = Field(description="the answer for the given query")
    reason: str = Field(description="the reason for the response.")
    organization_name: str = Field(description="the organization name that the query is about.")
    contexts: list[str] = Field(description="the context that the response was based on with its file path and page number.")
    reference: str = Field(description="the reference source of the context.")


@retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=log_retry)
def answer_query(query: str, vectorstore: FduaVectorStore):
    reference = search_source_to_refer(query)
    context = vectorstore.as_retriever().invoke(query, filter={"source": reference.source})

    parsed_context = yaml.dump(
        [{"content": i.page_content, "metadata": i.metadata} for i in context],
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )

    role = textwrap.dedent(
        """ 
        You are a research assistant. You have access to the user’s query and a set of documents referred to as “context.”
        You must answer the query using only the information from the context. If the answer cannot be found in the context, 
        simply state that the information is unavailable or unknown.

        Your output must follow this exact JSON structure:
        {
            "query": "the original user question",
            "response": "a concise answer with no more than 50 tokens, no commas",
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

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{role}"),
            ("system", "context:\n{context}"),
            ("user", "query:\n{query}"),
        ]
    )

    chat_model = create_chat_model().with_structured_output(AnswerQueryOutput)
    chain = prompt_template | chat_model

    return chain.invoke({"role": role, "context": parsed_context, "query": query, "language": "japanese"})

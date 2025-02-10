import textwrap

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random

from fdua_competition.baes_models import AnswerQueryOutput
from fdua_competition.logging_config import logger
from fdua_competition.models import create_chat_model
from fdua_competition.utils import dict_to_yaml
from fdua_competition.utils import before_sleep_hook


class CleanseResponseOutput(BaseModel):
    query: str = Field(..., title="The query string that was used to generate the answer.")
    input: str = Field(..., title="The raw answer output provided in the 'response' field.")
    output: str = Field(..., title="The cleansed 'response' string that satisfies the requirements.")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def cleanse_response(answer: AnswerQueryOutput) -> CleanseResponseOutput:
    role = textwrap.dedent(
        """
        You are an intelligent assistant specializing in text refinement.
        Your task is to simplify and clarify the provided answer.

        ## Instructions:
        - Rewrite the answer using minimal and clear language.
        - Remove any unnecessary words to ensure direct clarity.
        - The final output should be concise and within 54 tokens.

        ## Input:
        - **answer**: The original answer provided in the "response" field.

        ## Output:
        Return a refined "response" string that is clear, minimal, and concise.
        """
    )

    chat_model = create_chat_model().with_structured_output(CleanseResponseOutput)
    prompt_template = ChatPromptTemplate.from_messages([("system", role), ("user", "answer: {answer}")])
    chain = prompt_template | chat_model
    res = chain.invoke({"answer": dict_to_yaml(answer.model_dump())})

    logger.info(f"[cleanse_answer_query]\n{dict_to_yaml(res.model_dump())}\n")
    return res

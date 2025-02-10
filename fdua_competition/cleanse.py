import textwrap

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random

from fdua_competition.baes_models import AnswerQueryOutput
from fdua_competition.logging_config import logger
from fdua_competition.models import create_chat_model
from fdua_competition.tools import divide_number, round_number
from fdua_competition.utils import before_sleep_hook, dict_to_yaml


class CleanseResponseOutput(BaseModel):
    query: str = Field(description="The query string that was used to generate the answer.")
    input: str = Field(description="The raw answer output provided in the 'response' field.")
    output: str = Field(description="The cleansed 'response' string that satisfies the requirements.")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def cleanse_response(answer: AnswerQueryOutput) -> CleanseResponseOutput:
    role = textwrap.dedent(
        """
        You are an intelligent assistant specializing in text refinement.
        Your task is to return the provided answer exactly as given—with only minimal cleanup (such as whitespace or punctuation adjustments) if needed.
        
        ## Instructions:
        - Do not rephrase or add any additional context or commentary.
        - Do not include extra subject headers, company names, or explanations that are not present in the original answer.
        - Simply output the essential answer text, as is, ensuring it is clear and minimal.
        - If the answer is null, return an empty string.
        - The final output should be a single, minimal phrase or value, within 54 tokens.
        
        ## Input:
        - **answer**: The original answer from the "response" field.
        
        ## Output:
        Return the refined "response" string exactly as provided, with only minimal corrections.
        """
    )
    chat_model = create_chat_model().bind_tools([divide_number, round_number]).with_structured_output(CleanseResponseOutput)
    prompt_template = ChatPromptTemplate.from_messages([("system", role), ("user", "answer: {answer}")])
    chain = prompt_template | chat_model
    res = chain.invoke({"answer": dict_to_yaml(answer.model_dump())})

    logger.info(f"[cleanse_answer_query]\n{dict_to_yaml(res.model_dump())}\n")
    return res

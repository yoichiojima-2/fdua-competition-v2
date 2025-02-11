import textwrap

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random

from fdua_competition.baes_models import AnswerQueryOutput
from fdua_competition.models import create_chat_model
from fdua_competition.utils import before_sleep_hook, dict_to_yaml


class MergeResultsOutput(BaseModel):
    query: str = Field(description="The original user question.")
    res_index: str = Field(description="The answer generated using context retrieved via index-based search.")
    certainty_index: float = Field(description="The certainty score for the index-based answer.")
    res_simple: str = Field(description="The answer generated using context retrieved via simple retrieval.")
    certainty_simple: float = Field(description="The certainty score for the simple retrieval answer.")
    output: str = Field(description="The final merged answer after evaluating both results.")
    reason: str = Field(description="A brief explanation of how the merged answer was derived.")


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def merge_results(res_index: AnswerQueryOutput, res_simple: AnswerQueryOutput) -> MergeResultsOutput:
    role = textwrap.dedent(
        """
        You are a research assistant tasked with merging two answers generated from different retrieval methods for the same query.
        Below are two sets of results:
        
        - **Result 1 (res_index):** This answer is based on context retrieved using an index-based search.
        - **Result 2 (res_simple):** This answer is based on context retrieved using a simple retrieval method.

        *Result 2 might be less reliable than Result 1 since it potentially includes contexts which should not be referenced.*
        
        Your task is to produce a merged answer that meets the following requirements:
        
        1. **Query:** Use the original user question (which is identical in both results) for the "query" field.
        2. **Original Answers:** Include the original answers and their corresponding certainty scores in the fields "res_index", "certainty_index", "res_simple", and "certainty_simple".
        3. **Merged Answer ("output"):** 
           - If one result is clearly more reliable (e.g. higher certainty or more complete), use that answer.
           - If both results are similar in certainty and content, synthesize a concise answer.
           - If both indicate that the information is unknown, set the merged answer to "unknown".
        4. **Reason:** Provide a brief explanation in the "reason" field describing how you determined the merged answer.
        5. do not use commas or special characters that may break json parsing.
        6. Simply output the essential answer text, as is, ensuring it is clear and minimal.
        
        Your output must strictly follow this JSON structure without any additional fields or text:
        {
            "query": "The original user question",
            "res_index": "Answer from index-based retrieval",
            "certainty_index": <float>,
            "res_simple": "Answer from simple retrieval",
            "certainty_simple": <float>,
            "output": "The merged answer",
            "reason": "Explanation for the merged answer"
        }
        
        Please ensure that the entire response is written in japanese and that no extraneous text is included.
        """
    )
    res_combined = textwrap.dedent(
        f"""
        ### result 1: the answer based on context retrieved with index 
        {dict_to_yaml(res_index.model_dump())}

        ### result 2: the answer based on context 
        {dict_to_yaml(res_simple.model_dump())}
        """
    )
    chat_model = create_chat_model().with_structured_output(AnswerQueryOutput)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{role}"),
            ("user", "results: {results}"),
        ]
    )
    chain = prompt_template | chat_model
    return chain.invoke({"role": role, "results": res_combined})

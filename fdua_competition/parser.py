from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    query: str = Field(description="the query that was asked.")
    response: str = Field(description="the answer for the given query")
    reason: str = Field(description="the reason for the response.")
    organization_name: str = Field(description="the organization name that the query is about.")
    contexts: list[str] = Field(description="the context that the response was based on with its file path and page number.")


def get_output_parser() -> JsonOutputParser:
    return JsonOutputParser(pydantic_object=ChatResponse)

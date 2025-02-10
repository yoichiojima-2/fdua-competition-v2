from pydantic import BaseModel, Field


class AnswerQueryOutput(BaseModel):
    query: str = Field(description="the query that was asked.")
    response: str = Field(description="the answer for the given query")
    reason: str = Field(description="the reason for the response.")
    organization_name: str = Field(description="the organization name that the query is about.")
    contexts: list[str] = Field(description="the context that the response was based on with its file path and page number.")
    source: str = Field(description="the given context source")
    pages: list[int] = Field(description="the page numbers of the given contexts.")

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, pattern=r".*\S.*", description="The question to ask against the document.")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list, description="Relevant source chunks used to generate the answer.")

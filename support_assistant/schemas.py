"""Request, response, and provider validation models."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class IntentResult(BaseModel):
    intent: Literal["policy_question", "general_question"]

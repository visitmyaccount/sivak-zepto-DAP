"""Optional real-provider calls with structured-output retries."""

from __future__ import annotations

import json
import os
from typing import TypeVar

from groq import Groq
from pydantic import BaseModel, ValidationError

MODEL_TYPE = TypeVar("MODEL_TYPE", bound=BaseModel)
DEFAULT_MODEL = "openai/gpt-oss-20b"


def provider_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


def request_validated_json(
    messages: list[dict[str, str]], response_model: type[MODEL_TYPE]
) -> MODEL_TYPE:
    """Call the provider and retry twice when JSON or schema validation fails."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    working_messages = list(messages)
    last_error = "No response received"

    for attempt in range(3):
        completion = client.chat.completions.create(
            model=provider_model(),
            messages=working_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__.lower(),
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            },
            temperature=0,
        )
        raw_content = completion.choices[0].message.content or ""
        try:
            return response_model.model_validate(json.loads(raw_content))
        except (json.JSONDecodeError, ValidationError) as error:
            last_error = str(error)
            if attempt < 2:
                working_messages.extend(
                    [
                        {"role": "assistant", "content": raw_content},
                        {
                            "role": "user",
                            "content": (
                                "The previous response did not match the required JSON schema. "
                                "Return only one corrected JSON object with every required field."
                            ),
                        },
                    ]
                )

    raise RuntimeError(f"Provider response failed validation after 3 attempts: {last_error}")

"""Run and record one policy query and one general query in mock mode."""

from __future__ import annotations

import json
import os
from pathlib import Path

from assistant_graph import SupportAssistant

MODULE_DIR = Path(__file__).parent
OUTPUT_PATH = MODULE_DIR / "outputs" / "mock_examples.json"


def main() -> None:
    os.environ["MOCK_LLM"] = "1"
    assistant = SupportAssistant()
    examples = {
        "policy_question": {
            "query": "What is the delivery fee for a small order?",
            "response": assistant.ask(
                "What is the delivery fee for a small order?"
            ).model_dump(),
        },
        "general_question": {
            "query": "Who won the football match?",
            "response": assistant.ask("Who won the football match?").model_dump(),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()

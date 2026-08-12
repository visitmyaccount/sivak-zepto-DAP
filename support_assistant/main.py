"""FastAPI wrapper for the Zepto support graph."""

from functools import lru_cache

from fastapi import FastAPI

from assistant_graph import SupportAssistant, mock_mode_enabled
from schemas import AskRequest, AskResponse

app = FastAPI(title="Zepto Policy Support Assistant", version="1.0.0")


@lru_cache(maxsize=1)
def get_assistant() -> SupportAssistant:
    return SupportAssistant()


@app.get("/")
def status() -> dict[str, str]:
    return {
        "status": "ready",
        "mode": "mock" if mock_mode_enabled() else "real-provider",
        "usage": "Send a POST request to /ask with a JSON query field.",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return get_assistant().ask(request.query)

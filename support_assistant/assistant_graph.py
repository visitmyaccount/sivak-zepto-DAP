"""LangGraph intent router and grounded answer flow."""

from __future__ import annotations

import os
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from prompt_template import GENERAL_PROMPT, POLICY_PROMPT
from provider import request_validated_json
from retrieval import PolicyRetriever
from schemas import AskResponse, IntentResult

POLICY_KEYWORDS = (
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
)


class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_chunks: list[dict[str, str | float]]
    response: dict[str, object]


def mock_mode_enabled() -> bool:
    """Mock mode is on unless the environment explicitly contains 0."""
    return os.getenv("MOCK_LLM", "1") != "0"


class SupportAssistant:
    """Own the retriever and compiled three-node graph."""

    def __init__(self) -> None:
        self.retriever = PolicyRetriever()
        self.graph = self._build_graph()

    def classify_intent(self, state: GraphState) -> GraphState:
        query = state["query"]
        if mock_mode_enabled():
            lowered = query.lower()
            intent = (
                "policy_question"
                if any(keyword in lowered for keyword in POLICY_KEYWORDS)
                else "general_question"
            )
        else:
            result = request_validated_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "Classify the query as policy_question when it needs the Zepto "
                            "policy corpus, otherwise classify it as general_question."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                IntentResult,
            )
            intent = result.intent
        return {"intent": intent}

    def retrieve_and_answer(self, state: GraphState) -> GraphState:
        chunks = self.retriever.retrieve(state["query"], limit=3)
        chunk_dicts = [chunk.to_dict() for chunk in chunks]
        sources = [chunk.chunk_id for chunk in chunks]

        if mock_mode_enabled():
            snippet = chunks[0].text[:200].strip()
            if len(chunks[0].text) > 200:
                snippet += "..."
            response = AskResponse(
                answer=f"Based on the retrieved context: {snippet}",
                sources=sources,
                confidence=1.0,
            )
        else:
            context = "\n\n".join(
                f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks
            )
            response = request_validated_json(
                [
                    {"role": "system", "content": "Return grounded structured JSON."},
                    {
                        "role": "user",
                        "content": POLICY_PROMPT.format(
                            context=context, query=state["query"]
                        ),
                    },
                ],
                AskResponse,
            )

        return {
            "retrieved_chunks": chunk_dicts,
            "response": response.model_dump(),
        }

    def direct_answer(self, state: GraphState) -> GraphState:
        if mock_mode_enabled():
            response = AskResponse(
                answer="I can only answer questions about Zepto policies right now.",
                sources=[],
                confidence=1.0,
            )
        else:
            response = request_validated_json(
                [
                    {"role": "system", "content": "Return structured JSON only."},
                    {
                        "role": "user",
                        "content": GENERAL_PROMPT.format(query=state["query"]),
                    },
                ],
                AskResponse,
            )
            response.sources = []
        return {"response": response.model_dump()}

    @staticmethod
    def route_after_classification(
        state: GraphState,
    ) -> Literal["retrieve_and_answer", "direct_answer"]:
        return (
            "retrieve_and_answer"
            if state["intent"] == "policy_question"
            else "direct_answer"
        )

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("classify_intent", self.classify_intent)
        builder.add_node("retrieve_and_answer", self.retrieve_and_answer)
        builder.add_node("direct_answer", self.direct_answer)
        builder.add_edge(START, "classify_intent")
        builder.add_conditional_edges(
            "classify_intent",
            self.route_after_classification,
            {
                "retrieve_and_answer": "retrieve_and_answer",
                "direct_answer": "direct_answer",
            },
        )
        builder.add_edge("retrieve_and_answer", END)
        builder.add_edge("direct_answer", END)
        return builder.compile()

    def ask(self, query: str) -> AskResponse:
        result = self.graph.invoke({"query": query})
        return AskResponse.model_validate(result["response"])

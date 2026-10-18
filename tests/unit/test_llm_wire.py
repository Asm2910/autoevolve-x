"""What actually goes over the wire to the Anthropic API (the HTTP reply is faked, no network).

These guard against two real failures seen on the first live run: current Claude models reject a
`temperature` setting and forced tool calling, so the request must use the JSON-schema mode.
"""

import json

import pytest

from agents.llm_proposer import LLMProposer
from agents.spec import ExperimentSpec

httpx2 = pytest.importorskip("httpx2")
anthropic = pytest.importorskip("anthropic")
chat = pytest.importorskip("langchain_anthropic")

ANSWER = {
    "stop": False, "stop_reason": "", "model": "xgboost", "groups": ["usage", "network"], "windows": [30, 90],
    "settings": [{"name": "max_depth", "value": 4}],
    "hypothesis": "usage drives the signal",
}


def test_request_shape_and_response_parsing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen = {}

    def handler(request):
        body = json.loads(request.content)
        seen.update(body)
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": json.dumps(ANSWER)}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 5},
        })

    llm = chat.ChatAnthropic(model="claude-sonnet-5-5", max_tokens=500, max_retries=0)
    llm._client = anthropic.Anthropic(
        api_key="test-key", http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    proposer = LLMProposer(llm, {"max_depth": 3}, ["usage", "network"], [7, 30, 90], 0.01)
    champion = ExperimentSpec(model="xgboost", groups=["usage"], windows=[7])

    spec = proposer.propose(champion, {"lessons": [], "tried": []})

    assert "temperature" not in seen
    assert "tools" not in seen  # no forced tool calling
    assert seen["output_config"]["format"]["type"] == "json_schema"
    assert "anyOf" not in json.dumps(seen["output_config"]["format"]["schema"])  # union types were rejected live
    assert spec.key() == 'xgboost|network,usage|30,90|{"max_depth": 4.0}'
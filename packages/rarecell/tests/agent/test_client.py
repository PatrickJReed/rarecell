import respx
from httpx import Response
from rarecell.agent.client import AnthropicClient


def test_client_loads_system_prompt():
    client = AnthropicClient(api_key="fake-key")
    assert "single-cell genomics advisor" in client.system_prompt


@respx.mock
def test_client_call_with_messages_returns_text():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "model": "claude-haiku-4-5",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )
    )

    client = AnthropicClient(api_key="fake-key", model="claude-haiku-4-5")
    resp = client.messages_create(messages=[{"role": "user", "content": "Hi"}])
    assert resp["content"][0]["text"] == "Hello"

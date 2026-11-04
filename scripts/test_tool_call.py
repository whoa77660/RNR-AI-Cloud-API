"""Verify whether the configured local Ollama model emits native tool_calls."""
import json
import os
import sys

import requests

ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
model_name = os.getenv("MODEL_NAME", "qwen3.8-27b-uncensored-mtp")

payload = {
    "model": model_name,
    "stream": False,
    "temperature": 0,
    "messages": [{
        "role": "user",
        "content": "You must call the function rnr_agent_probe with value exactly 'RNR_TOOL_CALL_OK'. Do not answer in normal text.",
    }],
    "tools": [{
        "type": "function",
        "function": {
            "name": "rnr_agent_probe",
            "description": "A diagnostic function used only to verify native tool calling.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }],
}

print("Testing Ollama native tool calling")
print("URL:", ollama_url)
print("Model:", model_name)

try:
    response = requests.post(
        f"{ollama_url}/v1/chat/completions",
        json=payload,
        timeout=(20, 600),
    )
    response.raise_for_status()
    data = response.json()
except Exception as exc:
    print("TOOL CALL TEST FAILED:", repr(exc))
    sys.exit(2)

choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
tool_calls = message.get("tool_calls") or []

if tool_calls:
    print("\nNATIVE TOOL CALLING: SUPPORTED")
    print(json.dumps(tool_calls, indent=2, ensure_ascii=False))
    print("finish_reason:", choice.get("finish_reason"))
    sys.exit(0)

print("\nNATIVE TOOL CALLING: NOT CONFIRMED")
print("finish_reason:", choice.get("finish_reason"))
print("model content:", message.get("content"))
print("The model may be ignoring the tools schema or its Ollama template may not support tools.")
sys.exit(1)

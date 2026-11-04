# RNR AI Cloud API — Agent Tool-Calling Update

OpenAI-compatible Render gateway for a Kaggle-hosted Ollama model. This update preserves the existing Ask-mode chat flow and forwards native OpenAI-style tool schemas and tool-call messages.

## What changed

- The gateway preserves `tools`, assistant `tool_calls`, `tool_call_id`, tool-result messages, and streaming chunks.
- `tool_choice: "auto"` is removed before Ollama because Ollama's OpenAI-compatible Chat Completions API currently documents `tools` support but not `tool_choice`.
- The Kaggle connector uses asynchronous HTTP requests so inference does not block the WebSocket event loop.
- The connector sends an application heartbeat every 20 seconds so Render does not mark an idle Kaggle agent stale.
- Added `scripts/test_tool_call.py` to verify that the exact Ollama model emits a native `tool_calls` response before enabling VS Code Agent mode.

## Important architecture boundary

VS Code's local Agent host executes workspace/file/terminal tools. Render and Kaggle only transport model requests and responses. The Kaggle notebook does not get direct access to your Windows workspace. VS Code's own permission and confirmation prompts still control local tool execution.

## 1. Update Render service

Replace the repository files with this project and deploy the `main` branch. Keep the current Render Environment values (`ADMIN_API_KEY`, `AGENT_SECRET`, `KEEPALIVE_SECRET`, `MODEL_NAME`) unchanged. The gateway code is backward-compatible with the existing `/v1/models`, `/v1/chat/completions`, and `/agent/connect` routes.

Render build installs the added `httpx` dependency through `requirements.txt`.

## 2. Update Kaggle connector

In the Kaggle notebook, install/update dependencies:

```python
!pip install -q requests websockets httpx
```

After Ollama has started and the model is ready, use this notebook cell. It reads the saved Kaggle Secrets and uses top-level `await` (do not use `%run` for this async connector inside a notebook event loop):

```python
import os
import sys
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ["RNR_GATEWAY_URL"] = "https://rnr-ai-cloud-api.onrender.com"
os.environ["RNR_AGENT_SECRET"] = user_secrets.get_secret("RNR_AGENT_SECRET")
os.environ["OLLAMA_URL"] = "http://127.0.0.1:11434"
os.environ["MODEL_NAME"] = "qwen3.8-27b-uncensored-mtp"

sys.path.insert(0, "/kaggle/working/rnr-ai-cloud-api")
from kaggle.connector import main
await main()
```

Make sure the existing secrets/environment variables are still configured:

- `RNR_GATEWAY_URL`
- `RNR_AGENT_SECRET`
- `OLLAMA_URL` (normally `http://127.0.0.1:11434`)
- `MODEL_NAME` (normally `qwen3.8-27b-uncensored-mtp`)

Stop the old connector cell, pull/copy the updated `kaggle/connector.py`, and run it. Keep the cell running.

## 3. Verify native tool calls BEFORE enabling Agent

Run this in Kaggle after Ollama is ready:

```python
%run /kaggle/working/rnr-ai-cloud-api/scripts/test_tool_call.py
```

Run it from the repository root, or copy `scripts/test_tool_call.py` into your working directory. Expected success includes:

```text
NATIVE TOOL CALLING: SUPPORTED
```

and a JSON object containing `function.name` equal to `rnr_agent_probe`.

If it prints `NOT CONFIRMED`, do not set `toolCalling: true` yet. The model response or Ollama Modelfile/template needs investigation. Transport support alone cannot make a model generate tool calls.

## 4. Enable VS Code Agent mode

After the Kaggle tool-call test succeeds, set the RNR model entry in `chatLanguageModels.json` to:

```json
{
    "name": "RNR AI Cloud API",
    "vendor": "customendpoint",
    "apiKey": "${input:chat.lm.secret.1b399c79}",
    "apiType": "chat-completions",
    "models": [
        {
            "id": "qwen3.8-27b-uncensored-mtp",
            "name": "RNR Qwen 27B Uncensored",
            "url": "https://rnr-ai-cloud-api.onrender.com/v1/chat/completions",
            "toolCalling": true,
            "vision": false,
            "streaming": false,
            "maxInputTokens": 65536,
            "maxOutputTokens": 8192
        }
    ]
}
```

Keep the API key in VS Code Secret Storage, not in the JSON file. Use the existing Render `ADMIN_API_KEY` value in the VS Code key prompt.

Start with `streaming: false` because that was the confirmed working mode in the existing setup. Tool calls are still returned in the non-streaming OpenAI chat completion response. Streaming can be enabled later after separate verification.

## 5. Agent smoke test

In VS Code, choose **Agent** mode and the RNR model. Ask it to:

1. Inspect the current workspace and summarize the top-level files.
2. Create `rnr-agent-test.txt` containing `RNR AGENT TOOL TEST OK`.
3. Read the file back and confirm its content.

Approve the file/terminal actions when VS Code prompts. This tests actual tool execution by VS Code, not by the Kaggle server.

## API and security notes

- The gateway currently supports automatic tool selection only. Explicit forced `tool_choice` modes are rejected because Ollama's compatible endpoint does not document that field.
- Tool definitions and tool results are relayed as JSON; this gateway does not execute arbitrary tools.
- Do not commit API keys or secrets. Rotate any key that has been shared publicly or pasted into chat.
- Use one Render instance because WebSocket request state is held in memory.
- A stopped Kaggle session takes the model offline. Free Render instances may also sleep.

"""Kaggle-to-Render WebSocket connector for an Ollama OpenAI-compatible API."""
import asyncio
import json
import os
import random
import time

import httpx
import websockets

GATEWAY_URL = os.environ["RNR_GATEWAY_URL"].rstrip("/")
AGENT_SECRET = os.environ["RNR_AGENT_SECRET"]
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.8-27b-uncensored-mtp")
REQUEST_TIMEOUT = float(os.getenv("RNR_REQUEST_TIMEOUT", "900"))


def ws_url():
    scheme_url = GATEWAY_URL.replace("https://", "wss://").replace("http://", "ws://")
    return f"{scheme_url}/agent/connect?token={AGENT_SECRET}"


def normalize_request(body):
    if not isinstance(body, dict):
        raise ValueError("Gateway payload must be a JSON object")
    payload = dict(body)
    payload["model"] = MODEL_NAME

    # Ollama supports the OpenAI tools schema. Its OpenAI-compatible
    # Chat Completions endpoint does not currently implement tool_choice.
    if payload.get("tool_choice") in ("auto", None):
        payload.pop("tool_choice", None)
    elif payload.get("tool_choice") is not None:
        raise ValueError("Ollama Chat Completions supports automatic tool selection only")

    return payload


async def answer_request(ws, message):
    request_id = message["id"]
    try:
        body = normalize_request(message.get("payload", {}))
        timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=20.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            if body.get("stream", False):
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/v1/chat/completions",
                    json=body,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(f"Ollama returned invalid SSE JSON: {data[:300]}") from exc
                        if isinstance(chunk, dict):
                            await ws.send(json.dumps({
                                "type": "chunk",
                                "id": request_id,
                                "data": chunk,
                            }, ensure_ascii=False))
                await ws.send(json.dumps({"type": "done", "id": request_id}))
                return

            response = await client.post(
                f"{OLLAMA_URL}/v1/chat/completions",
                json=body,
            )
            response.raise_for_status()
            try:
                result = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"Ollama returned non-JSON response: {response.text[:500]}"
                ) from exc
            if not isinstance(result, dict):
                raise RuntimeError("Ollama returned an unexpected JSON response")

            # Preserve choices[].message.tool_calls, finish_reason=tool_calls,
            # usage, and all other OpenAI response fields unchanged.
            await ws.send(json.dumps({
                "type": "result",
                "id": request_id,
                "data": result,
            }, ensure_ascii=False))
            await ws.send(json.dumps({"type": "done", "id": request_id}))

    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1000]
        await ws.send(json.dumps({
            "type": "error",
            "id": request_id,
            "error": f"Ollama HTTP {exc.response.status_code}: {detail}",
        }, ensure_ascii=False))
        await ws.send(json.dumps({"type": "done", "id": request_id}))
    except Exception as exc:
        await ws.send(json.dumps({
            "type": "error",
            "id": request_id,
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False))
        await ws.send(json.dumps({"type": "done", "id": request_id}))


async def heartbeat(ws):
    while True:
        await asyncio.sleep(20)
        await ws.send(json.dumps({"type": "pong", "timestamp": int(time.time())}))


async def main():
    delay = 2
    while True:
        try:
            print("Connecting to RNR gateway:", GATEWAY_URL)
            async with websockets.connect(
                ws_url(),
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
                max_size=20 * 1024 * 1024,
            ) as ws:
                print("RNR agent connected. Model:", MODEL_NAME)
                delay = 2
                heartbeat_task = asyncio.create_task(heartbeat(ws))
                tasks = set()
                try:
                    async for raw in ws:
                        message = json.loads(raw)
                        kind = message.get("type")
                        if kind == "hello":
                            print("Gateway accepted agent:", message)
                        elif kind == "request":
                            task = asyncio.create_task(answer_request(ws, message))
                            tasks.add(task)
                            task.add_done_callback(tasks.discard)
                finally:
                    heartbeat_task.cancel()
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print("Gateway connection lost:", repr(exc))
            print(f"Reconnecting in {delay}s...")
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 30)


if __name__ == "__main__":
    asyncio.run(main())

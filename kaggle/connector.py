"""
Run this in a Kaggle notebook after Ollama is running.
Set RNR_GATEWAY_URL and RNR_AGENT_SECRET from the Render environment.
"""
import asyncio
import json
import os
import random
import time

import requests
import websockets

GATEWAY_URL = os.environ["RNR_GATEWAY_URL"].rstrip("/")
AGENT_SECRET = os.environ["RNR_AGENT_SECRET"]
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.8-27b-uncensored-mtp")

def ws_url():
    return GATEWAY_URL.replace("https://", "wss://").replace("http://", "ws://") + "/agent/connect?token=" + AGENT_SECRET

async def answer_request(ws, message):
    request_id = message["id"]
    body = message["payload"]
    body["model"] = MODEL_NAME
    # This connector supports both streaming and non-streaming Ollama OpenAI API.
    try:
        if body.get("stream", False):
            with requests.post(
                f"{OLLAMA_URL}/v1/chat/completions",
                json=body,
                stream=True,
                timeout=(20, 600)
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        await ws.send(json.dumps({
                            "type": "chunk",
                            "id": request_id,
                            "data": json.loads(data)
                        }))
            await ws.send(json.dumps({"type": "done", "id": request_id}))
        else:
            response = requests.post(
                f"{OLLAMA_URL}/v1/chat/completions",
                json=body,
                timeout=(20, 600)
            )
            response.raise_for_status()
            await ws.send(json.dumps({
                "type": "result",
                "id": request_id,
                "data": response.json()
            }))
            await ws.send(json.dumps({"type": "done", "id": request_id}))
    except Exception as exc:
        await ws.send(json.dumps({
            "type": "error",
            "id": request_id,
            "error": str(exc)
        }))
        await ws.send(json.dumps({"type": "done", "id": request_id}))

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
                max_size=20 * 1024 * 1024
            ) as ws:
                print("RNR agent connected. Model:", MODEL_NAME)
                delay = 2
                async for raw in ws:
                    message = json.loads(raw)
                    if message.get("type") == "hello":
                        print("Gateway accepted agent:", message)
                    elif message.get("type") == "request":
                        asyncio.create_task(answer_request(ws, message))
        except Exception as exc:
            print("Gateway connection lost:", repr(exc))
            print(f"Reconnecting in {delay}s...")
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 30)

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import hashlib
import hmac
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.8-27b-uncensored-mtp")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
AGENT_SECRET = os.getenv("AGENT_SECRET", "")
KEEPALIVE_SECRET = os.getenv("KEEPALIVE_SECRET", "")
MAX_INFLIGHT = max(1, int(os.getenv("MAX_INFLIGHT", "2")))
AGENT_STALE_SECONDS = max(30, int(os.getenv("AGENT_STALE_SECONDS", "75")))


class AgentHub:
    """Correlates HTTP requests with responses from the single Kaggle agent."""

    def __init__(self) -> None:
        self.ws: WebSocket | None = None
        self.lock = asyncio.Lock()
        self.pending: dict[str, asyncio.Queue] = {}
        self.last_seen = 0.0
        self.send_lock = asyncio.Lock()
        self.slots = asyncio.Semaphore(MAX_INFLIGHT)

    async def attach(self, ws: WebSocket) -> None:
        async with self.lock:
            old = self.ws
            self.ws = ws
            self.last_seen = time.time()
        if old and old is not ws:
            try:
                await old.close(code=4001, reason="Replaced by new agent")
            except Exception:
                pass

    async def detach(self, ws: WebSocket) -> None:
        async with self.lock:
            if self.ws is ws:
                self.ws = None
                pending = list(self.pending.values())
                self.pending.clear()
            else:
                pending = []
        for queue in pending:
            await queue.put({"type": "error", "error": "Kaggle agent disconnected"})

    async def send(self, payload: dict) -> None:
        async with self.send_lock:
            if self.ws is None:
                raise RuntimeError("Model agent is offline")
            await self.ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def request(self, payload: dict):
        request_id = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue()
        self.pending[request_id] = queue
        try:
            await self.send({"type": "request", "id": request_id, "payload": payload})
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=900)
                yield item
                if item.get("type") in ("done", "error"):
                    break
        finally:
            self.pending.pop(request_id, None)

    async def receive(self, data: dict) -> None:
        self.last_seen = time.time()
        request_id = data.get("id")
        queue = self.pending.get(request_id)
        if queue is not None:
            await queue.put(data)


hub = AgentHub()


def safe_equal(left: str, right: str) -> bool:
    return bool(left and right and hmac.compare_digest(left.encode(), right.encode()))


def require_api_key(authorization: str | None) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(503, "Server API key is not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer API key")
    if not safe_equal(authorization[7:].strip(), ADMIN_API_KEY):
        raise HTTPException(401, "Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="RNR AI Cloud API", version="1.1.0", lifespan=lifespan)


@app.get("/")
async def root():
    return {"service": "RNR AI Cloud API", "status": "online", "docs": "/docs"}


@app.get("/health")
async def health():
    connected = hub.ws is not None and (time.time() - hub.last_seen) < AGENT_STALE_SECONDS
    return {
        "status": "ok",
        "agent_connected": connected,
        "model": MODEL_NAME,
        "timestamp": int(time.time()),
    }


@app.get("/v1/status")
async def status(authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    connected = hub.ws is not None and (time.time() - hub.last_seen) < AGENT_STALE_SECONDS
    return {
        "gateway": "online",
        "model_agent": "online" if connected else "offline",
        "model": MODEL_NAME,
        "tool_calling_relay": True,
    }


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    if hub.ws is None or (time.time() - hub.last_seen) >= AGENT_STALE_SECONDS:
        raise HTTPException(503, "Model Offline: Kaggle agent is not connected")
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rnr",
        }],
    }


def validate_chat_request(body: Any) -> dict:
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise HTTPException(400, "Request must contain a messages array")
    if not body["messages"]:
        raise HTTPException(400, "messages must not be empty")
    if not all(isinstance(message, dict) for message in body["messages"]):
        raise HTTPException(400, "Every messages item must be a JSON object")

    # Keep the backend model fixed, but preserve all messages and tool fields.
    # VS Code's custom endpoint sends tool definitions and later replays
    # assistant tool_calls + tool result messages in the chat history.
    body["model"] = MODEL_NAME

    # Ollama's OpenAI-compatible Chat Completions API currently documents
    # `tools` support but not `tool_choice`. VS Code commonly sends "auto";
    # remove that hint so Ollama can select tools from the supplied schema.
    tool_choice = body.get("tool_choice")
    if tool_choice in ("auto", None):
        body.pop("tool_choice", None)
    elif tool_choice is not None:
        raise HTTPException(
            400,
            "This Ollama Chat Completions connector supports automatic tool selection only; "
            "explicit tool_choice modes are not supported.",
        )

    # Do not drop tools, tool_calls, tool_call_id, tool results, or streaming.
    return body


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    try:
        body = validate_chat_request(await request.json())
    except json.JSONDecodeError:
        raise HTTPException(400, "Request body must be valid JSON")

    if hub.ws is None or (time.time() - hub.last_seen) >= AGENT_STALE_SECONDS:
        raise HTTPException(503, "Model Offline: Start the Kaggle connector")

    if not body.get("stream", False):
        async with hub.slots:
            final_result = None
            try:
                async for item in hub.request(body):
                    kind = item.get("type")
                    if kind == "result":
                        final_result = item.get("data")
                    elif kind == "error":
                        raise HTTPException(502, item.get("error", "Inference failed"))
            except asyncio.TimeoutError:
                raise HTTPException(504, "Inference timed out")
            except RuntimeError as exc:
                raise HTTPException(503, str(exc))
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(502, f"Agent request failed: {exc}")

        if not isinstance(final_result, dict):
            raise HTTPException(502, "Agent returned no valid completion result")
        return JSONResponse(final_result)

    async def event_stream():
        async with hub.slots:
            try:
                async for item in hub.request(body):
                    kind = item.get("type")
                    if kind == "chunk":
                        chunk = item.get("data")
                        if isinstance(chunk, dict):
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    elif kind == "error":
                        error_payload = {"error": {"message": item.get("error", "Inference failed")}}
                        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    elif kind == "done":
                        yield "data: [DONE]\n\n"
            except asyncio.TimeoutError:
                payload = {"error": {"message": "Inference timed out"}}
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                payload = {"error": {"message": str(exc)}}
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/agent/connect")
async def agent_connect(ws: WebSocket, token: str = ""):
    if not safe_equal(token, AGENT_SECRET):
        await ws.close(code=4401, reason="Unauthorized agent")
        return
    await ws.accept()
    await hub.attach(ws)
    try:
        await ws.send_text(json.dumps({
            "type": "hello",
            "model": MODEL_NAME,
            "capabilities": ["chat_completions", "tools", "streaming"],
        }))
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "pong":
                hub.last_seen = time.time()
            else:
                await hub.receive(data)
    except WebSocketDisconnect:
        pass
    finally:
        await hub.detach(ws)


@app.get("/keepalive")
async def keepalive(ts: str = "", sig: str = ""):
    if not KEEPALIVE_SECRET:
        raise HTTPException(503, "Keepalive secret is not configured")
    try:
        timestamp = int(ts)
    except (TypeError, ValueError):
        raise HTTPException(400, "Invalid timestamp")
    if abs(int(time.time()) - timestamp) > 300:
        raise HTTPException(401, "Expired keepalive signature")
    expected = hmac.new(KEEPALIVE_SECRET.encode(), ts.encode(), hashlib.sha256).hexdigest()
    if not safe_equal(sig, expected):
        raise HTTPException(401, "Invalid keepalive signature")
    return {"ok": True, "timestamp": timestamp}

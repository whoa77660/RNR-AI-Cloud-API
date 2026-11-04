import asyncio
import hashlib
import hmac
import json
import os
import secrets
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
AGENT_STALE_SECONDS = 75

class AgentHub:
    def __init__(self):
        self.ws: WebSocket | None = None
        self.lock = asyncio.Lock()
        self.pending: dict[str, asyncio.Queue] = {}
        self.last_seen = 0.0
        self.send_lock = asyncio.Lock()
        self.slots = asyncio.Semaphore(MAX_INFLIGHT)

    async def attach(self, ws: WebSocket):
        async with self.lock:
            old = self.ws
            self.ws = ws
            self.last_seen = time.time()
        if old and old is not ws:
            try:
                await old.close(code=4001, reason="Replaced by new agent")
            except Exception:
                pass

    async def detach(self, ws: WebSocket):
        async with self.lock:
            if self.ws is ws:
                self.ws = None
                for q in self.pending.values():
                    await q.put({"type": "error", "error": "Kaggle agent disconnected"})
                self.pending.clear()

    async def send(self, payload: dict):
        async with self.send_lock:
            if not self.ws:
                raise RuntimeError("Model agent is offline")
            await self.ws.send_text(json.dumps(payload))

    async def request(self, payload: dict):
        request_id = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue()
        self.pending[request_id] = queue
        try:
            await self.send({"type": "request", "id": request_id, "payload": payload})
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=600)
                yield item
                if item.get("type") in ("done", "error"):
                    break
        finally:
            self.pending.pop(request_id, None)

    async def receive(self, data: dict):
        self.last_seen = time.time()
        request_id = data.get("id")
        queue = self.pending.get(request_id)
        if queue:
            await queue.put(data)

hub = AgentHub()

def safe_equal(a: str, b: str) -> bool:
    return bool(a and b and hmac.compare_digest(a.encode(), b.encode()))

def require_api_key(authorization: str | None):
    if not ADMIN_API_KEY:
        raise HTTPException(503, "Server API key is not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer API key")
    supplied = authorization[7:].strip()
    if not safe_equal(supplied, ADMIN_API_KEY):
        raise HTTPException(401, "Invalid API key")

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="RNR AI Cloud API", version="1.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {"service": "RNR AI Cloud API", "status": "online", "docs": "/docs"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent_connected": hub.ws is not None and (time.time() - hub.last_seen) < AGENT_STALE_SECONDS,
        "model": MODEL_NAME,
        "timestamp": int(time.time())
    }

@app.get("/v1/status")
async def status(authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    connected = hub.ws is not None and (time.time() - hub.last_seen) < AGENT_STALE_SECONDS
    return {"gateway": "online", "model_agent": "online" if connected else "offline", "model": MODEL_NAME}

@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    if not hub.ws or (time.time() - hub.last_seen) >= AGENT_STALE_SECONDS:
        raise HTTPException(503, "Model Offline: Kaggle agent is not connected")
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "rnr"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    body = await request.json()
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise HTTPException(400, "Request must contain a messages array")
    body["model"] = MODEL_NAME
    body.pop("reasoning_effort", None) if body.get("reasoning_effort") == "max" else None

    if not hub.ws or (time.time() - hub.last_seen) >= AGENT_STALE_SECONDS:
        raise HTTPException(503, "Model Offline: Start the Kaggle connector")
    stream = bool(body.get("stream", False))

    async def event_stream():
        async with hub.slots:
            try:
                async for item in hub.request(body):
                    kind = item.get("type")
                    if kind == "chunk":
                        chunk = item.get("data", {})
                        if stream:
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        else:
                            # Non-stream responses are returned as one complete object.
                            pass
                    elif kind == "result":
                        result = item.get("data", {})
                        if stream:
                            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
                        else:
                            yield json.dumps(result, ensure_ascii=False)
                    elif kind == "error":
                        message = item.get("error", "Inference failed")
                        if stream:
                            yield f"data: {json.dumps({'error': {'message': message}})}\n\n"
                        else:
                            yield json.dumps({"error": {"message": message}})
                    elif kind == "done":
                        if stream:
                            yield "data: [DONE]\n\n"
            except asyncio.TimeoutError:
                payload = {"error": {"message": "Inference timed out"}}
                yield (f"data: {json.dumps(payload)}\n\n" if stream else json.dumps(payload))
            except Exception as exc:
                payload = {"error": {"message": str(exc)}}
                yield (f"data: {json.dumps(payload)}\n\n" if stream else json.dumps(payload))

    # Use an event stream for both modes; non-stream is buffered by collecting chunks below.
    if stream:
        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    collected = []
    final_result = None
    async with hub.slots:
        try:
            async for item in hub.request(body):
                if item.get("type") == "result":
                    final_result = item.get("data")
                elif item.get("type") == "error":
                    raise HTTPException(502, item.get("error", "Inference failed"))
        except asyncio.TimeoutError:
            raise HTTPException(504, "Inference timed out")
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))
    if final_result is None:
        raise HTTPException(502, "Agent returned no completion result")
    return JSONResponse(final_result)

@app.websocket("/agent/connect")
async def agent_connect(ws: WebSocket, token: str = ""):
    if not safe_equal(token, AGENT_SECRET):
        await ws.close(code=4401, reason="Unauthorized agent")
        return
    await ws.accept()
    await hub.attach(ws)
    try:
        await ws.send_text(json.dumps({"type": "hello", "model": MODEL_NAME}))
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

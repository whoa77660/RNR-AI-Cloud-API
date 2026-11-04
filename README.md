# RNR AI Cloud API

OpenAI-compatible gateway for a Kaggle-hosted Ollama model.

## 1. Deploy to Render

1. Push this repository to GitHub.
2. In Render, choose **New + → Blueprint** and select this repository.
3. Render reads `render.yaml` and creates the web service.
4. In the Render service Environment page, copy these generated values:
   - `ADMIN_API_KEY`
   - `AGENT_SECRET`
   - `KEEPALIVE_SECRET`
5. Copy your service URL, e.g. `https://rnr-ai-cloud-api.onrender.com`.

Do not publish these secrets or commit them to GitHub.

## 2. Start the Kaggle connector

In Kaggle, install the connector dependencies:

```bash
pip install requests websockets
```

Set environment variables in a notebook cell (use your actual Render values):

```python
import os
os.environ["RNR_GATEWAY_URL"] = "https://YOUR-SERVICE.onrender.com"
os.environ["RNR_AGENT_SECRET"] = "YOUR_AGENT_SECRET"
os.environ["OLLAMA_URL"] = "http://127.0.0.1:11434"
os.environ["MODEL_NAME"] = "qwen3.8-27b-uncensored-mtp"
```

Then run `kaggle/connector.py` in a notebook cell:

```python
%run /kaggle/working/connector.py
```

Upload `kaggle/connector.py` to Kaggle or clone your repository and run it from the repository folder. Keep the cell running while you need the model online.

## 3. Test the API

```bash
curl https://YOUR-SERVICE.onrender.com/health
curl https://YOUR-SERVICE.onrender.com/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_API_KEY"
```

Chat completion:

```bash
curl https://YOUR-SERVICE.onrender.com/v1/chat/completions \
  -H "Authorization: Bearer YOUR_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-uncensored-mtp","messages":[{"role":"user","content":"Hello"}]}'
```

For streaming, add `"stream": true`.

## 4. Use in OpenCode / VS Code clients

- Base URL: `https://YOUR-SERVICE.onrender.com/v1`
- API key: your `ADMIN_API_KEY`
- Model: `qwen3.8-27b-uncensored-mtp`

Configure the client as an OpenAI-compatible custom provider.

## 5. Signed Render keep-alive

`/keepalive` requires a timestamp and HMAC-SHA256 signature. It is not an unauthenticated ping.

The repository includes `scripts/keepalive.py`. Run it from an external machine or scheduler every 10–14 minutes with these environment variables:

```bash
RENDER_SERVICE_URL=https://YOUR-SERVICE.onrender.com
KEEPALIVE_SECRET=YOUR_RENDER_KEEPALIVE_SECRET
```

Install its dependency with `pip install requests`. The script generates a fresh timestamped HMAC-SHA256 signature for every request. Do not put the secret in GitHub. A request to `/health` is not a signed keepalive.

Important: Render free instances may spin down after inactivity. External pings are best-effort and should not be treated as guaranteed uptime. For reliable always-on service, use an always-on paid Render instance.

## Notes

- The gateway keeps one active Kaggle agent connection in memory. Deploy a single Render instance; multiple instances need shared coordination.
- Kaggle sessions are temporary. When Kaggle stops, the API returns `503 Model Offline`.
- API keys are currently a single admin key for the initial deployment. Add per-user key storage/quotas before distributing access publicly.
- Keep the `ADMIN_API_KEY`, `AGENT_SECRET`, and `KEEPALIVE_SECRET` private.

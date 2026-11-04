"""Run from an external machine/scheduler every 10-14 minutes."""
import hashlib
import hmac
import os
import time

import requests

base_url = os.environ["RENDER_SERVICE_URL"].rstrip("/")
secret = os.environ["KEEPALIVE_SECRET"]
timestamp = str(int(time.time()))
signature = hmac.new(
    secret.encode("utf-8"), timestamp.encode("utf-8"), hashlib.sha256
).hexdigest()

response = requests.get(
    f"{base_url}/keepalive",
    params={"ts": timestamp, "sig": signature},
    timeout=20,
)
print("HTTP:", response.status_code)
print(response.text)
response.raise_for_status()

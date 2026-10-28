"""Minimal client for Muse Spark on the Meta Model API (OpenAI-compatible chat completions).

The key is read from the META_API_KEY environment variable and never written anywhere. Every call
is appended to results/agent/calls.jsonl (model, messages, reply, usage), so each narration and
proposal can be traced to the exact prompt that produced it.
"""
import json
import os
import time
import urllib.request

BASE_URL = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
MODEL = os.environ.get("MUSE_MODEL", "muse-spark-1.3")   # not the -contributor variant: our data is unpublished


class MuseClient:
    def __init__(self, log_path: str = "results/agent/calls.jsonl", model: str = MODEL, timeout: float = 120.0):
        self.key = os.environ.get("META_API_KEY")
        if not self.key:
            raise SystemExit("Set META_API_KEY in this shell first (read -rs META_API_KEY; export META_API_KEY).")
        self.model, self.timeout, self.log_path = model, timeout, log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def chat(self, messages: list, temperature: float = 0.2, max_tokens: int = 2000, response_format: dict = None) -> str:
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if response_format:
            body["response_format"] = response_format
        req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        t = time.time()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise SystemExit(f"Muse API error {e.code}: {e.read()[:500]!r}")
        text = out["choices"][0]["message"]["content"]
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": self.model,
                                 "seconds": round(time.time() - t, 2), "messages": messages, "reply": text,
                                 "usage": out.get("usage")}) + "\n")
        return text

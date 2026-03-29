import json
import time
import urllib.error
import urllib.request
from typing import Any


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def _post_json(self, path: str, payload: dict[str, Any], timeout: int | float | None = 300) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            if timeout is None or float(timeout) <= 0:
                resp_ctx = urllib.request.urlopen(req)
            else:
                resp_ctx = urllib.request.urlopen(req, timeout=float(timeout))
            with resp_ctx as resp:
                data = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Could not connect to Ollama at http://127.0.0.1:11434") from exc

        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned non-JSON response") from exc

    def _get_json(self, path: str, timeout: int = 30) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError("Could not connect to Ollama at http://127.0.0.1:11434") from exc
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned non-JSON response") from exc

    def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        *,
        prior_messages: list[dict[str, str]] | None = None,
        user_images: list[str] | None = None,
        temperature: float = 0.3,
        num_ctx: int = 8192,
        think: bool | None = None,
        timeout: int = 300,
        retry_attempts: int = 1,
        retry_backoff_sec: float = 1.25,
        fallback_models: list[str] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if prior_messages:
            for item in prior_messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip().lower()
                content = str(item.get("content", "")).strip()
                if role not in {"user", "assistant"} or not content:
                    continue
                messages.append({"role": role, "content": content})
        user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
        if user_images:
            clean_images = [str(x).strip() for x in user_images if str(x).strip()]
            if clean_images:
                user_message["images"] = clean_images
        messages.append(user_message)

        attempts = max(1, int(retry_attempts))
        backoff = max(0.0, float(retry_backoff_sec))

        models: list[str] = []
        for name in [model, *(fallback_models or [])]:
            key = str(name or "").strip()
            if not key or key in models:
                continue
            models.append(key)
        if not models:
            raise RuntimeError("No model specified.")

        errors: list[str] = []
        for model_name in models:
            payload: dict[str, Any] = {
                "model": model_name,
                "stream": False,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                },
            }
            if think is not None:
                payload["think"] = think

            for attempt in range(1, attempts + 1):
                try:
                    response = self._post_json("/api/chat", payload, timeout=timeout)
                    message = response.get("message") or {}
                    content = message.get("content")
                    if not isinstance(content, str):
                        raise RuntimeError("Ollama response missing message content")
                    clean = content.strip()
                    if not clean:
                        raise RuntimeError("Ollama returned empty message content")
                    return clean
                except Exception as exc:
                    errors.append(f"{model_name} attempt {attempt}/{attempts}: {exc}")
                    if attempt < attempts and backoff > 0:
                        sleep_sec = backoff * (1.0 + (attempt - 1) * 0.5)
                        time.sleep(sleep_sec)
                    continue

        tail = " | ".join(errors[-6:]) if errors else "unknown failure"
        raise RuntimeError(f"Ollama chat failed after retries/fallbacks: {tail}")

    def embed(self, model: str, text: str, *, timeout: int = 60) -> list[float]:
        response = self._post_json("/api/embed", {"model": model, "input": text}, timeout=timeout)
        embeddings = response.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if isinstance(first, list):
                return [float(x) for x in first]
        raise RuntimeError("embed: unexpected response format")

    def list_local_models(self) -> list[str]:
        data = self._get_json("/api/tags")
        models = data.get("models") or []
        names: list[str] = []
        for item in models:
            name = item.get("name")
            if isinstance(name, str):
                names.append(name)
        return names

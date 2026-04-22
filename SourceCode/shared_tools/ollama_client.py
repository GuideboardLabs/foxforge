import json
import logging
import time
import urllib.error
import urllib.request
from threading import Lock
from typing import Any

LOGGER = logging.getLogger(__name__)


class OllamaClient:
    _HEALTH_LOCK = Lock()
    _MODEL_HEALTH: dict[str, dict[str, float]] = {}
    _FAIL_WINDOW_SEC = 60.0
    _DECAY_WINDOW_SEC = 90.0

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
        num_predict: int | None = -1,
        timeout: int = 300,
        retry_attempts: int = 1,
        retry_backoff_sec: float = 1.25,
        fallback_models: list[str] | None = None,
        keep_alive: str = "10m",
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
        primary_model = models[0]
        if self._is_degraded(primary_model) and len(models) > 1:
            models = models[1:] + [primary_model]

        errors: list[str] = []
        try:
            predict = int(num_predict) if num_predict is not None else -1
        except (TypeError, ValueError):
            predict = -1
        if predict == 0:
            predict = -1
        for model_name in models:
            payload: dict[str, Any] = {
                "model": model_name,
                "stream": False,
                "messages": messages,
                "keep_alive": str(keep_alive) if keep_alive is not None else "10m",
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": predict,
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
                    self._record_success(model_name)
                    return clean
                except Exception as exc:
                    self._record_failure(model_name)
                    errors.append(f"{model_name} attempt {attempt}/{attempts}: {exc}")
                    if attempt < attempts and backoff > 0:
                        sleep_sec = backoff * (1.0 + (attempt - 1) * 0.5)
                        time.sleep(sleep_sec)
                    continue

        tail = " | ".join(errors[-6:]) if errors else "unknown failure"
        raise RuntimeError(f"Ollama chat failed after retries/fallbacks: {tail}")

    def wait_for_available(
        self,
        model: str,
        *,
        max_wait_sec: int = 300,
        poll_interval_sec: int = 15,
    ) -> bool:
        """Poll until the model responds to a trivial ping or timeout expires."""
        name = str(model or "").strip()
        if not name:
            return False
        elapsed = 0
        while elapsed < max_wait_sec:
            try:
                self.chat(
                    model=name,
                    system_prompt="",
                    user_prompt="ping",
                    num_predict=1,
                    timeout=10,
                    retry_attempts=1,
                    retry_backoff_sec=0.0,
                )
                return True
            except Exception:
                LOGGER.debug(
                    "wait_for_available: model=%r not responding, elapsed=%ds",
                    name,
                    elapsed,
                )
                time.sleep(poll_interval_sec)
                elapsed += poll_interval_sec
        return False

    def release_model(self, model: str) -> None:
        """Tell Ollama to immediately unload a model from VRAM (keep_alive=0)."""
        try:
            self._post_json(
                "/api/generate",
                {"model": model, "prompt": "", "keep_alive": 0},
                timeout=10,
            )
        except Exception:
            pass

    def ping(self, model: str, *, timeout: int = 8) -> bool:
        name = str(model or "").strip()
        if not name:
            return False
        try:
            self._post_json("/api/show", {"name": name}, timeout=timeout)
            self._record_success(name)
            return True
        except Exception:
            self._record_failure(name)
            return False

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

    def get_status(self) -> dict[str, dict[str, Any]]:
        now = time.time()
        out: dict[str, dict[str, Any]] = {}
        with self._HEALTH_LOCK:
            for model, state in self._MODEL_HEALTH.items():
                last_ok_ts = float(state.get("last_ok_ts", 0.0) or 0.0)
                last_fail_ts = float(state.get("last_fail_ts", 0.0) or 0.0)
                failures = int(state.get("consecutive_failures", 0) or 0)
                degraded = failures >= 2 and (now - last_fail_ts) <= self._DECAY_WINDOW_SEC
                out[model] = {
                    "last_ok_ts": last_ok_ts,
                    "last_fail_ts": last_fail_ts,
                    "consecutive_failures": failures,
                    "degraded": degraded,
                }
        return out

    @classmethod
    def _record_success(cls, model: str) -> None:
        key = str(model or "").strip()
        if not key:
            return
        now = time.time()
        with cls._HEALTH_LOCK:
            state = dict(cls._MODEL_HEALTH.get(key, {}))
            state["last_ok_ts"] = now
            state["consecutive_failures"] = 0
            cls._MODEL_HEALTH[key] = state

    @classmethod
    def _record_failure(cls, model: str) -> None:
        key = str(model or "").strip()
        if not key:
            return
        now = time.time()
        with cls._HEALTH_LOCK:
            state = dict(cls._MODEL_HEALTH.get(key, {}))
            last_fail = float(state.get("last_fail_ts", 0.0) or 0.0)
            prev_failures = int(state.get("consecutive_failures", 0) or 0)
            if last_fail > 0 and (now - last_fail) <= cls._FAIL_WINDOW_SEC:
                failures = prev_failures + 1
            else:
                failures = 1
            state["last_fail_ts"] = now
            state["consecutive_failures"] = failures
            cls._MODEL_HEALTH[key] = state

    @classmethod
    def _is_degraded(cls, model: str) -> bool:
        key = str(model or "").strip()
        if not key:
            return False
        now = time.time()
        with cls._HEALTH_LOCK:
            state = dict(cls._MODEL_HEALTH.get(key, {}))
            if not state:
                return False
            last_fail = float(state.get("last_fail_ts", 0.0) or 0.0)
            failures = int(state.get("consecutive_failures", 0) or 0)
            if failures < 2:
                return False
            if last_fail <= 0 or (now - last_fail) > cls._DECAY_WINDOW_SEC:
                state["consecutive_failures"] = 0
                cls._MODEL_HEALTH[key] = state
                return False
            return True

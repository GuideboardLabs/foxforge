from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class ComfyUIClient:
    """Minimal urllib-only client for ComfyUI's HTTP API.

    Supports the prompt-queue / history-poll / view-download cycle used
    to generate images from a pre-defined workflow template.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self.base_url = base_url.rstrip("/")
        self._client_id = "foxforge"

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _post_json(self, path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ComfyUI HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not connect to ComfyUI at {self.base_url}") from exc

    def _get_json(self, path: str, timeout: int = 30) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"ComfyUI HTTP {exc.code} on GET {path}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not connect to ComfyUI at {self.base_url}") from exc

    def _get_bytes(self, path: str, timeout: int = 60) -> bytes:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"ComfyUI HTTP {exc.code} fetching image") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not retrieve image from ComfyUI at {self.base_url}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self, timeout: int = 5) -> bool:
        """Return True if ComfyUI is reachable."""
        try:
            self._get_json("/queue", timeout=timeout)
            return True
        except Exception:
            return False

    def queue_info(self, timeout: int = 10) -> dict[str, Any]:
        """Return ComfyUI queue snapshot."""
        return self._get_json("/queue", timeout=timeout)

    def interrupt(self, prompt_id: str | None = None) -> None:
        """Best-effort interrupt for the current running prompt."""
        try:
            payload = {"prompt_id": str(prompt_id).strip()} if str(prompt_id or "").strip() else {}
            self._post_json("/interrupt", payload, timeout=10)
        except Exception:
            pass

    def clear_queue(self) -> None:
        """Best-effort queue clear to avoid stale blocked prompts."""
        try:
            self._post_json("/queue", {"clear": True}, timeout=10)
        except Exception:
            pass

    def generate(
        self,
        workflow: dict[str, Any],
        *,
        poll_interval: float = 1.5,
        timeout: int = 300,
    ) -> bytes:
        """Queue a workflow and block until the first output image is ready.

        Returns the raw PNG bytes of the generated image.
        Raises RuntimeError on timeout or any ComfyUI error.
        """
        resp = self._post_json("/prompt", {"prompt": workflow, "client_id": self._client_id})
        prompt_id: str = str(resp.get("prompt_id", "")).strip()
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return a prompt_id: {resp}")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            try:
                history = self._get_json(f"/history/{prompt_id}")
            except Exception:
                continue

            entry = history.get(prompt_id)
            if not entry:
                continue

            # ComfyUI sets status.completed when done
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                self.interrupt(prompt_id)
                self.clear_queue()
                raise RuntimeError(f"ComfyUI workflow error: {msgs}")

            outputs = entry.get("outputs", {})
            for node_output in outputs.values():
                images = node_output.get("images", [])
                if images:
                    img_info = images[0]
                    filename = img_info.get("filename", "")
                    subfolder = img_info.get("subfolder", "")
                    img_type = img_info.get("type", "output")
                    params = f"filename={urllib.request.quote(filename)}&subfolder={urllib.request.quote(subfolder)}&type={img_type}"
                    return self._get_bytes(f"/view?{params}")

        # Timeout can leave ComfyUI with a stale running prompt. Clear it so
        # later image requests do not get wedged behind a zombie run.
        self.interrupt(prompt_id)
        self.clear_queue()
        raise RuntimeError(f"ComfyUI generation timed out after {timeout}s (prompt_id={prompt_id})")

    def unload_models(self) -> None:
        """Ask ComfyUI to free loaded models from VRAM (best-effort)."""
        try:
            self._post_json("/free", {"unload_models": True, "free_memory": True}, timeout=10)
        except Exception:
            pass

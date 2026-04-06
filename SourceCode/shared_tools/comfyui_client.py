from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
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

    @staticmethod
    def _string_choices(value: Any) -> list[str]:
        out: list[str] = []

        def _walk(node: Any) -> None:
            if isinstance(node, str):
                text = node.strip()
                if text:
                    out.append(text)
                return
            if isinstance(node, dict):
                for child in node.values():
                    _walk(child)
                return
            if isinstance(node, (list, tuple, set)):
                for child in node:
                    _walk(child)

        _walk(value)
        return out

    def _choices_from_object_info(self, payload: dict[str, Any], field_name: str) -> list[str]:
        if not isinstance(payload, dict):
            return []
        candidates: list[str] = []
        targets: list[dict[str, Any]] = [payload]
        targets.extend([v for v in payload.values() if isinstance(v, dict)])
        for node in targets:
            inputs = node.get("input")
            if not isinstance(inputs, dict):
                continue
            required = inputs.get("required")
            if not isinstance(required, dict):
                continue
            field = required.get(field_name)
            if field is None:
                continue
            candidates.extend(self._string_choices(field))
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            text = str(item).strip()
            low = text.lower()
            if not text:
                continue
            if low in {"string", "none", "required", "optional", "lora_name"}:
                continue
            if text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        return cleaned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_image(self, image_path: str, timeout: int = 30) -> str:
        """Upload a local image file to ComfyUI's input directory.

        Returns the filename ComfyUI assigned (use this in LoadImage nodes).
        Raises RuntimeError on failure.
        """
        import mimetypes
        import uuid
        path_bytes = open(image_path, "rb").read()
        filename = image_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8") + path_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
        url = f"{self.base_url}/upload/image"
        req = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            uploaded_name = str(data.get("name", "")).strip()
            if not uploaded_name:
                raise RuntimeError(f"ComfyUI upload response missing name: {data}")
            return uploaded_name
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ComfyUI upload failed HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not connect to ComfyUI for upload at {self.base_url}") from exc

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

    def list_loras(self, timeout: int = 15) -> list[str]:
        """Return available LoRA filenames from ComfyUI."""
        names: list[str] = []
        for path in ("/object_info/LoraLoader", "/object_info/LoraLoaderModelOnly", "/object_info"):
            try:
                payload = self._get_json(path, timeout=timeout)
            except Exception:
                continue
            choices = self._choices_from_object_info(payload, "lora_name")
            if not choices:
                continue
            names.extend(choices)

        cleaned: list[str] = []
        seen: set[str] = set()
        for item in names:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        cleaned.sort(key=lambda row: row.lower())
        return cleaned

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

    def generate_video(
        self,
        workflow: dict[str, Any],
        *,
        poll_interval: float = 3.0,
        timeout: int = 900,
    ) -> bytes:
        """Queue a workflow and block until the first output video is ready.

        Returns the raw MP4 bytes of the generated video.
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

            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                self.interrupt(prompt_id)
                self.clear_queue()
                raise RuntimeError(f"ComfyUI workflow error: {msgs}")

            outputs = entry.get("outputs", {})
            for node_output in outputs.values():
                gifs = node_output.get("gifs", [])
                if gifs:
                    vid_info = gifs[0]
                    filename = vid_info.get("filename", "")
                    subfolder = vid_info.get("subfolder", "")
                    vid_type = vid_info.get("type", "output")
                    params = (
                        f"filename={urllib.request.quote(filename)}"
                        f"&subfolder={urllib.request.quote(subfolder)}"
                        f"&type={vid_type}"
                    )
                    return self._get_bytes(f"/view?{params}", timeout=120)

        self.interrupt(prompt_id)
        self.clear_queue()
        raise RuntimeError(f"ComfyUI video generation timed out after {timeout}s (prompt_id={prompt_id})")

    def unload_models(self) -> None:
        """Ask ComfyUI to free loaded models from VRAM (best-effort)."""
        try:
            self._post_json("/free", {"unload_models": True, "free_memory": True}, timeout=10)
        except Exception:
            pass

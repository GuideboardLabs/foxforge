"""Inference router that dispatches model calls to Ollama or llama.cpp servers.

Drop-in replacement for OllamaClient — same .chat() / .embed() / .list_local_models() API.
When ``llama_cpp_servers`` is configured in model_routing.json, matching models are routed
to a TurboQuant-enabled llama.cpp server; everything else goes through Ollama as usual.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from shared_tools.llamacpp_client import LlamaCppClient
from shared_tools.model_routing import load_model_routing
from shared_tools.ollama_client import OllamaClient

LOGGER = logging.getLogger(__name__)


class InferenceRouter:
    def __init__(self, repo_root: Path | None = None) -> None:
        self._ollama = OllamaClient()
        self._llama_clients: dict[str, LlamaCppClient] = {}
        self._model_map: dict[str, str] = {}  # model_name -> server_key
        self._fallback_flags: dict[str, bool] = {}  # server_key -> fallback_to_ollama

        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[2]

        routing = load_model_routing(repo_root)
        servers = routing.get("llama_cpp_servers")
        if not isinstance(servers, dict):
            return

        for key, cfg in servers.items():
            if not isinstance(cfg, dict):
                continue
            base_url = str(cfg.get("base_url", "")).strip()
            if not base_url:
                continue
            models = cfg.get("models", [])
            if not isinstance(models, list):
                continue
            self._llama_clients[key] = LlamaCppClient(base_url)
            self._fallback_flags[key] = bool(cfg.get("fallback_to_ollama", True))
            for model_name in models:
                name = str(model_name).strip()
                if name:
                    self._model_map[name] = key

        if self._model_map:
            LOGGER.info("InferenceRouter: llama.cpp routing active for %s", list(self._model_map.keys()))

    def _client_for_model(self, model: str) -> OllamaClient | LlamaCppClient:
        server_key = self._model_map.get(model)
        if server_key and server_key in self._llama_clients:
            return self._llama_clients[server_key]
        return self._ollama

    def _should_fallback(self, model: str) -> bool:
        server_key = self._model_map.get(model, "")
        return self._fallback_flags.get(server_key, False)

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
        client = self._client_for_model(model)
        kwargs: dict[str, Any] = dict(
            prior_messages=prior_messages,
            user_images=user_images,
            temperature=temperature,
            num_ctx=num_ctx,
            think=think,
            timeout=timeout,
            retry_attempts=retry_attempts,
            retry_backoff_sec=retry_backoff_sec,
            fallback_models=fallback_models,
        )
        if client is not self._ollama:
            try:
                return client.chat(model, system_prompt, user_prompt, **kwargs)
            except RuntimeError:
                if self._should_fallback(model):
                    LOGGER.warning("llama.cpp server failed for %s, falling back to Ollama", model)
                    return self._ollama.chat(model, system_prompt, user_prompt, **kwargs)
                raise
        return self._ollama.chat(model, system_prompt, user_prompt, **kwargs)

    def embed(self, model: str, text: str, *, timeout: int = 60) -> list[float]:
        return self._ollama.embed(model, text, timeout=timeout)

    def list_local_models(self) -> list[str]:
        models = self._ollama.list_local_models()
        for model_name in self._model_map:
            if model_name not in models:
                models.append(model_name)
        return models

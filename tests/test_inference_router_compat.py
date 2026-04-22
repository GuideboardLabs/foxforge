from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from tests.common import ROOT  # noqa: F401
from shared_tools.inference_router import InferenceRouter


class InferenceRouterCompatTests(unittest.TestCase):
    def test_chat_forwards_num_predict_to_ollama(self) -> None:
        router = InferenceRouter(ROOT)
        router._model_map = {}
        router._ollama = MagicMock()
        router._ollama.chat.return_value = "ok"

        out = router.chat(
            model="qwen3:8b",
            system_prompt="sys",
            user_prompt="user",
            num_predict=321,
            retry_attempts=1,
        )

        self.assertEqual(out, "ok")
        kwargs = router._ollama.chat.call_args.kwargs
        self.assertEqual(kwargs.get("num_predict"), 321)

    def test_chat_forwards_num_predict_to_llama_cpp(self) -> None:
        router = InferenceRouter(ROOT)
        router._model_map = {"qwen3:8b": "llama_srv"}
        router._llama_clients = {"llama_srv": MagicMock()}
        router._fallback_flags = {"llama_srv": False}
        router._llama_clients["llama_srv"].chat.return_value = "ok"
        router._server_declares_model = MagicMock(return_value=True)

        out = router.chat(
            model="qwen3:8b",
            system_prompt="sys",
            user_prompt="user",
            num_predict=777,
            retry_attempts=1,
        )

        self.assertEqual(out, "ok")
        kwargs = router._llama_clients["llama_srv"].chat.call_args.kwargs
        self.assertEqual(kwargs.get("num_predict"), 777)

    def test_wait_for_available_uses_compatible_chat_signature(self) -> None:
        router = InferenceRouter(ROOT)
        router.chat = MagicMock(return_value="pong")

        self.assertTrue(
            router.wait_for_available(
                "qwen3:8b",
                fallback_models=["deepseek-r1:8b"],
                max_wait_sec=1,
                poll_interval_sec=1,
            )
        )
        kwargs = router.chat.call_args.kwargs
        self.assertEqual(kwargs.get("fallback_models"), ["deepseek-r1:8b"])
        self.assertGreaterEqual(int(kwargs.get("timeout", 0) or 0), 30)
        self.assertFalse(bool(kwargs.get("think", True)))

    def test_wait_for_available_treats_empty_ping_as_available(self) -> None:
        router = InferenceRouter(ROOT)
        router.chat = MagicMock(side_effect=RuntimeError("Ollama returned empty message content"))
        self.assertTrue(
            router.wait_for_available(
                "qwen3:8b",
                fallback_models=["deepseek-r1:8b"],
                max_wait_sec=1,
                poll_interval_sec=1,
            )
        )


if __name__ == "__main__":
    unittest.main()

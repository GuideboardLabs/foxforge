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

    def test_wait_for_available_uses_llama_declaration_without_chat(self) -> None:
        router = InferenceRouter(ROOT)
        router._server_declares_model = MagicMock(side_effect=lambda m: m == "deepseek-r1:8b")
        router._ollama = MagicMock()
        router.chat = MagicMock(side_effect=AssertionError("preflight must not call chat"))

        self.assertTrue(
            router.wait_for_available(
                "qwen3:8b",
                fallback_models=["deepseek-r1:8b"],
                max_wait_sec=1,
                poll_interval_sec=1,
            )
        )
        router.chat.assert_not_called()
        router._ollama.wait_for_available.assert_not_called()

    def test_wait_for_available_delegates_to_ollama_without_chat(self) -> None:
        router = InferenceRouter(ROOT)
        router._server_declares_model = MagicMock(return_value=False)
        router._ollama = MagicMock()
        router._ollama.wait_for_available.return_value = True
        router._ollama.last_wait_polls = 2
        router._ollama.last_wait_error = ""
        router.chat = MagicMock(side_effect=AssertionError("preflight must not call chat"))

        self.assertTrue(
            router.wait_for_available(
                "qwen3:8b",
                fallback_models=["deepseek-r1:8b"],
                max_wait_sec=1,
                poll_interval_sec=1,
            )
        )
        router.chat.assert_not_called()
        kwargs = router._ollama.wait_for_available.call_args.kwargs
        self.assertEqual(router._ollama.wait_for_available.call_args.args[0], "qwen3:8b")
        self.assertEqual(kwargs.get("fallback_models"), ["deepseek-r1:8b"])


if __name__ == "__main__":
    unittest.main()

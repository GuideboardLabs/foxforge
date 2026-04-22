from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.synthesizer import SynthesisUnavailableError, synthesize


def _valid_summary() -> str:
    return (
        "# Research Synthesis\n\n"
        "## Executive Summary\n"
        "Evidence Confidence: Mixed — primary model unavailable but fallback was used. "
        "This summary intentionally includes enough detail to pass synthesis validation. "
        "It captures concrete recall-training guidance, risk framing, and practical sequencing.\n\n"
        "## Key Findings\n"
        "- Positive reinforcement outperforms aversive methods for reliable recall when training is consistent.\n"
        "- Short, frequent sessions with controlled distractions improve compliance over time.\n"
        "- Long-line safety practice reduces risk during early outdoor generalization.\n\n"
        "## Uncertainties & Risks\n"
        "- Some recommendations come from practitioner guidance rather than randomized trials.\n"
        "- Household consistency remains a major risk factor for regressions in recall reliability.\n\n"
        "## Next Steps\n"
        "- Standardize cue wording across handlers.\n"
        "- Run three short sessions daily and log outcomes.\n"
        "- Increase distraction complexity only after stable success.\n"
    )


class _FallbackAwareClient:
    def __init__(self) -> None:
        self.wait_calls: list[dict] = []

    def wait_for_available(self, model: str, **kwargs) -> bool:
        self.wait_calls.append({"model": model, **kwargs})
        return bool(kwargs.get("fallback_models"))

    def chat(self, **_kwargs) -> str:
        return _valid_summary()


class _NoModelClient:
    def wait_for_available(self, model: str, **_kwargs) -> bool:
        return False

    def chat(self, **_kwargs) -> str:
        raise RuntimeError("should not be called")


class _NoModelClientWithReason:
    def __init__(self) -> None:
        self.last_wait_error = "RuntimeError: timed out waiting for llama runner to start: context canceled"

    def wait_for_available(self, model: str, **_kwargs) -> bool:
        return False

    def chat(self, **_kwargs) -> str:
        raise RuntimeError("should not be called")


class _ChatFailClient:
    def wait_for_available(self, model: str, **_kwargs) -> bool:
        return True

    def chat(self, **_kwargs) -> str:
        raise RuntimeError("backend down")


class SynthesizerPreflightFallbackTests(unittest.TestCase):
    def test_preflight_wait_includes_fallback_models(self) -> None:
        client = _FallbackAwareClient()
        out = synthesize(
            "question",
            [{"agent": "a1", "finding": "finding"}],
            client=client,
            model_cfg={
                "model": "qwen3:8b",
                "synthesis_fallback_models": ["deepseek-r1:8b"],
                "synthesis_validation_cycles": 1,
                "synthesis_retry_attempts": 1,
            },
        )
        self.assertIn("Executive Summary", out)
        self.assertEqual(len(client.wait_calls), 1)
        self.assertEqual(client.wait_calls[0].get("fallback_models"), ["deepseek-r1:8b"])

    def test_preflight_error_mentions_all_candidates(self) -> None:
        with self.assertRaises(SynthesisUnavailableError) as ctx:
            synthesize(
                "question",
                [{"agent": "a1", "finding": "finding"}],
                client=_NoModelClient(),
                model_cfg={
                    "model": "qwen3:8b",
                    "synthesis_fallback_models": ["deepseek-r1:8b", "qwen2.5-coder:7b"],
                    "synthesis_validation_cycles": 1,
                    "synthesis_retry_attempts": 1,
                },
            )
        msg = str(ctx.exception)
        self.assertIn("qwen3:8b", msg)
        self.assertIn("deepseek-r1:8b", msg)
        self.assertIn("qwen2.5-coder:7b", msg)

    def test_preflight_error_includes_last_wait_error_when_available(self) -> None:
        with self.assertRaises(SynthesisUnavailableError) as ctx:
            synthesize(
                "question",
                [{"agent": "a1", "finding": "finding"}],
                client=_NoModelClientWithReason(),
                model_cfg={
                    "model": "qwen3:8b",
                    "synthesis_fallback_models": ["deepseek-r1:8b"],
                    "synthesis_validation_cycles": 1,
                    "synthesis_retry_attempts": 1,
                },
            )
        msg = str(ctx.exception).lower()
        self.assertIn("last error", msg)
        self.assertIn("context canceled", msg)

    def test_generation_unavailable_surfaces_last_error(self) -> None:
        with self.assertRaises(SynthesisUnavailableError) as ctx:
            synthesize(
                "question",
                [{"agent": "a1", "finding": "finding"}],
                client=_ChatFailClient(),
                model_cfg={
                    "model": "qwen3:8b",
                    "synthesis_fallback_models": ["deepseek-r1:8b"],
                    "synthesis_validation_cycles": 1,
                    "synthesis_retry_attempts": 1,
                },
            )
        msg = str(ctx.exception).lower()
        self.assertIn("last error", msg)
        self.assertIn("backend down", msg)


if __name__ == "__main__":
    unittest.main()

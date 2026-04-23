from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.synthesizer import SynthesisUnavailableError, run_skeptic_pass, synthesize


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


class _TapClient:
    def __init__(self) -> None:
        self.wait_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.last_wait_error = ""

    def wait_for_available(self, model: str, **kwargs) -> bool:
        self.wait_calls.append({"model": model, **kwargs})
        return True

    def chat(self, **kwargs) -> str:
        self.chat_calls.append(dict(kwargs))
        return _valid_summary()


class _SkepticNoLinkClient:
    def chat(self, **_kwargs) -> str:
        return (
            "<REVISED_SUMMARY>\n"
            "# Research Synthesis\n\n"
            "## Executive Summary\n"
            "Evidence Confidence: Mixed.\n\n"
            "## Key Findings\n"
            "- [E] Evidence noted.\n\n"
            "## Uncertainties & Risks\n"
            "- Caveat.\n\n"
            "## Next Steps\n"
            "- Proceed.\n"
            "</REVISED_SUMMARY>\n"
            "<CRITIQUE_LOG>ok</CRITIQUE_LOG>"
        )


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

    def test_preflight_wait_for_available_never_requires_direct_chat(self) -> None:
        client = _TapClient()
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
        # Exactly one synthesis generation call should occur for this happy path.
        self.assertEqual(len(client.chat_calls), 1)
        self.assertEqual(str(client.chat_calls[0].get("model", "")).strip(), "qwen3:8b")
        self.assertIn(
            "include an inline markdown URL citation",
            str(client.chat_calls[0].get("system_prompt", "")),
        )

    def test_synthesize_adds_inline_links_from_raw_findings_when_missing(self) -> None:
        client = _TapClient()
        out = synthesize(
            "question",
            [
                {
                    "agent": "a1",
                    "finding": (
                        "## Findings\n"
                        "- [E] Strong claim [source: https://avma.org/resources-tools/pet-owners]\n"
                    ),
                }
            ],
            client=client,
            model_cfg={
                "model": "qwen3:8b",
                "synthesis_fallback_models": ["deepseek-r1:8b"],
                "synthesis_validation_cycles": 1,
                "synthesis_retry_attempts": 1,
            },
        )
        self.assertIn("Source Anchors", out)
        self.assertIn("(https://avma.org/resources-tools/pet-owners)", out)

    def test_skeptic_pass_adds_inline_links_from_raw_findings_when_missing(self) -> None:
        revised, critique = run_skeptic_pass(
            question="q",
            synthesis=_valid_summary(),
            client=_SkepticNoLinkClient(),
            model_cfg={"model": "dummy-model"},
            findings=[
                {
                    "agent": "a1",
                    "finding": "[E] Raw evidence [source: https://aspca.org/pet-care]",
                }
            ],
        )
        self.assertEqual(critique, "ok")
        self.assertIn("Source Anchors", revised)
        self.assertIn("(https://aspca.org/pet-care)", revised)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.common import ROOT  # noqa: F401
from agents_research.deep_researcher import SynthesisUnavailableError, run_research_pool


class _Bus:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def emit(self, actor: str, event: str, details: dict) -> None:
        self.events.append((actor, event, details))


class _DummyClient:
    def release_models(self, _models: list[str]) -> None:
        return None


class _DummyLearning:
    def __init__(self, *_args, **_kwargs) -> None:
        return None

    def guidance_for_lane(self, *_args, **_kwargs) -> str:
        return ""


class ResearchPoolSynthesisUnavailableTests(unittest.TestCase):
    @patch("agents_research.deep_researcher.synthesize", side_effect=SynthesisUnavailableError("test unavailable"))
    @patch("agents_research.deep_researcher.build_retrieved_chunks", return_value=[])
    @patch("agents_research.deep_researcher._audit_evidence_labels", side_effect=lambda rows: rows)
    @patch("agents_research.deep_researcher._self_check", return_value=3)
    @patch(
        "agents_research.deep_researcher._run_multipass_agent",
        return_value={
            "agent": "critical_analyst",
            "model": "dummy-model",
            "requested_model": "dummy-model",
            "finding": "## Findings\n- [E] Example evidence [source: https://example.com]\n\n## Evidence Signals\n- signal\n\n## Open Questions\n- question",
        },
    )
    @patch(
        "agents_research.deep_researcher._agent_specs",
        return_value=[{"persona": "critical_analyst", "model": "dummy-model", "directive": "focus", "role": "primary"}],
    )
    @patch("agents_research.deep_researcher.FeedbackLearningEngine", _DummyLearning)
    @patch("agents_research.deep_researcher.InferenceRouter", return_value=_DummyClient())
    @patch(
        "agents_research.deep_researcher.lane_model_config",
        side_effect=lambda _repo_root, lane: (
            {"parallel_agents": 1, "model": "dummy-model", "retry_attempts": 1, "validation_cycles": 1}
            if lane == "research_pool"
            else {"model": "dummy-model"}
        ),
    )
    def test_returns_clean_synthesis_unavailable_result(self, *_mocks) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = run_research_pool(
                question="test question",
                repo_root=Path(td),
                project_slug="test_project",
                bus=_Bus(),
                web_context="",
            )
        self.assertTrue(bool(out.get("synthesis_unavailable", False)))
        self.assertEqual(str(out.get("summary_path", "")), "")
        self.assertIn("synthesis model was unavailable", str(out.get("message", "")).lower())
        raw_path = str(out.get("raw_path", ""))
        self.assertTrue(raw_path.endswith("_research_raw.md"))
        self.assertIn(raw_path, str(out.get("message", "")))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_make.content_pool import _run_compositor as content_run_compositor
from agents_make.content_pool import _run_planner as content_run_planner
from agents_make.longform_pool import _run_compositor as longform_run_compositor
from agents_make.longform_pool import _run_planner as longform_run_planner


class _CaptureClient:
    def __init__(self, result: str = "ok") -> None:
        self.result = result
        self.last_system_prompt = ""

    def chat(self, **kwargs):
        self.last_system_prompt = str(kwargs.get("system_prompt", ""))
        return self.result


class PublicContentGuardrailTests(unittest.TestCase):
    def test_content_pool_planner_and_compositor_include_guardrail(self) -> None:
        client = _CaptureClient("plan")
        content_run_planner(
            client=client,
            question="Write a blog post",
            kind="blog",
            research_context="",
            project_context="",
        )
        self.assertIn("PUBLIC-CONTENT GUARDRAIL", client.last_system_prompt)

        content_run_compositor(
            client=client,
            sections=[("Hook", "Example section body")],
            kind="blog",
            question="Write a blog post",
        )
        self.assertIn("PUBLIC-CONTENT GUARDRAIL", client.last_system_prompt)

    def test_longform_pool_planner_and_compositor_include_guardrail(self) -> None:
        client = _CaptureClient("plan")
        longform_run_planner(
            client=client,
            question="Write a guide",
            type_id="guide",
            sections=[("Overview", "Thesis")],
            research_context="",
            project_context="",
        )
        self.assertIn("PUBLIC-CONTENT GUARDRAIL", client.last_system_prompt)

        longform_run_compositor(
            client=client,
            question="Write a guide",
            type_id="guide",
            sections=[("Overview", "Section body")],
            research_context="",
        )
        self.assertIn("PUBLIC-CONTENT GUARDRAIL", client.last_system_prompt)


if __name__ == "__main__":
    unittest.main()

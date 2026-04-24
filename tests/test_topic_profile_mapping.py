from __future__ import annotations

import unittest
from unittest import mock

from tests.common import ROOT  # noqa: F401  # ensure SourceCode on sys.path
from agents_research.deep_researcher import (
    ANALYSIS_PROFILE_ANIMAL_CARE,
    ANALYSIS_PROFILE_GENERAL,
    TOPIC_TYPE_TO_PROFILE,
    _analysis_profile_for_type,
    _profile_agent_templates,
)
from shared_tools.fact_policy import detect_topic_type


class TopicProfileMappingTests(unittest.TestCase):
    def test_no_topic_aliases_any_profile(self) -> None:
        self.assertEqual(len(set(TOPIC_TYPE_TO_PROFILE.values())), len(TOPIC_TYPE_TO_PROFILE))

    def test_animal_care_does_not_use_human_medical_personas(self) -> None:
        templates = _profile_agent_templates(ANALYSIS_PROFILE_ANIMAL_CARE)
        self.assertTrue(templates)
        blocked_personas = {"clinical_evidence_researcher", "safety_risk_researcher", "guideline_verifier"}
        persona_names = {str(row.get("persona", "")).strip() for row in templates}
        self.assertFalse(persona_names & blocked_personas)
        self.assertTrue(
            any(name.startswith("veterinary_") or name.startswith("species_") for name in persona_names),
            "animal care profile should include species/veterinary personas",
        )

    def test_every_mapped_profile_has_templates(self) -> None:
        for topic_type, profile in TOPIC_TYPE_TO_PROFILE.items():
            with self.subTest(topic_type=topic_type, profile=profile):
                templates = _profile_agent_templates(profile)
                self.assertTrue(templates)

    def test_every_template_directive_has_domain_guardrail(self) -> None:
        guard_tokens = (
            "stay in domain",
            "focus on",
            "focus exclusively",
            "cross-check",
            "actively seek",
            "output contract",
        )
        for topic_type, profile in TOPIC_TYPE_TO_PROFILE.items():
            with self.subTest(topic_type=topic_type, profile=profile):
                templates = _profile_agent_templates(profile)
                for template in templates:
                    directive = str(template.get("directive", "")).strip()
                    self.assertTrue(directive)
                    directive_low = directive.lower()
                    self.assertTrue(any(token in directive_low for token in guard_tokens))

    def test_uncatalogued_topic_warns_and_falls_back_to_general(self) -> None:
        with mock.patch("agents_research.deep_researcher.LOGGER.warning") as warning_mock:
            with mock.patch("agents_research.deep_researcher.telemetry_emit") as telemetry_mock:
                resolved = _analysis_profile_for_type("frobnicate")
        self.assertEqual(resolved, ANALYSIS_PROFILE_GENERAL)
        warning_mock.assert_called_once()
        telemetry_mock.assert_called_once()

    def test_detect_topic_returns_only_mapped_keys(self) -> None:
        fixtures = [
            ("best protein and joint routine for my senior dog", "general"),
            ("UFC 318 main card odds and weight class breakdown", "sports"),
            ("Chiefs vs Bills spread and kickoff weather", "sports"),
            ("Help me with a go-to-market strategy", "business"),
        ]
        for query, topic_type in fixtures:
            with self.subTest(query=query, topic_type=topic_type):
                detected = detect_topic_type(query, topic_type=topic_type)
                self.assertIn(detected, TOPIC_TYPE_TO_PROFILE)


if __name__ == "__main__":
    unittest.main()

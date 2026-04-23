from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from shared_tools.answer_composer import compose_research_summary, format_source_label


class AnswerComposerSourcesTests(unittest.TestCase):
    def test_format_source_label_uses_hostname_for_noisy_long_domains(self) -> None:
        label = format_source_label({"url": "https://westernstatesk9college.com/training"})
        self.assertEqual(label, "westernstatesk9college.com")

    def test_format_source_label_keeps_known_domain_aliases(self) -> None:
        label = format_source_label({"url": "https://www.espn.com/mma/"})
        self.assertEqual(label, "ESPN")

    def test_format_source_label_keeps_common_vet_acronyms(self) -> None:
        self.assertEqual(format_source_label({"url": "https://www.avma.org/resources"}), "AVMA")
        self.assertEqual(format_source_label({"url": "https://www.aspca.org/pet-care"}), "ASPCA")
        self.assertEqual(format_source_label({"url": "https://www.petmd.com/dog"}), "PetMD")
        self.assertEqual(format_source_label({"url": "https://vcahospitals.com/know-your-pet"}), "VCA Hospitals")

    def test_compose_summary_uses_sources_emphasized_when_body_has_inline_links(self) -> None:
        text = (
            "## Event Overview\n"
            "Claim with inline citation ([ref](https://example.org/path)).\n\n"
            "## Bottom Line\n"
            "Done."
        )
        out = compose_research_summary(text, sources=[{"url": "https://example.org/a"}])
        self.assertIn("Sources emphasized: Example.", out)
        self.assertNotIn("Background domains consulted:", out)

    def test_compose_summary_uses_background_domains_when_body_has_no_inline_links(self) -> None:
        text = (
            "## Event Overview\n"
            "Summary without inline links.\n\n"
            "## Bottom Line\n"
            "Done."
        )
        out = compose_research_summary(text, sources=[{"url": "https://example.org/a"}])
        self.assertIn("Background domains consulted: Example.", out)
        self.assertNotIn("Sources emphasized:", out)


if __name__ == "__main__":
    unittest.main()

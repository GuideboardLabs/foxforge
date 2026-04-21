from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.citation_linker import build_retrieved_chunks, link


class CitationLinkerTests(unittest.TestCase):
    def test_build_retrieved_chunks_and_marker_linking(self) -> None:
        findings = [
            {
                "source_evidence": [
                    {"url": "https://example.com/a", "domain": "example.com", "snippet": "alpha beta context", "score": 0.8},
                    {"url": "https://example.com/b", "domain": "example.com", "snippet": "gamma delta context", "score": 0.7},
                ]
            }
        ]
        chunks = build_retrieved_chunks(findings)
        self.assertEqual(len(chunks), 2)

        payload = link(
            "First claim [S1]. Second claim [S2].",
            retrieved_chunks=chunks,
            threshold=0.45,
            embedding_client=None,
        )
        self.assertEqual(payload["type"], "research_reply")
        self.assertEqual(len(payload["sentences"]), 2)
        self.assertTrue(payload["sentences"][0]["citation_ids"])
        self.assertTrue(payload["sentences"][1]["citation_ids"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tests.common import ROOT
from infra.tools import ToolRegistry
from orchestrator.services.agent_contracts import AgentTask
from orchestrator.services.agent_registry import AppPoolAgent, ResearchPoolAgent, build_default_agent_registry


class AgentContractTests(unittest.TestCase):
    def test_default_agent_registry_exposes_capabilities(self) -> None:
        registry = build_default_agent_registry()
        lanes = set(registry.lanes())
        self.assertTrue({'research', 'project', 'make_app', 'make_doc', 'ui', 'make_tool'}.issubset(lanes))
        self.assertNotIn('personal', lanes)
        descriptions = registry.describe()
        self.assertTrue(any(row['supports_progress'] for row in descriptions if row['lane'] == 'research'))

    def test_research_agent_returns_worker_result_contract(self) -> None:
        tools = ToolRegistry()
        tools.register('bus', object())
        task = AgentTask(
            lane='research',
            prompt='Explain local-first architecture',
            project_slug='general',
            repo_root=Path(ROOT),
            history=[{'role': 'user', 'content': 'hi'}],
            context={'topic_type': 'technical', 'project_context': 'ctx', 'web_context': 'web'},
        )
        with patch('orchestrator.services.agent_registry.run_research_pool', return_value={'message': 'done', 'summary_path': 'x.md', 'canceled': False}) as runner:
            result = ResearchPoolAgent().run(task, tools)
        self.assertEqual(result.lane, 'research')
        self.assertEqual(result.summary_path, 'x.md')
        self.assertIn('summary_path', result.payload)
        runner.assert_called_once()

    def test_app_agent_uses_registry_and_normalizes_output(self) -> None:
        tools = ToolRegistry()
        tools.register('bus', object())
        task = AgentTask(
            lane='make_app',
            prompt='Build a habit tracker app',
            project_slug='general',
            repo_root=Path(ROOT),
            context={'project_context': 'ctx', 'research_context': 'notes'},
        )
        with patch('orchestrator.services.agent_registry.run_app_pool', return_value={'message': 'built', 'path': 'app/out.md'}) as runner:
            result = AppPoolAgent().run(task, tools)
        self.assertEqual(result.lane, 'make_app')
        self.assertIn('app/out.md', result.artifact_paths)
        runner.assert_called_once()


if __name__ == '__main__':
    unittest.main()

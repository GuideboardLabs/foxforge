from __future__ import annotations

import unittest
from pathlib import Path

from tests.common import ROOT, ensure_runtime
from infra.tools import ToolRegistry
from orchestrator.services.infra_runtime import OrchestratorInfraRuntime


class FakeOllama:
    pass


class ToolRegistryTests(unittest.TestCase):
    def test_basic_registry_contract(self) -> None:
        registry = ToolRegistry()
        registry.register('alpha', 123, description='demo')
        self.assertTrue(registry.has('alpha'))
        self.assertEqual(registry.require('alpha'), 123)
        self.assertEqual(registry.describe()[0]['description'], 'demo')
        self.assertRaises(KeyError, registry.require, 'missing')

    def test_infra_runtime_builds_shared_registry(self) -> None:
        ensure_runtime(ROOT)
        runtime = OrchestratorInfraRuntime(Path(ROOT), FakeOllama())
        registry = runtime.build_tool_registry(bus=object())
        names = set(registry.names())
        expected = {
            'ollama', 'web_engine', 'cloud_engine', 'project_memory', 'topic_memory',
            'pipeline_store', 'learning_engine', 'reflection_engine', 'workspace_tools',
            'embedding_memory', 'bus',
        }
        self.assertTrue(expected.issubset(names))


if __name__ == '__main__':
    unittest.main()

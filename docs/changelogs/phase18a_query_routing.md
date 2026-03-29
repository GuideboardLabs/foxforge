# Phase 18A — Query Routing and Adaptive Paths

## Summary
Added adaptive routing so the orchestrator can distinguish between simple factual requests, live-event requests, deeper research, workspace/code prompts, and legacy planning/home prompts.

## Changes
- Added `SourceCode/shared_tools/query_router.py`
- Added light-research path in `SourceCode/orchestrator/main.py`
- Added local prior-summary retrieval via `EmbeddingMemory`
- Added confidence gate for low-confidence answers

## Expected impact
- Faster answers for simple factual and live-event requests
- Less unnecessary full-pool foraging
- Better reuse of prior local summaries

## Accuracy guardrails
- Full foraging remains available for deep research
- Light path still uses grounded web snippets and explicit uncertainty handling

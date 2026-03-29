# Phase 18C — Confidence Gating and Memory Reuse

## Summary
Added lightweight confidence evaluation and local retrieval of prior summaries to reduce repeated work and improve transparency.

## Changes
- Added `evaluate_answer_confidence()` to `answer_composer.py`
- Added `EmbeddingMemory` retrieval helper
- Applied confidence gate before final user-facing replies in research/project flows
- Added local perf traces under `Runtime/logs/perf_trace.jsonl`

## Expected impact
- Better handling of low-confidence source sets
- Faster follow-up answers when similar project research already exists
- Clearer observability for future optimization

## Accuracy guardrails
- Low-confidence answers are narrowed and explicitly caveated
- Prior-summary reuse is retrieval-only; it does not overwrite fresh evidence

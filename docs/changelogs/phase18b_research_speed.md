# Phase 18B — Research Speed and Diversity Stops

## Summary
Improved the research engine to stop earlier once enough diverse, high-quality evidence is available, while also adding lightweight caching and profiling.

## Changes
- Added suffix-based domain matching in `web_research.py`
- Added page cache with TTL
- Added quality/diversity stop conditions
- Added performance tracing for query expansion, search, crawl, scoring, and conflict detection

## Expected impact
- Lower crawl overhead on repeated or well-covered queries
- Better use of CPU/network time
- Easier profiling for future tuning

## Accuracy guardrails
- Early stop requires quality thresholds, domain diversity, and tier-1 coverage
- Conflict detection remains enabled before final answer use

# External Requests Migration Spec (Not Applied Yet)

This is the SQL migration spec for external request persistence. It is defined for readiness, but intentionally not registered in the active migration list in this phase.

## Up migration (spec)

```sql
CREATE TABLE IF NOT EXISTS external_requests (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  intent TEXT NOT NULL,
  project TEXT NOT NULL,
  lane TEXT NOT NULL,
  summary TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL,
  policy_json TEXT NOT NULL,
  result_json TEXT NOT NULL,
  external_ref TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_external_requests_provider_external_ref
ON external_requests(provider, external_ref)
WHERE external_ref <> '';

CREATE INDEX IF NOT EXISTS idx_external_requests_status_created
ON external_requests(status, created_at);

CREATE INDEX IF NOT EXISTS idx_external_requests_project_created
ON external_requests(project, created_at);
```

## Rollback migration (spec)

```sql
DROP INDEX IF EXISTS idx_external_requests_project_created;
DROP INDEX IF EXISTS idx_external_requests_status_created;
DROP INDEX IF EXISTS idx_external_requests_provider_external_ref;
DROP TABLE IF EXISTS external_requests;
```

## Notes

- Current phase keeps this migration out of the formal migration chain by design.
- Runtime support currently uses a guarded schema bootstrap local to the new store contract.

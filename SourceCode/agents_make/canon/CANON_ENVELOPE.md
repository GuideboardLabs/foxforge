# Canon Response Envelope

- Success collection: `{"items": [...], "meta": {"count": N}}`
- Success single: `{"item": {...}}`
- Error: `{"error": {"code": "UPPER_SNAKE", "message": "...", "details": null|object}}`

Status codes:
- Success: `200`, `201`, `204`
- Error: `400`, `404`, `409`, `500`

Route handlers should return through helper functions:
- `ok_item(payload, status=200)`
- `ok_items(payloads, status=200)`
- `err(code, message, status=400, details=None)`

# Artwork Automation

Internal tooling for DTF artwork generation and QA. See `SETUP.md` for install
and deployment.

## Public Print Ready QA API

A key-authenticated HTTP API over the same Print Ready QA measurement the web UI
uses (OpenCV checks + score + band). It is the only public API surface; it does
not use the operator login session.

### Auth

Send your key in the `X-API-Key` header. Keys are configured server-side via the
`QA_API_KEYS` environment variable (comma-separated). A missing or unknown key
returns `401`. Each key is rate limited (default 60 requests/minute, configurable
with `QA_API_RATE_PER_MIN`); exceeding it returns `429`.

### `POST /api/v1/qa/print-ready`

Measure one image and return the score, band, summary and per-check results.

Provide the image in one of three ways:

- multipart form field `file` (PNG / JPG / WEBP)
- JSON body `{"image_url": "https://..."}`
- JSON body `{"image_base64": "<base64>"}` (a `data:` URL prefix is tolerated)

Optional parameters: `dpi` (float, overrides the DPI read from the file; defaults
to 300 when the file has none) and `assessment` (bool, default false).

Limits: 25 MB per image, 12000 px per side. A bad or unsupported image returns
`400` with a clear message.

```bash
curl -X POST https://<server>/api/v1/qa/print-ready \
     -H "X-API-Key: <key>" \
     -F "file=@design.png" \
     -F "dpi=300"
```

JSON body with a base64 image:

```bash
curl -X POST https://<server>/api/v1/qa/print-ready \
     -H "X-API-Key: <key>" \
     -H "Content-Type: application/json" \
     -d '{"image_base64": "iVBORw0KGgo...", "dpi": 300}'
```

Response `200` (`assessment=false`):

```json
{
  "ok": true,
  "score": 87,
  "band": "fixable",
  "summary": "...",
  "image": {"width": 3806, "height": 4470, "dpi": 300,
            "print_in": [12.69, 14.9], "has_alpha": true, "file_kb": 7501.0},
  "checks": [
    {"id": "dpi", "label": "Resolution (DPI)", "value": 300,
     "value_str": "300 DPI", "status": "pass",
     "threshold": "warn <300, fail <150", "weight": 8, "detail": "..."}
  ],
  "measurements": { "...": "raw numbers" },
  "duration_ms": 2350
}
```

`band` is one of `ready`, `fixable`, `not_ready`. Each check `status` is
`pass`, `warn`, `fail` or `na`.

### `assessment=true`

Also queues a normal Print Ready job (PhotoCraft off) for a printer's written
assessment and returns `202` with the OpenCV result above plus:

```json
{"job_id": "ab12...", "status": "queued",
 "poll_url": "/api/v1/qa/print-ready/ab12..."}
```

The written assessment needs an agent online. If none is available the call
still returns `200` with the measurement and an `assessment_error` field.

### `GET /api/v1/qa/print-ready/{job_id}`

Poll an `assessment=true` job (same `X-API-Key`):

```json
{"status": "queued|running|done|failed",
 "score": 87, "band": "fixable",
 "checks": [ ... ],
 "assessment": "<printer's written assessment>",
 "error": null}
```

### OpenAPI / docs

The endpoints appear in FastAPI's interactive docs at `/docs` (and the schema at
`/openapi.json`) under the **QA API** tag, with the request and response models.

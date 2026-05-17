# Jewelify Server — Backend Session Log

## Session 001 — 2026-05-17

**Goal**: Full backend audit + planning

**Completed**:
- Full static analysis of all Python files in `Jewelify_server/`
- Mapped all API endpoints, ML pipeline, DB schema, auth flow
- Identified 19 issues across 4 severity tiers (see `backend_errors.md`)
- Confirmed ADR-002: Twilio removed, auth migrating to email+password+OTP
- Confirmed ADR-003: API URL centralization (backend side: env vars)
- Documented fix priority order in `backend_errors.md`

---

## Session 002 — 2026-05-17

**Goal**: Backend Phase 1 — Auth overhaul (ADR-002) + Critical/High fixes

### Auth Overhaul ✅ COMPLETE

| Task | File | What |
|------|------|------|
| B-01 | `models/user.py` | Full rewrite — `mobileNo→email`, added `name`, password strength validator, 7 new model classes |
| B-02 | `services/email.py` | New file — SMTP email OTP sender via stdlib `smtplib` (STARTTLS), reads SMTP config from env |
| B-03 | `services/auth.py` | Removed all Twilio code, fixed OTP to use `secrets.randbelow()`, store/verify now uses `email` field |
| B-04 | `api/routes/auth.py` | Full rewrite — 7 endpoints: `send-otp`, `verify-otp`, `register`, `login`, `forgot-password`, `reset-password`, `check-user/{email}` |

### Infrastructure Fixes ✅ COMPLETE

| Task | File | What |
|------|------|------|
| B-05 | `api/dependencies.py` | Added `validate_object_id()` helper (raises HTTP 400 on bad ObjectId), `tokenUrl` fixed to `auth/login` |
| B-06 | `services/database.py` | Added `ensure_indexes()` (6 indexes), MongoDB pool tuned: `maxPoolSize=50`, timeouts |
| B-07 | `main.py` | Migrated `@app.on_event("startup")` → `lifespan`, CORS added, `ensure_indexes()` called at startup |
| B-08 | `render_api_calling.py` | Deleted (dead code) |
| B-09 | `.env.example` | Removed Twilio vars, added SMTP vars |

### Performance + Security Fixes ✅ COMPLETE

| Task | File | What |
|------|------|------|
| C-001 | `main.py`, `api/routes/predictions.py` | ML models loaded once in `lifespan`, cached in `app.state.predictor`, injected via `request.app.state` |
| C-002 | `services/predictor.py` | All TF/XGBoost/OpenCV calls wrapped in `asyncio.to_thread()` — event loop no longer blocked |
| H-002 | `api/routes/auth.py`, `main.py` | `slowapi` rate limits: `send-otp` 3/min, `verify-otp` 5/min, `login` 5/min |
| H-006 | `main.py` | Upload size middleware — 10 MB cap (env: `MAX_UPLOAD_SIZE_MB`) |
| M-003 | `services/predictor.py` | Removed unused `pairwise_features.npy` load (was wasting RAM) |
| M-005 | `api/routes/history.py` | Fixed — all `{"error": ...}` dicts replaced with `HTTPException` |
| M-007 | `services/predictor.py` | Fixed S3 URL typo `"earings"` → `"earrings"` |
| M-009 | `api/routes/predictions.py`, `history.py` | Internal errors logged only, generic messages returned to client |
| L-003 | `keep_alive.py` | Keep-alive URL reads from `RENDER_URL` env var |
| requirements | `requirements.txt` | `twilio` removed, `slowapi==0.1.9` added |

---

## Session 003 — TBD

**Goal**: Backend Phase 2 — Remaining medium issues

**Planned**:
- M-002: Wire actual ML category → recommendation filtering (currently hardcoded S3 URLs)
- M-004: Migrate `services/database.py` to `motor` async driver
- M-008: JWT refresh token mechanism (short-lived access + long-lived refresh)
- L-002: Password strength validation already done (in `models/user.py`)
- L-005: Remove duplicate model files from root vs `trained_features/`
- C-003: DB indexes already done in Session 002

**Status**: Pending

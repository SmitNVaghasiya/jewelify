# Jewelify Server — Backend Errors & Issues

**Generated**: 2026-05-17  
**Last Updated**: 2026-05-17 (Session 002)  
**Source**: Full static analysis of `Jewelify_server/`

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fixed |
| ⏳ | Pending |

## Severity Legend

| Symbol | Level |
|--------|-------|
| 🔴 | Critical |
| 🟠 | High |
| 🟡 | Medium |
| 🟢 | Low |

---

## 🔴 Critical

| ID | Issue | Status |
|----|-------|--------|
| C-001 | ML models load per-request — `JewelryPredictor()` in `predictor.py` instantiated on every POST | ✅ Fixed — cached in `lifespan`, injected via `app.state.predictor` |
| C-002 | All TF/XGBoost/OpenCV calls block async event loop | ✅ Fixed — all blocking calls wrapped in `asyncio.to_thread()` |
| C-003 | No database indexes — full collection scans on every query | ✅ Fixed — `ensure_indexes()` creates 6 indexes at startup |

---

## 🟠 High

| ID | Issue | Status |
|----|-------|--------|
| H-001 | Weak OTP via `random.choice()` — not cryptographically secure | ✅ Fixed — `secrets.randbelow(10**6)` |
| H-002 | No rate limiting on auth endpoints — brute-force possible | ✅ Fixed — `slowapi`: 3/min on send-otp, 5/min on verify-otp+login |
| H-003 | No CORS middleware | ✅ Fixed — `CORSMiddleware` with `allow_origins=["*"]` |
| H-004 | Unvalidated `ObjectId()` — raises 500 on malformed IDs | ✅ Fixed — `validate_object_id()` helper in `dependencies.py` |
| H-005 | Twilio still in codebase, auth built around mobile OTP (ADR-002 decided email) | ✅ Fixed — full auth rewrite, Twilio removed, email OTP via SMTP |
| H-006 | No file upload size limit — DoS via large images | ✅ Fixed — 10 MB middleware in `main.py` |

---

## 🟡 Medium

| ID | Issue | Status |
|----|-------|--------|
| M-001 | Deprecated `@app.on_event("startup")` | ✅ Fixed — migrated to `lifespan` context manager |
| M-002 | Recommendations hardcoded S3 URLs — ML category output ignored | ⏳ Pending (Session 003) |
| M-003 | `pairwise_features.npy` loaded into RAM but never used | ✅ Fixed — load removed |
| M-004 | `motor` (async MongoDB) in requirements but `pymongo` (sync) used | ⏳ Pending (Session 003) |
| M-005 | `history.py` returns `{"error": ...}` dict instead of `HTTPException` | ✅ Fixed — all `HTTPException` now |
| M-006 | MongoDB connection pool not tuned | ✅ Fixed — `maxPoolSize=50`, `serverSelectionTimeoutMS=5000` |
| M-007 | S3 URL typo `"earings"` → should be `"earrings"` | ✅ Fixed |
| M-008 | No JWT refresh token — 30-day tokens, no rotation | ⏳ Pending (Session 003) |
| M-009 | `str(e)` in `HTTPException` detail leaks internal errors | ✅ Fixed — generic messages returned, full error logged internally |
| M-010 | `render_api_calling.py` dead code | ✅ Fixed — deleted |

---

## 🟢 Low

| ID | Issue | Status |
|----|-------|--------|
| L-001 | Mobile number validation too permissive | ✅ N/A — mobile auth removed (ADR-002) |
| L-002 | No password strength validation | ✅ Fixed — validator in `models/user.py` (min 8, uppercase, digit, special char) |
| L-003 | Keep-alive URL hardcoded in `keep_alive.py` | ✅ Fixed — reads from `RENDER_URL` env var |
| L-004 | Twilio client fails silently | ✅ N/A — Twilio removed |
| L-005 | Duplicate model files in root + `trained_features/` | ⏳ Pending (Session 003) |

---

## Remaining Work (Session 003)

| Priority | Issue | Effort |
|----------|-------|--------|
| 1 | M-002 — Wire ML category → recommendation filtering | High |
| 2 | M-004 — Migrate to `motor` async MongoDB driver | Medium |
| 3 | M-008 — JWT refresh token mechanism | Medium |
| 4 | L-005 — Remove duplicate model files | Trivial |

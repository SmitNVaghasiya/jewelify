# Jewelify — Architecture Decisions

## ADR-001: Terracotta & Cream Design Theme
**Date**: 2026-05-17  
**Status**: Accepted

**Context**: App needed full UI redesign from minimal Material defaults. Three finalists evaluated: Rose & Feminine, Terracotta & Cream, Emerald & Gold.

**Decision**: Terracotta & Cream.

**Reasons**:
- Most unique personality in the app market — artisan, handmade, soulful
- Cormorant Garamond (display) + Poppins (body) gives editorial magazine quality
- Warm cream backgrounds suit jewelry photography
- ui-ux-pro-max confirmed Cormorant/Montserrat as ideal typography for jewelry/luxury
- Narrower risk: design stays timeless, not trend-dependent

**Rejected**:
- Rose & Feminine — too common (thousands of pink beauty apps)
- Emerald & Gold — strong but darker UI heavy on small screens

---

## ADR-002: Auth Changed from Mobile OTP to Email + Password + OTP
**Date**: 2026-05-17  
**Status**: Accepted

**Context**: Original auth used Twilio SMS OTP (mobile number). Twilio free tier limited; SMS cost per user not sustainable for indie project.

**Decision**: Replace with email + password + email OTP.

**New auth flow**:
1. Register: name → username → email → password → OTP sent to email → verify OTP
2. Login: email + password → JWT
3. Forgot password: email → OTP → new password
4. Resend OTP: button on OTP screen after 60s cooldown

**Backend changes required** (FastAPI):
- `POST /auth/register` — body: `{name, username, email, password}`
- `POST /auth/send-otp` — body: `{email}` → sends 6-digit OTP via email provider
- `POST /auth/verify-otp` — body: `{email, otp}` → returns JWT
- `POST /auth/login` — body: `{email, password}` → returns JWT
- `POST /auth/forgot-password` — body: `{email}` → sends reset OTP
- `POST /auth/reset-password` — body: `{email, otp, new_password}`
- User model: replace `mobileNo` field with `email`

**Email OTP provider options**: SendGrid (free 100/day), Mailgun, or SMTP via Gmail

**Frontend `UserOut` model change**: `mobileNo: String` → `email: String`

---

## ADR-003: API URL Centralized to Single Constant
**Date**: 2026-05-17  
**Status**: Accepted

**Context**: `https://jewelify-server.onrender.com` was hardcoded in 4+ files.

**Decision**: Single source at `lib/constants/api.dart` → `ApiConstants.baseUrl`.

---

## ADR-004: Typography Stack
**Date**: 2026-05-17  
**Status**: Accepted

**Decision**:
- Display/headings: Cormorant Garamond — italic serif, luxury/fashion mood
- Body/UI text: Poppins — already bundled (16 weights), clean and readable
- No Montserrat — Poppins already serves same role, adding another sans-serif is redundant

**Cormorant weights needed**: Regular (400), Italic (400i), SemiBold (600), SemiBold Italic (600i)

---

## ADR-005: Removed Unused Dependencies
**Date**: 2026-05-17  
**Status**: Accepted

**Removed from pubspec.yaml**:
- `photo_view` — commented out, replaced by custom `image_zoom_widget.dart`
- `easy_image_viewer` — never used
- `smooth_page_indicator` — commented out
- `cached_network_image` — unused in current build (images are local)

**Kept**:
- `transparent_image` — used as placeholder in FadeInImage
- All others actively used

---

## ADR-006: Skeleton Loading Screens
**Date**: 2026-05-17  
**Status**: Accepted

**Context**: History and results screens showed only a spinner during loads. UX felt cheap.

**Decision**: Add shimmer skeleton loader widget (`lib/widgets/skeleton_loader.dart`) used in history list and while prediction results poll. Animated shimmer using AnimationController + gradient.

---

## ADR-007: Render Cold-Start UX Messaging
**Date**: 2026-05-17  
**Status**: Accepted

**Context**: Backend on Render free tier sleeps after inactivity → 30–60s first request delay. Users had no feedback.

**Decision**: ProcessingScreen detects slow response (>8s) and shows message: "Server is warming up, please wait a moment..." Resolved when response arrives.

---

## ADR-008: Backend ML Model Caching
**Date**: 2026-05-17  
**Status**: Accepted

**Context**: `JewelryPredictor()` was instantiated per-request, loading TF + XGBoost + Haar Cascade each time (3–5s overhead, OOM risk).

**Decision**: Load `JewelryPredictor` once in FastAPI `lifespan` context, cache in `app.state.predictor`, inject via `request.app.state.predictor` in route handlers.

---

## ADR-009: Async ML Inference via Thread Pool
**Date**: 2026-05-17  
**Status**: Accepted

**Context**: All TF/XGBoost/OpenCV calls were synchronous in async FastAPI handlers — blocked event loop, caused timeout under concurrent load.

**Decision**: Wrap all blocking ML/OpenCV calls in `asyncio.to_thread()`. Sync helper methods (`_predict_xgboost_sync`, `_predict_mlp_sync`, `_extract_features_sync`, `_validate_face_sync`) called from async methods.

---

## ADR-010: Rate Limiting with slowapi
**Date**: 2026-05-17  
**Status**: Accepted

**Decision**: Added `slowapi` for IP-based rate limiting on auth endpoints:
- `POST /auth/send-otp`: 3 requests/minute
- `POST /auth/verify-otp`: 5 requests/minute  
- `POST /auth/login`: 5 requests/minute

**Rejected**: `fastapi-limiter` (requires Redis), `fastapi-limiter2` (less maintained). `slowapi` is in-memory, zero infrastructure.

---

## ADR-011: SMTP Email for OTP (No Third-Party SDK)
**Date**: 2026-05-17  
**Status**: Accepted

**Decision**: Used Python stdlib `smtplib` + `email.mime` for OTP emails. No SendGrid/Mailgun SDK dependency.

**Config**: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM_NAME` from env vars. Default: Gmail SMTP on port 587 with STARTTLS.

---

## ADR-012: Replace google_ml_kit with Granular google_mlkit_face_detection
**Date**: 2026-05-17  
**Status**: Accepted

**Context**: `google_ml_kit ^0.19.0` is a monolithic umbrella package. It references language-specific text recognition classes (Chinese, Devanagari, Japanese, Korean) that are not bundled as AARs. R8 minifier found dangling class references → release build failed. App only uses face detection.

**Decision**: Replace with `google_mlkit_face_detection ^0.13.2`. Use granular ML Kit packages — only pull what the app actually uses.

**Migration**: `GoogleMlKit.vision.faceDetector()` → `FaceDetector(options: FaceDetectorOptions())`. Import changed from `google_ml_kit/google_ml_kit.dart` → `google_mlkit_face_detection/google_mlkit_face_detection.dart`.

**Result**: 13 unused ML Kit packages removed from dependency tree. R8 crash resolved. APK builds clean.

---

## ADR-013: WillPopScope → PopScope Migration
**Date**: 2026-05-17  
**Status**: Accepted

**Context**: `WillPopScope` deprecated after Flutter v3.12.0. Android predictive back gesture does not work with `WillPopScope`.

**Decision**: All `WillPopScope` usages replaced with `PopScope(canPop: false, onPopInvokedWithResult:)`.

**Files affected**: `processing_screen.dart`, `upload_screen.dart`.

---

## ADR-014: Android NDK and Kotlin Version Pinning
**Date**: 2026-05-17  
**Status**: Accepted

**Decision**:
- NDK: `28.2.13676358` in `android/app/build.gradle.kts` — required by `jni` plugin (transitive from `google_mlkit_face_detection`)
- Kotlin KGP: `2.1.0` in `android/settings.gradle.kts` — Flutter dropping support for <2.1.0 soon

**Rule**: When adding ML Kit or JNI-dependent packages, check required NDK version and bump to highest required.

---

## ADR-015: Local Dev Backend via --host 0.0.0.0
**Date**: 2026-05-17  
**Status**: Accepted

**Context**: Uvicorn default binds to `127.0.0.1` (loopback only). Phone on same WiFi hits `192.168.1.9:5000` — different interface → connection refused.

**Decision**: Run `uvicorn main:app --host 0.0.0.0 --port 5000` during local dev. `ApiConstants.baseUrl` set to `http://<local-ip>:5000` for local testing. Revert to Render URL before release build.

**Rule**: Never commit `ApiConstants.baseUrl` pointing to local IP.

---

## ADR-016: Register Flow — Verify OTP Before Register
**Date**: 2026-05-17  
**Status**: Accepted

**Context**: Flutter `register()` was sending `otp` field in body to `POST /auth/register`. Backend `UserRegister` Pydantic model has no `otp` field → 422 Unprocessable Entity. Also, OTP verification was being skipped entirely.

**Decision**: Split into two sequential calls in `_verify()`:
1. `POST /auth/verify-otp` — verify OTP first
2. `POST /auth/register` — register without `otp` field

Added `verifyOtp()` method to `AuthProvider`.

**Password validation**: Frontend validator updated to match backend rules (8+ chars, uppercase, digit, special char) so invalid passwords are caught before OTP is sent.

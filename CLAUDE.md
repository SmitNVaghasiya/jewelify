# Jewelify — Claude Context

## What This Is
AI-powered jewelry recommendation Flutter app. User uploads face photo + jewelry photo → FastAPI backend runs XGBoost + Keras/MLP → returns compatibility score + recommendations.

## Stack
- **Frontend**: Flutter (Dart), Provider state management, named routes, `google_fonts` package
- **Backend**: FastAPI at `https://jewelify-server.onrender.com` (Render free tier — cold starts expected)
- **DB**: MongoDB
- **Auth**: JWT bearer tokens stored in FlutterSecureStorage
- **ML**: Google ML Kit (face validation client-side), XGBoost + Keras server-side

## Design System (LOCKED)
**Theme**: Terracotta & Cream  
**Colors**:
- Background: `#fdf6ef`
- Surface (cards): `#ffffff`
- Border/divider: `#ede0d4`
- Soft surface: `#f5ede4`
- Primary CTA: `#b5451b` (terracotta)
- Heading dark brown: `#3d1f15`
- App name brown: `#6b3a2a`
- Muted/secondary text: `#a07060`

**Typography**:
- Display/headings: `Cormorant Garamond` via `google_fonts` package (no manual font files)
- Body/UI: `Poppins` (already bundled — 16 weights in assets/fonts/)
- Labels: Poppins uppercase, wide letter-spacing

**UI rules**:
- No emoji as icons — use Flutter Icons or SVG equivalents
- All interactive elements: visual press states
- Transitions: 150–300ms smooth curves
- Text contrast: 4.5:1 minimum (WCAG AA)
- Use `.withValues(alpha: x)` NOT `.withOpacity(x)` (deprecated Flutter 3.27+)

## Auth (CHANGED — ADR-002)
Old: mobile number + Twilio SMS OTP  
New: email + password + email OTP  
Fields at registration: name, username, email, password, OTP  
Screens: login, register (2-step), forgot-password (3-step: email→OTP→new pass)  
Backend: all endpoints updated — see decisions.md ADR-002

## Key Files
```
jewelify_app/lib/
├── constants/api.dart              # API base URL — ApiConstants.baseUrl
├── providers/auth_provider.dart   # Auth state, all API calls
├── screens/
│   ├── app_theme.dart             # ALL color/font/style constants (AppTheme.*)
│   ├── login_screen.dart          # Email + password login
│   ├── register_screen.dart       # 2-step registration
│   ├── forgot_password_screen.dart # 3-step password reset
│   ├── home_screen.dart           # ✅ Redesigned (terracotta)
│   ├── upload_screen.dart         # ✅ Redesigned (terracotta)
│   ├── processing_screen.dart     # ✅ Redesigned + cold-start UX (8s trigger)
│   ├── results_screen.dart        # ✅ Redesigned (skeleton loading, terracotta)
│   └── history_screen.dart        # ✅ Redesigned (terracotta)
├── widgets/
│   ├── skeleton_loader.dart       # ✅ New — shimmer loading widget
│   ├── prediction_module.dart     # Uses _loadFile() for image loading
│   ├── recommendation_card.dart
│   ├── score_display.dart
│   └── expandable_item.dart
└── models/
    └── jewelry_recommendation.dart

Jewelify_server/
├── main.py                        # lifespan, CORS, rate limiting, upload size limit
├── models/user.py                 # Email-based auth models
├── services/
│   ├── auth.py                    # JWT, bcrypt, OTP (secrets module)
│   ├── email.py                   # SMTP email sender (stdlib smtplib)
│   ├── database.py                # MongoDB + ensure_indexes()
│   └── predictor.py               # JewelryPredictor — asyncio.to_thread() for all ML
├── api/routes/
│   ├── auth.py                    # 7 endpoints, slowapi rate limits
│   ├── predictions.py             # Uses app.state.predictor
│   └── history.py                 # All HTTPException (no error dicts)
└── api/dependencies.py            # validate_object_id() helper
```

## What NOT to Do — Flutter
- Do NOT use `.withOpacity()` — use `.withValues(alpha: x)`
- Do NOT hardcode API URLs — use `ApiConstants.baseUrl`
- Do NOT use emoji as UI icons
- Do NOT import `image_storage.dart` anywhere — both `lib/widgets/image_storage.dart` and `lib/screens/image_storage.dart` are deleted. Use `dart:io File` directly for local image loading
- Do NOT use `photo_view`, `easy_image_viewer`, `smooth_page_indicator` (removed)
- Do NOT add `temp_` prefix to any file — `temp_registration.dart` deleted, `register_screen.dart` is canonical
- Do NOT use `CardTheme(` in ThemeData — use `CardThemeData(`
- Do NOT use `WillPopScope` — use `PopScope` with `canPop` + `onPopInvokedWithResult`
- Do NOT use `google_ml_kit` (monolithic) — use `google_mlkit_face_detection` (granular)
- Do NOT use `__` as wildcard parameter name — use `_` (Dart 3+ supports multiple `_` wildcards)

## What NOT to Do — Backend
- Do NOT instantiate `JewelryPredictor()` in route handlers — use `request.app.state.predictor`
- Do NOT call TF/XGBoost/OpenCV directly in async functions — use `asyncio.to_thread()`
- Do NOT use `random.choice()` for OTP — use `secrets.randbelow()`
- Do NOT return `{"error": "..."}` dicts — raise `HTTPException`
- Do NOT use bare `ObjectId(id_str)` — use `validate_object_id(id_str)`

## Render Cold Start
Backend on Render free tier sleeps after inactivity. First request may take 30–60s. Show "Server warming up, please wait a moment..." on processing screen when response is slow (>8s).

## Doc Update Rules (MANDATORY)
After any frontend work: update `docs/sessions.md` (new session entry) + `docs/decisions.md` (new ADR if architectural decision made).  
After any backend work: update `docs/backend_sessions.md` (new session entry) + `docs/backend_errors.md` (mark fixed issues, add new ones found).  
Both: update `CLAUDE.md` Pending Work section.

## Android Build Config
- NDK version: `28.2.13676358` (set in `android/app/build.gradle.kts`) — required by `jni` plugin
- Kotlin KGP: `2.1.0` (set in `android/settings.gradle.kts`)
- Build for Play Store: use `flutter build appbundle` — AAB lets Play serve per-ABI, cuts install size to ~35–40MB
- Direct APK distribution: use `flutter build apk --split-per-abi`

## Local Dev Setup
- Run backend: `uvicorn main:app --reload --port 5000 --host 0.0.0.0` (must use `--host 0.0.0.0` for phone on WiFi)
- Set `ApiConstants.baseUrl = 'http://192.168.1.9:5000'` for local testing
- **REVERT** to `https://jewelify-server.onrender.com` before any release build or git push
- Windows Firewall rule required: `netsh advfirewall firewall add rule name="Uvicorn 5000" dir=in action=allow protocol=TCP localport=5000`
- MongoDB Atlas free tier pauses after ~60 days idle — resume from Atlas dashboard if DNS errors appear

## Pending Work
**Frontend**: ✅ `flutter analyze` clean (0 issues). E2E flow verified working (Session 004).  
**Backend**: Session 003 (ML recommendations engine — wire actual category to recommendations, motor async driver, JWT refresh token)  
**Before release**: Revert `ApiConstants.baseUrl` to Render URL. Run `flutter build apk --split-per-abi`.

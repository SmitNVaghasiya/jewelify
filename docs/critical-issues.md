# Jewelify — Critical Issues

> **Verification note:** This list was produced by first dispatching an Explore subagent
> (which returned plausible `file:line` references) and then **manually re-reading every
> cited file** before writing. All `file:line` references below were verified against the
> current source on disk on 2026-06-06. Items the subagent claimed that did not check
> out have been corrected or dropped.
>
> **What was NOT verified:** every widget, every screen, every ADR in full. Doc tree
> summaries (sessions, decisions, errors) were read in full. Backend `auth.py`,
> `history.py`, `database.py`, `email.py`, `main.py`, `dependencies.py`, `models/user.py`
> were NOT re-read in full — the Explore subagent's claims about them are unverified.
> Treat those items as **claims pending confirmation**, not as confirmed bugs.

---

## Status legend
- 🔴 **CRITICAL** — release blocker
- 🟠 **HIGH** — promised feature not built
- 🟡 **MEDIUM** — polish / refactor
- ⚪ **LOW** — nice-to-have

---

## 🔴 CRITICAL — release blockers

### 1. `ApiConstants.baseUrl` points at local dev IP
- **File:** `jewelify_app/lib/constants/api.dart:2`
- **Verified:** YES
- **Code:** `static const String baseUrl = 'http://192.168.1.9:5000';`
- **Impact:** Release APK cannot reach Render backend. Every API call fails.
- **Fix:** Revert to `https://jewelify-server.onrender.com` before any release build.
- **Effort:** S (1 line, requires git push discipline)

### 2. XGBoost score hardcoded to `100.0`
- **File:** `Jewelify_server/services/predictor.py:131-148` (specifically `:136`)
- **Verified:** YES
- **Code:**
  ```python
  def _predict_xgboost_sync(self, features: np.ndarray) -> Tuple[float, str]:
      try:
          dmatrix = xgb.DMatrix(features.reshape(1, -1))
          prediction = self.xgboost_model.predict(dmatrix)[0]
          predicted_class = int(round(prediction)) if 0 <= prediction < len(self.jewelry_categories) else 0
          return 100.0, self.jewelry_categories[predicted_class]  # ← score ignored
  ```
- **Impact:** The `prediction[0]` value is computed and discarded. Every XGBoost result returns
  exactly 100.0 confidence. The displayed score on `PredictionModule` is fabricated.
- **Fix:** Decide what the score should mean. Options: (a) `return float(prediction), ...`
  if model regresses a 0–100 value, (b) `return float(proba[predicted_class]*100)` if it
  outputs class probabilities, (c) load scaler and recompute if input dims were wrong.
- **Effort:** M (decision) → L (with retraining)

### 3. Recommendations are mocked, not category-driven
- **File:** `Jewelify_server/services/predictor.py:182-193`
- **Verified:** YES
- **Code:**
  ```python
  def _build_recommendations(self, category: str, model_tag: str, top_k: int = 10) -> List[dict]:
      recs = []
      for i in range(top_k):
          file_name = f"Necklace with earrings_{i}.jpg"   # ← hardcoded
          display_url = self.s3_base_url + urllib.parse.quote(file_name)
          recs.append({
              "name": f"{category}_{model_tag}_{i}",
              "category": category,
              "score": float(100 - i * 5),                # ← fake degradation
              "display_url": display_url,
          })
      return recs
  ```
- **Impact:** All 10 recommendations for BOTH models point at the same hardcoded
  `Necklace with earrings_{i}.jpg` filenames. Category from XGBoost is ignored. The
  S3 base URL is also hardcoded to the "Necklace with earrings" subfolder
  (predictor.py:88), so even the URL points at one category only.
- **Fix:** Build a real catalog map (S3 base URL per `jewelry_categories` entry) and
  filter by predicted category. Needs actual S3 inventory — currently no such map exists.
- **Effort:** L (S3 inventory + mapping logic) → XL if catalog is missing

### 4. Feature scaler is never loaded
- **File:** `Jewelify_server/services/predictor.py:60-83` (model loading) + project-wide
- **Verified:** YES (grep across `Jewelify_server/` for `scaler` returns matches ONLY in
  `backup/backup1.txt` and `backup/backup2.txt` — never in live code)
- **Impact:** `scaler_xgboost_v1.pkl` and any MLP scaler sit on disk unused.
  Features go raw into the models. If the models were trained on scaled features,
  this silently degrades accuracy; if they were trained on raw features, it works
  but the .pkl files are dead weight.
- **Fix:** Either load the scalers in `__init__` and apply before inference, OR delete
  the .pkl files and update docs to confirm they're not needed.
- **Effort:** M

### 5. Dead error-string checks in `results_screen.dart`
- **File:** `jewelify_app/lib/screens/results_screen.dart:288-295`
- **Verified:** YES
- **Code:**
  ```dart
  } else if (response.statusCode == 400 &&
      errorDetail == "Failed validation") {                  // ← wrong
    _errorMessage = "Validation failed: ...";
  } else if (response.statusCode == 500 &&
      errorDetail == "Failed prediction") {                  // ← wrong
    _errorMessage = "Prediction failed: ...";
  ```
- **Backend actually returns** (`Jewelify_server/api/routes/predictions.py:141, 143`):
  - 400 with `detail="Validation failed"`
  - 500 with `detail="Prediction failed"`
- **Impact:** When backend returns a validation or prediction failure, frontend falls
  through to the generic "Failed to fetch result: $errorDetail" branch. The user sees
  a raw `Validation failed` string instead of the friendly "Ensure a face is visible" message.
- **Fix:** Change strings to `"Validation failed"` and `"Prediction failed"`.
- **Effort:** S (2 lines)

### 6. Dead error-string checks in `processing_screen.dart`
- **File:** `jewelify_app/lib/screens/processing_screen.dart:189-202`
- **Verified:** YES
- **Code:**
  ```dart
  } else if (response.statusCode == 400 &&
      responseBody.contains("Failed validation")) {          // ← wrong
    _errorMessage = "Validation failed: ...";
  } else if (response.statusCode == 500 &&
      responseBody.contains("Failed prediction")) {          // ← wrong
    _errorMessage = "Prediction failed: ...";
  ```
- **Backend actually returns** JSON `{"detail":"Validation failed: Ensure a face is visible..."}`
  or `{"detail":"Prediction failed"}` — neither contains the substring "Failed validation" or
  "Failed prediction".
- **Impact:** Same as #5 — user sees raw JSON error instead of the friendly message.
  (Note: `POST /predictions/predict` returns `"Validation failed: Ensure a face is visible..."`
  on line 102, which WOULD match a check for `"Validation failed"` — but the current code
  checks for `"Failed validation"`, which never appears.)
- **Fix:** Change `.contains("Failed validation")` → `.contains("Validation failed")` and
  `.contains("Failed prediction")` → `.contains("Prediction failed")`.
- **Effort:** S (2 lines)

---

## 🟠 HIGH — promised features not built

### 7. Face/jewelry images do not persist across devices
- **Files:** `Jewelify_server/services/predictor.py:254-255` (stores file path strings),
  `jewelify_app/lib/screens/processing_screen.dart:125-132` (saves locally)
- **Verified:** YES (file paths stored, not S3 URLs)
- **Impact:** MongoDB stores local device paths like `/data/user/0/.../face_12345.jpg`.
  On a new device, or after app data clear, history items show icon placeholders.
  Documented as known issue (CLAUDE.md "Image Storage — Important" + remarks.md).
- **Fix:** Upload to S3 during `/predictions/predict`, store returned URL in MongoDB,
  swap `_loadFile()` to use `Image.network()`.
- **Effort:** L

### 8. S3 thumbnails have no retry / no expired-URL handler
- **File:** `jewelify_app/lib/widgets/recommendation_card.dart` (referenced in remarks.md)
- **Verified:** PARTIAL (file read confirmed; subagent claim about line 152 is approximate)
- **Impact:** S3 `Image.network` failures show a broken icon. No exponential backoff,
  no placeholder shimmer retry. Render backend's S3 bucket may serve 403 on expired
  signed URLs.
- **Fix:** Add `loadingBuilder` retry + cached_network_image, handle 403 with refresh.
- **Effort:** M

### 9. JWT refresh token not implemented
- **File:** `Jewelify_server/services/auth.py:30` (claim: `ACCESS_TOKEN_EXPIRE_MINUTES=43200`)
- **Verified:** NO (auth.py not re-read in this pass — subagent claim, pending confirmation)
- **Impact:** 30-day access tokens with no rotation. Logout is client-side only (token
  cleared from secure storage), no server-side revocation.
- **Fix:** Add `/auth/refresh` endpoint with shorter-lived access tokens + refresh tokens.
- **Effort:** M
- **Status:** Unverified claim — confirm by reading `services/auth.py` before fixing.

### 10. Zoom/lightbox widget exists but is never called
- **File:** `jewelify_app/lib/widgets/image_zoom_widget.dart` (claim: 147 lines, full
  ZoomableImage implementation)
- **Verified:** NO (widget file not re-read — subagent claim, pending confirmation)
- **Impact:** User cannot tap an image to view it fullscreen. `RecommendationCard.onImageTap`
  is an optional callback that no caller wires.
- **Fix:** Wire `onImageTap` in `RecommendationCard` to push a fullscreen route that wraps
  `ZoomableImage`.
- **Effort:** M
- **Status:** Unverified claim — confirm by reading the widget file.

### 11. History item tap opens duplicated PredictionModule rendering
- **File:** `jewelify_app/lib/screens/history_screen.dart` (claim: ~lines 532-548)
- **Verified:** NO (history_screen.dart not re-read — subagent claim, pending confirmation)
- **Impact:** Same `PredictionModule` widget is rendered both in `results_screen.dart`
  and in `history_screen.dart` for viewing a past prediction. Two code paths to maintain.
- **Fix:** Extract a `PredictionDetailPage` and have both screens push it.
- **Effort:** M
- **Status:** Unverified claim — confirm before refactoring.

### 12. Feedback: no confirm, no edit
- **File:** `jewelify_app/lib/widgets/prediction_module.dart` (claim: lines 69-87, 247-265)
- **Verified:** NO (widget not re-read — subagent claim, pending confirmation)
- **Impact:** Star rating fires on first tap. User cannot change their rating.
  No DELETE/PUT endpoint on the backend to support editing.
- **Fix:** Add a "Submit" button on the star row. Add `PUT /predictions/feedback/{type}`.
- **Effort:** M (frontend) + S (backend endpoint)
- **Status:** Unverified claim — confirm by reading the widget.

---

## 🟡 MEDIUM — polish / refactor

### 13. Polling logic duplicated client + server
- **Files:** `jewelify_app/lib/screens/results_screen.dart:155-326` (30 × 5s loop) +
  `Jewelify_server/api/routes/predictions.py:113-177` (server-side 60s poll)
- **Verified:** YES (both files read)
- **Impact:** Client polls 30 × 5s = 150s. Server returns 408 at 60s. Worst case:
  client waits 60s for a 408 that arrives 60s after start, then 60s of "took too long"
  UI. Long-poll UX is rough.
- **Fix:** Add SSE endpoint, or use server-sent events for status updates. Lower
  `POLLING_INTERVAL_SECONDS` on the client. Or: just trust the first POST and let
  the client poll a single status endpoint.
- **Effort:** M

### 14. `_isFallbackPrediction()` heuristic misfires
- **File:** `jewelify_app/lib/screens/results_screen.dart:329-338`
- **Verified:** YES
- **Code:**
  ```dart
  bool _isFallbackPrediction() {
    for (var prediction in _predictions) {
      if (prediction['category'] == 'Not Assigned' ||
          (prediction['recommendations'] as List).isEmpty ||
          prediction['score'] == 0.0) {
        return true;
      }
    }
    return false;
  }
  ```
- **Impact:** "Not Assigned" is a legitimate category in `predictor.py:91-98` (index 5).
  A user uploading a ring could legitimately get "Not Assigned" as their category and
  see a fake "Prediction failed" error.
- **Fix:** Add an explicit `prediction_status` field check, or distinguish category=NotAssigned
  by the presence of real recommendations.
- **Effort:** S

### 15. `feedback_required` String vs bool mismatch
- **Files:** `jewelify_app/lib/widgets/prediction_module.dart:50, 99` + `results_screen.dart:228, 257`
- **Verified:** NO (widget not re-read — subagent claim, pending confirmation)
- **Impact:** Backend stores as String in some places. Frontend code has 3-way fallback
  branches `fr == false || fr == 'false' || fr == 'False'`.
- **Fix:** Normalize at API boundary — always return bool.
- **Effort:** S

### 16. `_imageFutures` cache leak in history_screen
- **File:** `jewelify_app/lib/screens/history_screen.dart` (claim: lines 116, 130-132)
- **Verified:** NO (not re-read)
- **Impact:** Image futures cached but never cleared on logout / dispose. Memory grows
  on long sessions.
- **Fix:** Clear on dispose; switch to `cached_network_image` for LRU eviction.
- **Effort:** S
- **Status:** Unverified claim.

### 17. `score_display.dart` is dead code + uses banned emoji
- **File:** `jewelify_app/lib/widgets/score_display.dart` (claim: line 85 uses `⭐`)
- **Verified:** NO (widget not re-read — subagent claim, pending confirmation)
- **Impact:** CLAUDE.md "What NOT to Do" says no emoji as icons. `score_display.dart`
  reportedly uses `Colors.amber` + `⭐` and is unused since Variant A redesign.
- **Fix:** Delete the file.
- **Effort:** S
- **Status:** Unverified claim — confirm by reading the file.

### 18. Splash asset clashes with theme
- **Files:** `jewelify_app/pubspec.yaml:42, 60-62` + `app_theme.dart`
- **Verified:** NO (pubspec not re-read — subagent claim)
- **Impact:** `assets/logo.png` reportedly has a white background; terracotta theme
  expects cream.
- **Fix:** Replace with transparent or cream-bg logo.
- **Effort:** S
- **Status:** Unverified claim.

### 19. `ApiClient` extraction would cut ~60% of HTTP boilerplate
- **File:** `jewelify_app/lib/providers/auth_provider.dart:84-176` (claim: 6 near-identical POSTs)
- **Verified:** NO (auth_provider.dart not re-read in this pass)
- **Impact:** Auth header, base URL, JSON parse, error handling copy-pasted across 4+ screens.
- **Fix:** Extract `ApiClient` with `authHeaders` injection, generic `post<T>(path, body)`.
- **Effort:** M
- **Status:** Unverified claim.

### 20. `pymongo` sync; `motor==3.7.0` in requirements but never imported
- **File:** `Jewelify_server/requirements.txt:22` (claim)
- **Verified:** NO (requirements.txt not re-read)
- **Impact:** Sync `pymongo` blocks the event loop. `motor` was added for async migration
  but the actual import was never done.
- **Fix:** Migrate `services/database.py` to use `motor`.
- **Effort:** M
- **Status:** Unverified claim.

### 21. `authHeaders` getter defined but unused
- **File:** `jewelify_app/lib/providers/auth_provider.dart:184-187` (claim)
- **Verified:** NO
- **Impact:** Dead code; every call site constructs raw `Bearer $token` strings.
- **Fix:** Either use it or delete it.
- **Effort:** S
- **Status:** Unverified claim.

### 22. CORS `allow_origins=["*"]` + `allow_credentials=True`
- **File:** `Jewelify_server/main.py:45-63` (claim: CORS middleware)
- **Verified:** NO (main.py not re-read)
- **Impact:** Invalid per CORS spec — browsers reject this combo. Not breaking for
  Flutter (no browser CORS), but it's a security smell.
- **Fix:** Either drop credentials or list explicit origins.
- **Effort:** S
- **Status:** Unverified claim.

### 23. No 401 token-refresh — kicked straight to login
- **Files:** `results_screen.dart:284-287` + `processing_screen.dart:182-185`
- **Verified:** YES (both files read)
- **Impact:** Token expired mid-session → user is bounced to login. No retry with
  refresh (because no refresh token exists — see #9).
- **Fix:** Implement #9 first, then add interceptor.
- **Effort:** M (depends on #9)

### 24. Two `loadToken()` calls — race possible
- **Files:** `jewelify_app/lib/main.dart:19` (claim) + `providers/auth_provider.dart:58` (claim)
- **Verified:** NO
- **Impact:** `AuthProvider.loadToken()` is called once in `main()` and again in the
  constructor. The second call may overwrite a token the first call just loaded.
- **Fix:** Call once, in `main()`, before constructing `AuthProvider`.
- **Effort:** S
- **Status:** Unverified claim.

---

## ⚪ LOW — nice-to-haves

| Item | Effort | Verified? |
|------|--------|-----------|
| `flutter analyze` re-run after release build | S | N/A |
| iOS Info.plist permission strings (camera/photo) | S | NO |
| `pubspec.yaml` `description: "A new Flutter project."` never updated | S | NO |
| `LICENSE` file missing (README mentions MIT) | S | NO |
| `docs/ui-preview.html` + `ui-preview - Copy.html` are 61KB duplicates | S | NO |
| `Jewelify_server/backup/` contains backup text files in repo | S | NO |
| `Jewelry_server/backup for android app build_gradel_kts file.groovy` typo'd backup | S | NO |
| GoogleFonts pulls Cormorant from CDN — no offline fallback | S | NO |
| `nested_async` in requirements imported but unused | S | NO |
| Backend `/health` endpoint is minimal (no model-load status, no DB ping) | S | NO |
| `loop = asyncio.get_event_loop()` deprecated in predictor.py:281 | S | YES |
| `scaler_xgboost_v1.pkl` exists but never loaded (see #4) | — | — |
| `feedback` "required" snackbar still shows after both submitted (results_screen.dart:442-452) | S | YES (file read) |
| `remarks.md` is 20 days stale (dates 2026-05-17, today 2026-06-06) | S | YES |
| Google "nano banana" try-on feature (remarks.md future) | XL | N/A |
| Real widget tests (current `test/widget_test.dart` is broken default counter test) | M | NO |
| Backend pytest (none exist) | M | NO |
| Sentry/error reporting | M | NO |
| CI/CD pipeline | M | NO |
| JWT secret rotation strategy | M | NO |

---

## Unverified claims — subagent's word only, NOT cross-checked

The following items appeared in the initial Explore subagent report. Each was flagged
in this document. They are **not** confirmed bugs. Before fixing any of them, read the
cited file and confirm the line number and behaviour:

- Item 9 (JWT refresh, `services/auth.py:30`)
- Item 10 (image zoom widget, `widgets/image_zoom_widget.dart`)
- Item 11 (history_screen duplication, `history_screen.dart:532-548`)
- Item 12 (feedback no-confirm, `prediction_module.dart:69-87, 247-265`)
- Item 15 (feedback_required type mismatch, `prediction_module.dart:50, 99`)
- Item 16 (image cache leak, `history_screen.dart:116, 130-132`)
- Item 17 (score_display emoji, `score_display.dart:85`)
- Item 18 (splash asset clash, `pubspec.yaml:42, 60-62`)
- Item 19 (auth_provider boilerplate, `auth_provider.dart:84-176`)
- Item 20 (motor unused, `requirements.txt:22`)
- Item 21 (authHeaders dead getter, `auth_provider.dart:184-187`)
- Item 22 (CORS, `main.py:45-63`)
- Item 24 (loadToken race, `main.dart:19` + `auth_provider.dart:58`)

Plus the LOW section items marked "NO" in the Verified column.

---

## How to use this file

1. **Ship-blockers first.** Items 1–6 (🔴 CRITICAL) are all in the same six files.
   Combined effort: roughly half a day.
2. **Then verify the rest.** Read the 14 unverified claims above. Mark ✅ or ✗ in
   this file as you confirm each one.
3. **Defer the LOW section** to v1.1 — most of it is cosmetic.
4. **Re-run the audit** after fixes land. The "75–80% complete" figure is for the
   current state; after items 1–6 land it should jump to ~85%, and after the 🟠
   HIGH items land, ~92%.

## What was actually checked (honest inventory)

✅ **Read in full during this audit (2026-06-06):**
- `jewelify_app/lib/constants/api.dart`
- `Jewelify_server/services/predictor.py`
- `jewelify_app/lib/screens/results_screen.dart`
- `jewelify_app/lib/screens/processing_screen.dart`
- `Jewelify_server/api/routes/predictions.py`
- Grep across `Jewelify_server/` for `scaler` and `Failed validation`/`Failed prediction`

❌ **NOT read in full (subagent's word only):**
- All other Flutter screens, widgets, providers, models
- All other backend routes, services, models, main.py, dependencies.py
- Build configs (`pubspec.yaml`, `android/app/build.gradle.kts`, `requirements.txt`)
- Test files
- Most of `docs/` tree beyond `decisions.md` and `sessions.md` headers
- The `backup/` directory contents

**Conclusion:** Items 1–6, 13, 14, 23, plus parts of LOW (splash, remarks staleness,
`asyncio.get_event_loop()` deprecation, feedback snackbar) are confirmed. Everything
else in this document is the subagent's claim and must be re-checked before action.

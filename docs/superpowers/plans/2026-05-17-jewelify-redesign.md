# Jewelify Full Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal**: Complete UI redesign (Terracotta & Cream theme), auth system overhaul (email+password+OTP), codebase cleanup, and UX improvements — keeping all existing prediction/history functionality intact.

**Architecture**: Provider state management unchanged. Named routes unchanged. New theme system replaces AppTheme. New auth screens replace mobile OTP flow. All API calls go through AuthProvider or direct HTTP with `ApiConstants.baseUrl`.

**Tech Stack**: Flutter 3.7+, Dart, Provider 6.x, http 1.3, FlutterSecureStorage, Google ML Kit, Cormorant Garamond font (new), Poppins (existing)

**Read before starting**: `CLAUDE.md` and `docs/decisions.md`

---

## Phase 1: Codebase Cleanup

### Task 1: Create API constants file

**Files:**
- Create: `jewelify_app/lib/constants/api.dart`

- [ ] Create file with single base URL constant:

```dart
class ApiConstants {
  static const String baseUrl = 'https://jewelify-server.onrender.com';
}
```

- [ ] Commit:
```bash
git add jewelify_app/lib/constants/api.dart
git commit -m "feat: centralize API base URL in ApiConstants"
```

---

### Task 2: Replace hardcoded URLs in all files

**Files:**
- Modify: `jewelify_app/lib/providers/auth_provider.dart`
- Modify: `jewelify_app/lib/screens/processing_screen.dart`
- Modify: `jewelify_app/lib/screens/results_screen.dart`
- Modify: `jewelify_app/lib/screens/history_screen.dart`

- [ ] In each file, add import:
```dart
import '../constants/api.dart';
```
(adjust relative path depth as needed — providers and screens are one level deep from lib/)

- [ ] Find and replace every occurrence of `'https://jewelify-server.onrender.com'` with `ApiConstants.baseUrl` in all 4 files.

- [ ] Verify no hardcoded URL remains:
```bash
grep -r "jewelify-server.onrender.com" jewelify_app/lib/
```
Expected: no output.

- [ ] Commit:
```bash
git add jewelify_app/lib/providers/auth_provider.dart jewelify_app/lib/screens/processing_screen.dart jewelify_app/lib/screens/results_screen.dart jewelify_app/lib/screens/history_screen.dart
git commit -m "refactor: replace hardcoded API URLs with ApiConstants.baseUrl"
```

---

### Task 3: Delete dead files and fix duplicates

**Files:**
- Delete: `jewelify_app/lib/temp_main.dart`
- Delete: `jewelify_app/lib/widgets/image_storage.dart` (keep screens/image_storage.dart)
- Delete: `jewelify_app/lib/widgets/action_buttons.dart` (unused)
- Delete: `jewelify_app/lib/backup_file.txt`

- [ ] Delete files:
```bash
rm jewelify_app/lib/temp_main.dart
rm jewelify_app/lib/widgets/image_storage.dart
rm jewelify_app/lib/widgets/action_buttons.dart
rm jewelify_app/lib/backup_file.txt
```

- [ ] Verify app still compiles:
```bash
cd jewelify_app && flutter analyze
```
Expected: no errors referencing deleted files.

- [ ] Commit:
```bash
git add -u
git commit -m "chore: remove dead files (temp_main, duplicate image_storage, unused action_buttons)"
```

---

### Task 4: Remove unused dependencies from pubspec.yaml

**Files:**
- Modify: `jewelify_app/pubspec.yaml`

- [ ] Remove these lines from `dependencies:`:
```yaml
  photo_view: ^0.15.0
  easy_image_viewer: ^1.5.1
  smooth_page_indicator: ^1.2.1
  cached_network_image: ^3.4.1
```

- [ ] Run:
```bash
cd jewelify_app && flutter pub get
```
Expected: resolves without errors.

- [ ] Verify no imports of removed packages remain:
```bash
grep -r "photo_view\|easy_image_viewer\|smooth_page_indicator\|cached_network_image" jewelify_app/lib/
```
Expected: no output.

- [ ] Commit:
```bash
git add jewelify_app/pubspec.yaml jewelify_app/pubspec.lock
git commit -m "chore: remove unused dependencies (photo_view, easy_image_viewer, smooth_page_indicator, cached_network_image)"
```

---

### Task 5: Fix deprecated .withOpacity() calls

**Files:**
- Modify: `jewelify_app/lib/screens/app_theme.dart`
- Modify: `jewelify_app/lib/screens/home_screen.dart`

- [ ] Find all `.withOpacity(` calls:
```bash
grep -rn "\.withOpacity(" jewelify_app/lib/
```

- [ ] Replace each `.withOpacity(x)` with `.withValues(alpha: x)` in every file listed.

Example — change:
```dart
color: textSecondaryLight.withOpacity(0.6)
```
To:
```dart
color: textSecondaryLight.withValues(alpha: 0.6)
```

- [ ] Verify no `.withOpacity` remains:
```bash
grep -r "\.withOpacity(" jewelify_app/lib/
```
Expected: no output.

- [ ] Run analyze:
```bash
cd jewelify_app && flutter analyze
```

- [ ] Commit:
```bash
git add jewelify_app/lib/
git commit -m "fix: replace deprecated withOpacity() with withValues(alpha:)"
```

---

## Phase 2: Auth System Overhaul (email + password + OTP)

### Task 6: Add Cormorant Garamond font files

**Files:**
- Create dir: `jewelify_app/assets/fonts/cormorant/`
- Modify: `jewelify_app/pubspec.yaml`

- [ ] Download Cormorant Garamond font files from Google Fonts (https://fonts.google.com/specimen/Cormorant+Garamond). Download the zip, extract these 4 files:
  - `CormorantGaramond-Regular.ttf`
  - `CormorantGaramond-Italic.ttf`
  - `CormorantGaramond-SemiBold.ttf`
  - `CormorantGaramond-SemiBoldItalic.ttf`

- [ ] Place all 4 files in `jewelify_app/assets/fonts/cormorant/`

- [ ] Add to `pubspec.yaml` under `flutter: fonts:`:
```yaml
    - family: CormorantGaramond
      fonts:
        - asset: assets/fonts/cormorant/CormorantGaramond-Regular.ttf
          weight: 400
        - asset: assets/fonts/cormorant/CormorantGaramond-Italic.ttf
          weight: 400
          style: italic
        - asset: assets/fonts/cormorant/CormorantGaramond-SemiBold.ttf
          weight: 600
        - asset: assets/fonts/cormorant/CormorantGaramond-SemiBoldItalic.ttf
          weight: 600
          style: italic
```

- [ ] Also update `flutter: assets:` to include font dir:
```yaml
    - assets/fonts/cormorant/
```

- [ ] Run:
```bash
cd jewelify_app && flutter pub get
```

- [ ] Commit:
```bash
git add jewelify_app/assets/fonts/cormorant/ jewelify_app/pubspec.yaml
git commit -m "feat: add Cormorant Garamond font (Regular, Italic, SemiBold, SemiBoldItalic)"
```

---

### Task 7: Rewrite AppTheme — Terracotta & Cream

**Files:**
- Rewrite: `jewelify_app/lib/screens/app_theme.dart`

- [ ] Replace entire file content with:

```dart
import 'package:flutter/material.dart';

class AppTheme {
  // ── Terracotta & Cream Palette ──────────────────────────────────────────
  static const Color background     = Color(0xFFFDF6EF); // cream
  static const Color surface        = Color(0xFFFFFFFF); // white cards
  static const Color softSurface    = Color(0xFFF5EDE4); // slightly warm
  static const Color border         = Color(0xFFEDE0D4); // dividers
  static const Color primary        = Color(0xFFB5451B); // terracotta CTA
  static const Color primaryDark    = Color(0xFF8B3214); // pressed state
  static const Color headingBrown   = Color(0xFF3D1F15); // dark headings
  static const Color appNameBrown   = Color(0xFF6B3A2A); // logo/app name
  static const Color mutedText      = Color(0xFFA07060); // secondary text
  static const Color errorColor     = Color(0xFFD32F2F);

  // Dark mode equivalents
  static const Color backgroundDark  = Color(0xFF1A0F09);
  static const Color surfaceDark     = Color(0xFF2C1A12);
  static const Color softSurfaceDark = Color(0xFF3A2318);
  static const Color borderDark      = Color(0xFF4A3228);
  static const Color mutedTextDark   = Color(0xFF8A6050);

  // ── Typography ──────────────────────────────────────────────────────────
  static const String _display = 'CormorantGaramond';
  static const String _body    = 'Poppins';

  static const TextStyle displayLarge = TextStyle(
    fontFamily: _display,
    fontSize: 28,
    fontWeight: FontWeight.w600,
    fontStyle: FontStyle.italic,
    color: appNameBrown,
    letterSpacing: 0.5,
  );

  static const TextStyle displayMedium = TextStyle(
    fontFamily: _display,
    fontSize: 22,
    fontWeight: FontWeight.w600,
    fontStyle: FontStyle.italic,
    color: headingBrown,
  );

  static const TextStyle titleStyle = TextStyle(
    fontFamily: _display,
    fontSize: 18,
    fontWeight: FontWeight.w600,
    fontStyle: FontStyle.italic,
    color: headingBrown,
  );

  static const TextStyle labelUppercase = TextStyle(
    fontFamily: _body,
    fontSize: 9,
    fontWeight: FontWeight.w700,
    letterSpacing: 2.0,
    color: mutedText,
  );

  static const TextStyle bodyMedium = TextStyle(
    fontFamily: _body,
    fontSize: 14,
    fontWeight: FontWeight.w400,
    color: headingBrown,
  );

  static const TextStyle bodySmall = TextStyle(
    fontFamily: _body,
    fontSize: 12,
    fontWeight: FontWeight.w400,
    color: mutedText,
  );

  // ── Button Styles ────────────────────────────────────────────────────────
  static final ButtonStyle primaryButton = ElevatedButton.styleFrom(
    backgroundColor: primary,
    foregroundColor: Colors.white,
    elevation: 0,
    padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 24),
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
    textStyle: const TextStyle(
      fontFamily: _body,
      fontSize: 13,
      fontWeight: FontWeight.w700,
      letterSpacing: 1.5,
    ),
  );

  static final ButtonStyle outlineButton = OutlinedButton.styleFrom(
    foregroundColor: primary,
    side: const BorderSide(color: primary, width: 1.5),
    padding: const EdgeInsets.symmetric(vertical: 13, horizontal: 24),
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
    textStyle: const TextStyle(
      fontFamily: _body,
      fontSize: 13,
      fontWeight: FontWeight.w600,
      letterSpacing: 1.5,
    ),
  );

  // ── Card Decoration ──────────────────────────────────────────────────────
  static final BoxDecoration cardDecoration = BoxDecoration(
    color: surface,
    borderRadius: BorderRadius.circular(12),
    border: Border.all(color: border),
  );

  static final BoxDecoration softCardDecoration = BoxDecoration(
    color: softSurface,
    borderRadius: BorderRadius.circular(12),
    border: Border.all(color: border),
  );

  // ── ThemeData ────────────────────────────────────────────────────────────
  static ThemeData get lightTheme => ThemeData(
    useMaterial3: true,
    fontFamily: _body,
    scaffoldBackgroundColor: background,
    colorScheme: const ColorScheme(
      brightness: Brightness.light,
      primary: primary,
      onPrimary: Colors.white,
      secondary: appNameBrown,
      onSecondary: Colors.white,
      error: errorColor,
      onError: Colors.white,
      surface: surface,
      onSurface: headingBrown,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: background,
      foregroundColor: appNameBrown,
      elevation: 0,
      centerTitle: false,
      titleTextStyle: TextStyle(
        fontFamily: _display,
        fontSize: 20,
        fontWeight: FontWeight.w600,
        fontStyle: FontStyle.italic,
        color: appNameBrown,
      ),
    ),
    cardTheme: CardTheme(
      color: surface,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: border),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(style: primaryButton),
    outlinedButtonTheme: OutlinedButtonThemeData(style: outlineButton),
    inputDecorationTheme: InputDecorationTheme(
      filled: false,
      enabledBorder: const UnderlineInputBorder(
        borderSide: BorderSide(color: border, width: 1.5),
      ),
      focusedBorder: const UnderlineInputBorder(
        borderSide: BorderSide(color: primary, width: 1.5),
      ),
      errorBorder: const UnderlineInputBorder(
        borderSide: BorderSide(color: errorColor),
      ),
      labelStyle: labelUppercase,
      hintStyle: const TextStyle(
        fontFamily: _body,
        fontSize: 14,
        color: mutedText,
      ),
    ),
    dividerTheme: const DividerThemeData(color: border, thickness: 1, space: 1),
    textTheme: const TextTheme(
      displayLarge: TextStyle(fontFamily: _display, fontStyle: FontStyle.italic),
      displayMedium: TextStyle(fontFamily: _display, fontStyle: FontStyle.italic),
      headlineLarge: TextStyle(fontFamily: _display, fontStyle: FontStyle.italic),
      headlineMedium: TextStyle(fontFamily: _display, fontStyle: FontStyle.italic),
      titleLarge: TextStyle(fontFamily: _display, fontStyle: FontStyle.italic, color: headingBrown),
      titleMedium: TextStyle(fontFamily: _body, color: headingBrown),
      bodyLarge: TextStyle(fontFamily: _body, color: headingBrown),
      bodyMedium: TextStyle(fontFamily: _body, color: headingBrown),
      bodySmall: TextStyle(fontFamily: _body, color: mutedText),
      labelSmall: TextStyle(fontFamily: _body, letterSpacing: 1.5, color: mutedText),
    ),
  );

  static ThemeData get darkTheme => ThemeData(
    useMaterial3: true,
    fontFamily: _body,
    scaffoldBackgroundColor: backgroundDark,
    colorScheme: const ColorScheme(
      brightness: Brightness.dark,
      primary: primary,
      onPrimary: Colors.white,
      secondary: Color(0xFFD4956B),
      onSecondary: Colors.white,
      error: Color(0xFFEF9A9A),
      onError: Colors.black,
      surface: surfaceDark,
      onSurface: Color(0xFFE8D5C4),
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: backgroundDark,
      foregroundColor: Color(0xFFD4956B),
      elevation: 0,
      centerTitle: false,
      titleTextStyle: TextStyle(
        fontFamily: _display,
        fontSize: 20,
        fontWeight: FontWeight.w600,
        fontStyle: FontStyle.italic,
        color: Color(0xFFD4956B),
      ),
    ),
    cardTheme: CardTheme(
      color: surfaceDark,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: borderDark),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(style: primaryButton),
    outlinedButtonTheme: OutlinedButtonThemeData(style: outlineButton),
    inputDecorationTheme: InputDecorationTheme(
      filled: false,
      enabledBorder: const UnderlineInputBorder(
        borderSide: BorderSide(color: borderDark, width: 1.5),
      ),
      focusedBorder: const UnderlineInputBorder(
        borderSide: BorderSide(color: primary, width: 1.5),
      ),
      hintStyle: const TextStyle(fontFamily: _body, fontSize: 14, color: mutedTextDark),
    ),
    dividerTheme: const DividerThemeData(color: borderDark, thickness: 1, space: 1),
  );
}
```

- [ ] Run:
```bash
cd jewelify_app && flutter analyze
```
Fix any type errors.

- [ ] Commit:
```bash
git add jewelify_app/lib/screens/app_theme.dart
git commit -m "feat: rewrite AppTheme with Terracotta & Cream design system"
```

---

### Task 8: Update UserOut model and AuthProvider for email auth

**Files:**
- Modify: `jewelify_app/lib/providers/auth_provider.dart`

- [ ] Replace `UserOut` class — change `mobileNo` to `email`:

```dart
class UserOut {
  final String id;
  final String? username;
  final String? name;
  final String email;
  final String? createdAt;
  final String? accessToken;

  UserOut({
    required this.id,
    this.username,
    this.name,
    required this.email,
    this.createdAt,
    this.accessToken,
  });

  factory UserOut.fromJson(Map<String, dynamic> json) {
    final id = json['id'] as String?;
    final email = json['email'] as String?;
    if (id == null || email == null) {
      throw const FormatException('Invalid JSON: id and email are required');
    }
    return UserOut(
      id: id,
      username: json['username'] as String?,
      name: json['name'] as String?,
      email: email,
      createdAt: json['created_at'] as String?,
      accessToken: json['access_token'] as String?,
    );
  }
}
```

- [ ] Replace `AuthProvider` class with full email-based implementation:

```dart
class AuthProvider with ChangeNotifier {
  String? _token;
  String? _userId;
  String? _username;
  String? _name;
  String? _email;

  final _storage = const FlutterSecureStorage();

  String? get token => _token;
  String? get userId => _userId;
  String? get username => _username;
  String? get name => _name;
  String? get email => _email;
  bool get isAuthenticated => _token != null;

  AuthProvider() {
    loadToken();
  }

  Future<void> loadToken() async {
    _token    = await _storage.read(key: 'auth_token');
    _userId   = await _storage.read(key: 'user_id');
    _username = await _storage.read(key: 'username');
    _name     = await _storage.read(key: 'name');
    _email    = await _storage.read(key: 'email');
    notifyListeners();
  }

  Future<void> _saveUser(UserOut user, String token) async {
    await _storage.write(key: 'auth_token', value: token);
    await _storage.write(key: 'user_id',    value: user.id);
    await _storage.write(key: 'username',   value: user.username ?? '');
    await _storage.write(key: 'name',       value: user.name ?? '');
    await _storage.write(key: 'email',      value: user.email);
    _token    = token;
    _userId   = user.id;
    _username = user.username;
    _name     = user.name;
    _email    = user.email;
    notifyListeners();
  }

  // Login with email + password → returns JWT
  Future<void> login(String email, String password) async {
    final res = await http.post(
      Uri.parse('${ApiConstants.baseUrl}/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'email': email, 'password': password}),
    );
    if (res.statusCode != 200) {
      final body = jsonDecode(res.body);
      throw Exception(body['detail'] ?? 'Login failed');
    }
    final body = jsonDecode(res.body);
    final token = body['access_token'] as String;
    final user = UserOut.fromJson(body['user'] as Map<String, dynamic>? ?? body);
    await _saveUser(user, token);
  }

  // Step 1 of registration: send OTP to email
  Future<void> sendRegistrationOtp(String email) async {
    final res = await http.post(
      Uri.parse('${ApiConstants.baseUrl}/auth/send-otp'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'email': email}),
    );
    if (res.statusCode != 200) {
      final body = jsonDecode(res.body);
      throw Exception(body['detail'] ?? 'Failed to send OTP');
    }
  }

  // Step 2: complete registration with OTP
  Future<void> register({
    required String name,
    required String username,
    required String email,
    required String password,
    required String otp,
  }) async {
    final res = await http.post(
      Uri.parse('${ApiConstants.baseUrl}/auth/register'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'name': name,
        'username': username,
        'email': email,
        'password': password,
        'otp': otp,
      }),
    );
    if (res.statusCode != 200 && res.statusCode != 201) {
      final body = jsonDecode(res.body);
      throw Exception(body['detail'] ?? 'Registration failed');
    }
    final body = jsonDecode(res.body);
    final token = body['access_token'] as String;
    final user = UserOut.fromJson(body['user'] as Map<String, dynamic>? ?? body);
    await _saveUser(user, token);
  }

  // Forgot password — step 1: send reset OTP
  Future<void> sendForgotPasswordOtp(String email) async {
    final res = await http.post(
      Uri.parse('${ApiConstants.baseUrl}/auth/forgot-password'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'email': email}),
    );
    if (res.statusCode != 200) {
      final body = jsonDecode(res.body);
      throw Exception(body['detail'] ?? 'Failed to send reset OTP');
    }
  }

  // Forgot password — step 2: verify OTP + set new password
  Future<void> resetPassword({
    required String email,
    required String otp,
    required String newPassword,
  }) async {
    final res = await http.post(
      Uri.parse('${ApiConstants.baseUrl}/auth/reset-password'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'email': email, 'otp': otp, 'new_password': newPassword}),
    );
    if (res.statusCode != 200) {
      final body = jsonDecode(res.body);
      throw Exception(body['detail'] ?? 'Password reset failed');
    }
  }

  Future<void> logout() async {
    await _storage.deleteAll();
    _token = _userId = _username = _name = _email = null;
    notifyListeners();
  }

  Map<String, String> get authHeaders => {
    'Authorization': 'Bearer $_token',
    'Content-Type': 'application/json',
  };
}
```

- [ ] Run:
```bash
cd jewelify_app && flutter analyze
```

- [ ] Commit:
```bash
git add jewelify_app/lib/providers/auth_provider.dart
git commit -m "feat: overhaul AuthProvider for email+password+OTP auth (replace mobile OTP)"
```

---

### Task 9: New Login Screen (email + password, terracotta style)

**Files:**
- Rewrite: `jewelify_app/lib/screens/login_screen.dart`

- [ ] Replace entire file:

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../screens/app_theme.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _formKey = GlobalKey<FormState>();
  final _emailCtrl = TextEditingController();
  final _passCtrl  = TextEditingController();
  bool _obscure   = true;
  bool _loading   = false;

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _loading = true);
    try {
      await context.read<AuthProvider>().login(
        _emailCtrl.text.trim(),
        _passCtrl.text,
      );
      if (!mounted) return;
      Navigator.pushNamedAndRemoveUntil(context, '/home', (_) => false);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.toString().replaceFirst('Exception: ', ''))),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 28),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 56),
                Center(
                  child: Column(
                    children: [
                      Text('Jewelify', style: AppTheme.displayLarge),
                      const SizedBox(height: 4),
                      Text('adorned beautifully', style: AppTheme.labelUppercase.copyWith(letterSpacing: 3)),
                      const SizedBox(height: 12),
                      Container(width: 40, height: 1.5, color: AppTheme.primary),
                    ],
                  ),
                ),
                const SizedBox(height: 48),
                Text('Sign in to continue', style: AppTheme.titleStyle),
                const SizedBox(height: 28),
                TextFormField(
                  controller: _emailCtrl,
                  keyboardType: TextInputType.emailAddress,
                  decoration: const InputDecoration(labelText: 'EMAIL ADDRESS'),
                  style: AppTheme.bodyMedium,
                  validator: (v) {
                    if (v == null || v.trim().isEmpty) return 'Enter your email';
                    if (!v.contains('@')) return 'Enter a valid email';
                    return null;
                  },
                ),
                const SizedBox(height: 20),
                TextFormField(
                  controller: _passCtrl,
                  obscureText: _obscure,
                  decoration: InputDecoration(
                    labelText: 'PASSWORD',
                    suffixIcon: IconButton(
                      icon: Icon(
                        _obscure ? Icons.visibility_off_outlined : Icons.visibility_outlined,
                        color: AppTheme.mutedText,
                        size: 20,
                      ),
                      onPressed: () => setState(() => _obscure = !_obscure),
                    ),
                  ),
                  style: AppTheme.bodyMedium,
                  validator: (v) => (v == null || v.isEmpty) ? 'Enter your password' : null,
                ),
                const SizedBox(height: 10),
                Align(
                  alignment: Alignment.centerRight,
                  child: TextButton(
                    onPressed: () => Navigator.pushNamed(context, '/forgot-password'),
                    style: TextButton.styleFrom(foregroundColor: AppTheme.primary, padding: EdgeInsets.zero),
                    child: Text('Forgot password?', style: AppTheme.bodySmall.copyWith(color: AppTheme.primary)),
                  ),
                ),
                const SizedBox(height: 24),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: _loading ? null : _submit,
                    style: AppTheme.primaryButton,
                    child: _loading
                        ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                        : const Text('SIGN IN'),
                  ),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    const Expanded(child: Divider(color: AppTheme.border)),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      child: Text('or', style: AppTheme.bodySmall),
                    ),
                    const Expanded(child: Divider(color: AppTheme.border)),
                  ],
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton(
                    onPressed: () => Navigator.pushNamed(context, '/register'),
                    style: AppTheme.outlineButton,
                    child: const Text('CREATE ACCOUNT'),
                  ),
                ),
                const SizedBox(height: 32),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
```

- [ ] Run `flutter analyze`. Fix any issues.

- [ ] Commit:
```bash
git add jewelify_app/lib/screens/login_screen.dart
git commit -m "feat: rewrite login screen with email+password auth and Terracotta theme"
```

---

### Task 10: New Register Screen (name + username + email + password + OTP)

**Files:**
- Rewrite: `jewelify_app/lib/screens/temp_registration.dart` → rename to `register_screen.dart`

- [ ] Create new file `jewelify_app/lib/screens/register_screen.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../screens/app_theme.dart';

enum _Step { details, otp }

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _detailsKey = GlobalKey<FormState>();
  final _nameCtrl     = TextEditingController();
  final _usernameCtrl = TextEditingController();
  final _emailCtrl    = TextEditingController();
  final _passCtrl     = TextEditingController();
  final _otpCtrl      = TextEditingController();

  _Step _step   = _Step.details;
  bool _obscure = true;
  bool _loading = false;
  int  _resendSeconds = 0;

  @override
  void dispose() {
    _nameCtrl.dispose(); _usernameCtrl.dispose();
    _emailCtrl.dispose(); _passCtrl.dispose(); _otpCtrl.dispose();
    super.dispose();
  }

  Future<void> _sendOtp() async {
    if (!_detailsKey.currentState!.validate()) return;
    setState(() => _loading = true);
    try {
      await context.read<AuthProvider>().sendRegistrationOtp(_emailCtrl.text.trim());
      if (!mounted) return;
      setState(() { _step = _Step.otp; _resendSeconds = 60; });
      _startResendTimer();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.toString().replaceFirst('Exception: ', ''))),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _startResendTimer() {
    Future.doWhile(() async {
      await Future.delayed(const Duration(seconds: 1));
      if (!mounted) return false;
      setState(() { if (_resendSeconds > 0) _resendSeconds--; });
      return _resendSeconds > 0;
    });
  }

  Future<void> _resendOtp() async {
    if (_resendSeconds > 0) return;
    setState(() => _loading = true);
    try {
      await context.read<AuthProvider>().sendRegistrationOtp(_emailCtrl.text.trim());
      if (!mounted) return;
      setState(() => _resendSeconds = 60);
      _startResendTimer();
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('OTP resent to your email')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.toString().replaceFirst('Exception: ', ''))),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _verify() async {
    if (_otpCtrl.text.trim().length != 6) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Enter the 6-digit OTP')),
      );
      return;
    }
    setState(() => _loading = true);
    try {
      await context.read<AuthProvider>().register(
        name:     _nameCtrl.text.trim(),
        username: _usernameCtrl.text.trim(),
        email:    _emailCtrl.text.trim(),
        password: _passCtrl.text,
        otp:      _otpCtrl.text.trim(),
      );
      if (!mounted) return;
      Navigator.pushNamedAndRemoveUntil(context, '/home', (_) => false);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.toString().replaceFirst('Exception: ', ''))),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppTheme.appNameBrown),
          onPressed: _step == _Step.otp
              ? () => setState(() { _step = _Step.details; _otpCtrl.clear(); })
              : () => Navigator.pop(context),
        ),
        elevation: 0,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 28),
          child: _step == _Step.details ? _buildDetails() : _buildOtp(),
        ),
      ),
    );
  }

  Widget _buildDetails() {
    return Form(
      key: _detailsKey,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 8),
          Text('Create account', style: AppTheme.displayMedium),
          const SizedBox(height: 4),
          Text('Join Jewelify', style: AppTheme.bodySmall),
          const SizedBox(height: 32),
          TextFormField(
            controller: _nameCtrl,
            decoration: const InputDecoration(labelText: 'FULL NAME'),
            style: AppTheme.bodyMedium,
            textCapitalization: TextCapitalization.words,
            validator: (v) => (v == null || v.trim().isEmpty) ? 'Enter your name' : null,
          ),
          const SizedBox(height: 20),
          TextFormField(
            controller: _usernameCtrl,
            decoration: const InputDecoration(labelText: 'USERNAME'),
            style: AppTheme.bodyMedium,
            validator: (v) {
              if (v == null || v.trim().isEmpty) return 'Choose a username';
              if (v.trim().length < 3) return 'At least 3 characters';
              return null;
            },
          ),
          const SizedBox(height: 20),
          TextFormField(
            controller: _emailCtrl,
            keyboardType: TextInputType.emailAddress,
            decoration: const InputDecoration(labelText: 'EMAIL ADDRESS'),
            style: AppTheme.bodyMedium,
            validator: (v) {
              if (v == null || v.trim().isEmpty) return 'Enter your email';
              if (!v.contains('@')) return 'Enter a valid email';
              return null;
            },
          ),
          const SizedBox(height: 20),
          TextFormField(
            controller: _passCtrl,
            obscureText: _obscure,
            decoration: InputDecoration(
              labelText: 'PASSWORD',
              suffixIcon: IconButton(
                icon: Icon(
                  _obscure ? Icons.visibility_off_outlined : Icons.visibility_outlined,
                  color: AppTheme.mutedText, size: 20,
                ),
                onPressed: () => setState(() => _obscure = !_obscure),
              ),
            ),
            style: AppTheme.bodyMedium,
            validator: (v) {
              if (v == null || v.isEmpty) return 'Enter a password';
              if (v.length < 6) return 'At least 6 characters';
              return null;
            },
          ),
          const SizedBox(height: 36),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: _loading ? null : _sendOtp,
              style: AppTheme.primaryButton,
              child: _loading
                  ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                  : const Text('SEND OTP'),
            ),
          ),
          const SizedBox(height: 16),
          Center(
            child: TextButton(
              onPressed: () => Navigator.pushReplacementNamed(context, '/login'),
              style: TextButton.styleFrom(foregroundColor: AppTheme.primary),
              child: Text('Already have an account? Sign in',
                  style: AppTheme.bodySmall.copyWith(color: AppTheme.primary)),
            ),
          ),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  Widget _buildOtp() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 8),
        Text('Verify email', style: AppTheme.displayMedium),
        const SizedBox(height: 8),
        Text('Enter the 6-digit code sent to', style: AppTheme.bodySmall),
        const SizedBox(height: 2),
        Text(_emailCtrl.text.trim(), style: AppTheme.bodyMedium.copyWith(color: AppTheme.primary)),
        const SizedBox(height: 36),
        TextFormField(
          controller: _otpCtrl,
          keyboardType: TextInputType.number,
          maxLength: 6,
          decoration: const InputDecoration(
            labelText: 'OTP CODE',
            counterText: '',
          ),
          style: AppTheme.bodyMedium.copyWith(fontSize: 22, letterSpacing: 8),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Text('Didn\'t receive it? ', style: AppTheme.bodySmall),
            TextButton(
              onPressed: _resendSeconds == 0 && !_loading ? _resendOtp : null,
              style: TextButton.styleFrom(
                foregroundColor: AppTheme.primary,
                padding: EdgeInsets.zero,
                minimumSize: Size.zero,
              ),
              child: Text(
                _resendSeconds > 0 ? 'Resend in ${_resendSeconds}s' : 'Resend now',
                style: AppTheme.bodySmall.copyWith(
                  color: _resendSeconds > 0 ? AppTheme.mutedText : AppTheme.primary,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 36),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton(
            onPressed: _loading ? null : _verify,
            style: AppTheme.primaryButton,
            child: _loading
                ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Text('VERIFY & CREATE ACCOUNT'),
          ),
        ),
        const SizedBox(height: 32),
      ],
    );
  }
}
```

- [ ] Delete old registration file:
```bash
rm jewelify_app/lib/screens/temp_registration.dart
```

- [ ] Update `main.dart` import from `temp_registration.dart` to `register_screen.dart` (same class name `RegisterScreen` — no other changes needed).

- [ ] Run `flutter analyze`.

- [ ] Commit:
```bash
git add jewelify_app/lib/screens/register_screen.dart
git commit -m "feat: new register screen with name/username/email/password/OTP flow and Terracotta theme"
```

---

### Task 11: Forgot Password Screen

**Files:**
- Create: `jewelify_app/lib/screens/forgot_password_screen.dart`

- [ ] Create file:

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../screens/app_theme.dart';

enum _FPStep { email, otp, newPass }

class ForgotPasswordScreen extends StatefulWidget {
  const ForgotPasswordScreen({super.key});

  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
  final _emailCtrl   = TextEditingController();
  final _otpCtrl     = TextEditingController();
  final _passCtrl    = TextEditingController();
  _FPStep _step      = _FPStep.email;
  bool _loading      = false;
  bool _obscure      = true;
  int  _resendSeconds = 0;

  @override
  void dispose() {
    _emailCtrl.dispose(); _otpCtrl.dispose(); _passCtrl.dispose();
    super.dispose();
  }

  void _startResendTimer() {
    Future.doWhile(() async {
      await Future.delayed(const Duration(seconds: 1));
      if (!mounted) return false;
      setState(() { if (_resendSeconds > 0) _resendSeconds--; });
      return _resendSeconds > 0;
    });
  }

  Future<void> _sendOtp() async {
    final email = _emailCtrl.text.trim();
    if (email.isEmpty || !email.contains('@')) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Enter a valid email address')),
      );
      return;
    }
    setState(() => _loading = true);
    try {
      await context.read<AuthProvider>().sendForgotPasswordOtp(email);
      if (!mounted) return;
      setState(() { _step = _FPStep.otp; _resendSeconds = 60; });
      _startResendTimer();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.toString().replaceFirst('Exception: ', ''))),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _verifyOtp() async {
    if (_otpCtrl.text.trim().length != 6) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Enter the 6-digit OTP')),
      );
      return;
    }
    setState(() => _step = _FPStep.newPass);
  }

  Future<void> _resetPassword() async {
    if (_passCtrl.text.length < 6) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Password must be at least 6 characters')),
      );
      return;
    }
    setState(() => _loading = true);
    try {
      await context.read<AuthProvider>().resetPassword(
        email: _emailCtrl.text.trim(),
        otp: _otpCtrl.text.trim(),
        newPassword: _passCtrl.text,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Password reset successfully. Please sign in.')),
      );
      Navigator.pushNamedAndRemoveUntil(context, '/login', (_) => false);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.toString().replaceFirst('Exception: ', ''))),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppTheme.appNameBrown),
          onPressed: () {
            if (_step == _FPStep.otp) setState(() => _step = _FPStep.email);
            else if (_step == _FPStep.newPass) setState(() => _step = _FPStep.otp);
            else Navigator.pop(context);
          },
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 28),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 8),
              Text('Reset password', style: AppTheme.displayMedium),
              const SizedBox(height: 36),
              if (_step == _FPStep.email) ...[
                Text('Enter the email linked to your account.', style: AppTheme.bodySmall),
                const SizedBox(height: 24),
                TextFormField(
                  controller: _emailCtrl,
                  keyboardType: TextInputType.emailAddress,
                  decoration: const InputDecoration(labelText: 'EMAIL ADDRESS'),
                  style: AppTheme.bodyMedium,
                ),
                const SizedBox(height: 36),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: _loading ? null : _sendOtp,
                    style: AppTheme.primaryButton,
                    child: _loading
                        ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                        : const Text('SEND RESET CODE'),
                  ),
                ),
              ],
              if (_step == _FPStep.otp) ...[
                Text('Enter the 6-digit code sent to', style: AppTheme.bodySmall),
                const SizedBox(height: 2),
                Text(_emailCtrl.text.trim(), style: AppTheme.bodyMedium.copyWith(color: AppTheme.primary)),
                const SizedBox(height: 24),
                TextFormField(
                  controller: _otpCtrl,
                  keyboardType: TextInputType.number,
                  maxLength: 6,
                  decoration: const InputDecoration(labelText: 'RESET CODE', counterText: ''),
                  style: AppTheme.bodyMedium.copyWith(fontSize: 22, letterSpacing: 8),
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Text('Didn\'t receive it? ', style: AppTheme.bodySmall),
                    TextButton(
                      onPressed: _resendSeconds == 0 ? _sendOtp : null,
                      style: TextButton.styleFrom(foregroundColor: AppTheme.primary, padding: EdgeInsets.zero, minimumSize: Size.zero),
                      child: Text(
                        _resendSeconds > 0 ? 'Resend in ${_resendSeconds}s' : 'Resend now',
                        style: AppTheme.bodySmall.copyWith(
                          color: _resendSeconds > 0 ? AppTheme.mutedText : AppTheme.primary,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 36),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: _loading ? null : _verifyOtp,
                    style: AppTheme.primaryButton,
                    child: const Text('CONTINUE'),
                  ),
                ),
              ],
              if (_step == _FPStep.newPass) ...[
                Text('Choose a new password for your account.', style: AppTheme.bodySmall),
                const SizedBox(height: 24),
                TextFormField(
                  controller: _passCtrl,
                  obscureText: _obscure,
                  decoration: InputDecoration(
                    labelText: 'NEW PASSWORD',
                    suffixIcon: IconButton(
                      icon: Icon(_obscure ? Icons.visibility_off_outlined : Icons.visibility_outlined,
                          color: AppTheme.mutedText, size: 20),
                      onPressed: () => setState(() => _obscure = !_obscure),
                    ),
                  ),
                  style: AppTheme.bodyMedium,
                ),
                const SizedBox(height: 36),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: _loading ? null : _resetPassword,
                    style: AppTheme.primaryButton,
                    child: _loading
                        ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                        : const Text('RESET PASSWORD'),
                  ),
                ),
              ],
              const SizedBox(height: 32),
            ],
          ),
        ),
      ),
    );
  }
}
```

- [ ] Run `flutter analyze`.

- [ ] Commit:
```bash
git add jewelify_app/lib/screens/forgot_password_screen.dart
git commit -m "feat: add forgot password screen (email OTP + reset password flow)"
```

---

### Task 12: Update main.dart routes

**Files:**
- Modify: `jewelify_app/lib/main.dart`

- [ ] Update imports to use `register_screen.dart` instead of `temp_registration.dart`:
```dart
import 'screens/register_screen.dart';
import 'screens/forgot_password_screen.dart';
```

- [ ] Add `/forgot-password` route in the routes map:
```dart
'/forgot-password': (context) => const ForgotPasswordScreen(),
```

- [ ] Remove old import of `temp_registration.dart`.

- [ ] Run `flutter analyze`.

- [ ] Commit:
```bash
git add jewelify_app/lib/main.dart
git commit -m "feat: add forgot-password route, update import to register_screen"
```

---

## Phase 3: UI Redesign — All Screens

### Task 13: Skeleton Loader Widget

**Files:**
- Create: `jewelify_app/lib/widgets/skeleton_loader.dart`

- [ ] Create file:

```dart
import 'package:flutter/material.dart';
import '../screens/app_theme.dart';

class SkeletonLoader extends StatefulWidget {
  final double width;
  final double height;
  final double borderRadius;

  const SkeletonLoader({
    super.key,
    required this.width,
    required this.height,
    this.borderRadius = 8,
  });

  @override
  State<SkeletonLoader> createState() => _SkeletonLoaderState();
}

class _SkeletonLoaderState extends State<SkeletonLoader> with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 1200))
      ..repeat(reverse: true);
    _anim = Tween<double>(begin: 0.3, end: 1.0).animate(
      CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _anim,
      builder: (_, __) => Container(
        width: widget.width,
        height: widget.height,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(widget.borderRadius),
          color: AppTheme.border.withValues(alpha: _anim.value),
        ),
      ),
    );
  }
}

// Convenience: full-width skeleton
class SkeletonLine extends StatelessWidget {
  final double height;
  final double? width;
  final double borderRadius;

  const SkeletonLine({super.key, this.height = 14, this.width, this.borderRadius = 4});

  @override
  Widget build(BuildContext context) {
    return SkeletonLoader(
      width: width ?? double.infinity,
      height: height,
      borderRadius: borderRadius,
    );
  }
}

// History item skeleton
class SkeletonHistoryItem extends StatelessWidget {
  const SkeletonHistoryItem({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: AppTheme.cardDecoration,
      child: const Row(
        children: [
          SkeletonLoader(width: 40, height: 40, borderRadius: 8),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SkeletonLine(height: 12, width: 120),
                SizedBox(height: 6),
                SkeletonLine(height: 10, width: 80),
              ],
            ),
          ),
          SkeletonLine(height: 10, width: 40),
        ],
      ),
    );
  }
}
```

- [ ] Commit:
```bash
git add jewelify_app/lib/widgets/skeleton_loader.dart
git commit -m "feat: add SkeletonLoader widget for shimmer loading states"
```

---

### Task 14: Redesign Home Screen

**Files:**
- Rewrite: `jewelify_app/lib/screens/home_screen.dart`

- [ ] Replace entire file:

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../screens/app_theme.dart';

class HomeScreen extends StatelessWidget {
  final VoidCallback toggleTheme;
  final bool isDarkMode;

  const HomeScreen({super.key, required this.toggleTheme, required this.isDarkMode});

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final displayName = auth.name ?? auth.username ?? 'there';

    return Scaffold(
      backgroundColor: AppTheme.background,
      body: SafeArea(
        child: CustomScrollView(
          slivers: [
            SliverToBoxAdapter(child: _buildTopBar(context, auth, displayName)),
            SliverToBoxAdapter(child: _buildHeroCard(context)),
            SliverToBoxAdapter(child: _buildSectionLabel('QUICK ACTIONS')),
            SliverToBoxAdapter(child: _buildActionGrid(context)),
            SliverToBoxAdapter(child: _buildSectionLabel('ABOUT JEWELIFY')),
            SliverToBoxAdapter(child: _buildAboutCard(context)),
            const SliverToBoxAdapter(child: SizedBox(height: 32)),
          ],
        ),
      ),
    );
  }

  Widget _buildTopBar(BuildContext context, AuthProvider auth, String displayName) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 20, 16, 8),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('WELCOME BACK', style: AppTheme.labelUppercase),
                const SizedBox(height: 2),
                Text(displayName, style: AppTheme.displayLarge.copyWith(fontSize: 22)),
              ],
            ),
          ),
          IconButton(
            icon: Icon(isDarkMode ? Icons.light_mode_outlined : Icons.dark_mode_outlined,
                color: AppTheme.appNameBrown),
            onPressed: toggleTheme,
          ),
          IconButton(
            icon: const Icon(Icons.logout_outlined, color: AppTheme.mutedText),
            onPressed: () {
              auth.logout();
              Navigator.pushNamedAndRemoveUntil(context, '/login', (_) => false);
            },
          ),
        ],
      ),
    );
  }

  Widget _buildHeroCard(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(20, 12, 20, 0),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFFB5451B), Color(0xFFD2691E)],
        ),
        borderRadius: BorderRadius.circular(20),
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('JEWELRY MATCHING', style: AppTheme.labelUppercase.copyWith(color: Colors.white60)),
          const SizedBox(height: 8),
          Text('Adorned\nbeautifully.', style: AppTheme.displayMedium.copyWith(color: Colors.white, fontSize: 26)),
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: () => Navigator.pushNamed(context, '/upload'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.white,
              foregroundColor: AppTheme.primary,
              elevation: 0,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
              textStyle: const TextStyle(fontFamily: 'Poppins', fontSize: 12, fontWeight: FontWeight.w700, letterSpacing: 1.5),
            ),
            child: const Text('UPLOAD PHOTO'),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionLabel(String label) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 24, 24, 12),
      child: Text(label, style: AppTheme.labelUppercase),
    );
  }

  Widget _buildActionGrid(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        children: [
          Expanded(
            child: _ActionCard(
              icon: Icons.camera_alt_outlined,
              title: 'New Analysis',
              subtitle: 'Upload face & jewelry photos',
              onTap: () => Navigator.pushNamed(context, '/upload'),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: _ActionCard(
              icon: Icons.history_outlined,
              title: 'History',
              subtitle: 'View past recommendations',
              onTap: () => Navigator.pushNamed(context, '/history'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAboutCard(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      padding: const EdgeInsets.all(18),
      decoration: AppTheme.softCardDecoration,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('How it works', style: AppTheme.titleStyle.copyWith(fontSize: 16)),
          const SizedBox(height: 12),
          _AboutStep(number: '01', text: 'Upload a clear face photo'),
          _AboutStep(number: '02', text: 'Add a photo of the jewelry'),
          _AboutStep(number: '03', text: 'AI analyzes compatibility'),
          _AboutStep(number: '04', text: 'Get your match score + recommendations'),
        ],
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _ActionCard({required this.icon, required this.title, required this.subtitle, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(18),
        decoration: AppTheme.cardDecoration,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: AppTheme.primary, size: 26),
            const SizedBox(height: 12),
            Text(title, style: AppTheme.bodyMedium.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 4),
            Text(subtitle, style: AppTheme.bodySmall, maxLines: 2),
          ],
        ),
      ),
    );
  }
}

class _AboutStep extends StatelessWidget {
  final String number;
  final String text;
  const _AboutStep({required this.number, required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          Text(number, style: AppTheme.labelUppercase.copyWith(color: AppTheme.primary, letterSpacing: 1)),
          const SizedBox(width: 14),
          Expanded(child: Text(text, style: AppTheme.bodySmall)),
        ],
      ),
    );
  }
}
```

- [ ] Run `flutter analyze`.

- [ ] Commit:
```bash
git add jewelify_app/lib/screens/home_screen.dart
git commit -m "feat: redesign home screen with hero card, action grid, and how-it-works section"
```

---

### Task 15: Apply terracotta theme to History Screen

**Files:**
- Modify: `jewelify_app/lib/screens/history_screen.dart`

- [ ] Apply these changes to `history_screen.dart`:
  - Replace `CircularProgressIndicator()` loading state with 5× `SkeletonHistoryItem()` in a `ListView`
  - Import `SkeletonHistoryItem` from `'../widgets/skeleton_loader.dart'`
  - Remove any hardcoded colors — use `AppTheme.*` constants
  - AppBar title: use `AppTheme.displayLarge` or `titleTextStyle` from theme
  - Card decoration: replace ad-hoc `BoxDecoration` with `AppTheme.cardDecoration`
  - Section headings: use `AppTheme.labelUppercase`
  - Score/match text: use `AppTheme.primary` color
  - Muted text: use `AppTheme.mutedText`

- [ ] Run `flutter analyze`.

- [ ] Commit:
```bash
git add jewelify_app/lib/screens/history_screen.dart
git commit -m "feat: apply Terracotta theme and skeleton loading to history screen"
```

---

### Task 16: Apply terracotta theme to Processing Screen + cold-start message

**Files:**
- Modify: `jewelify_app/lib/screens/processing_screen.dart`

- [ ] In `ProcessingScreen`, add a `_slowStart` bool state:
```dart
bool _slowStart = false;
```

- [ ] After 8 seconds of loading, set `_slowStart = true`:
```dart
Future.delayed(const Duration(seconds: 8), () {
  if (mounted && _isLoading) setState(() => _slowStart = true);
});
```

- [ ] In the loading widget, show additional message when `_slowStart == true`:
```dart
if (_slowStart)
  Padding(
    padding: const EdgeInsets.only(top: 12),
    child: Text(
      'Server is warming up,\nplease wait a moment...',
      textAlign: TextAlign.center,
      style: AppTheme.bodySmall.copyWith(color: AppTheme.mutedText),
    ),
  ),
```

- [ ] Replace spinner color with `AppTheme.primary`.
- [ ] Replace any hardcoded text colors with `AppTheme.*` constants.
- [ ] Add `import '../screens/app_theme.dart';` if not present.

- [ ] Run `flutter analyze`.

- [ ] Commit:
```bash
git add jewelify_app/lib/screens/processing_screen.dart
git commit -m "feat: apply Terracotta theme + cold-start UX message to processing screen"
```

---

### Task 17: Apply terracotta theme to Results Screen

**Files:**
- Modify: `jewelify_app/lib/screens/results_screen.dart`

- [ ] Apply these changes:
  - Score card background: `AppTheme.primary` (terracotta gradient: `Color(0xFFB5451B)` → `Color(0xFFD2691E)`)
  - Score percentage text: white, Cormorant Garamond font for the number
  - Recommendations section title: `AppTheme.labelUppercase`
  - Recommendation items: `AppTheme.cardDecoration`
  - Feedback stars: `AppTheme.primary` color
  - Back button/AppBar: terracotta theme colors
  - Loading state: `SkeletonHistoryItem` × 3 while polling
  - Import `SkeletonHistoryItem` from `'../widgets/skeleton_loader.dart'`
  - Replace all hardcoded colors with `AppTheme.*` constants

- [ ] Run `flutter analyze`.

- [ ] Commit:
```bash
git add jewelify_app/lib/screens/results_screen.dart
git commit -m "feat: apply Terracotta theme and skeleton loading to results screen"
```

---

### Task 18: Apply terracotta theme to Upload Screen

**Files:**
- Modify: `jewelify_app/lib/screens/upload_screen.dart`

- [ ] Apply these changes:
  - Image upload boxes: `AppTheme.softSurface` background, `AppTheme.border` border, dashed border effect
  - Upload icon color: `AppTheme.mutedText`
  - Section labels: `AppTheme.labelUppercase`
  - Primary button: `AppTheme.primaryButton` style
  - AppBar: terracotta theme
  - Replace all hardcoded colors with `AppTheme.*` constants

- [ ] Run `flutter analyze`.

- [ ] Commit:
```bash
git add jewelify_app/lib/screens/upload_screen.dart
git commit -m "feat: apply Terracotta theme to upload screen"
```

---

## Phase 4: Final Polish

### Task 19: Final analyze + clean developer.log calls

**Files:**
- Modify: all files with `developer.log` calls

- [ ] Find all developer.log usage:
```bash
grep -rn "developer.log" jewelify_app/lib/
```

- [ ] Wrap each in a compile-time constant check, or remove entirely for non-critical logs:
```dart
// Replace:
developer.log('message', name: 'ScreenName');
// With: (remove it, or wrap in assert for debug-only)
assert(() { debugPrint('message'); return true; }());
```

- [ ] Run final analyze:
```bash
cd jewelify_app && flutter analyze
```
Expected: 0 errors, 0 warnings.

- [ ] Commit:
```bash
git add jewelify_app/lib/
git commit -m "chore: remove/wrap developer.log calls, final analyzer clean"
```

---

### Task 20: Update pubspec for google_fonts (optional Cormorant fallback)

**Note**: Only needed if font `.ttf` files were NOT added manually in Task 6.

- [ ] If Cormorant Garamond .ttf files are in place (Task 6), skip this task.
- [ ] Otherwise add `google_fonts: ^6.2.1` to dependencies and use `GoogleFonts.cormorantGaramond()` in AppTheme.

---

## Self-Review Checklist

- [x] Every screen covered: login, register, forgot-password, home, upload, processing, results, history
- [x] Auth overhaul: email+password+OTP, forgot password, resend OTP with 60s cooldown
- [x] Theme: all AppTheme constants defined, light + dark, Cormorant + Poppins
- [x] Cleanup: temp files deleted, API URL centralized, unused deps removed, withOpacity fixed
- [x] UX: skeleton loaders in history + results, cold-start message in processing
- [x] No hardcoded API URLs remain after Task 2
- [x] No emoji icons (using Icons.* throughout)
- [x] All routes registered in main.dart (Task 12)
- [x] UserOut model uses `email` not `mobileNo`
- [x] `ApiConstants` imported with correct relative path in each file

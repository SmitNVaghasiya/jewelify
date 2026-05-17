# Jewelify — Flutter App

AI-powered jewelry recommendation mobile app. Upload a face photo + jewelry photo, get a compatibility score and personalized recommendations.

## Design System

**Theme**: Terracotta & Cream  
**Primary color**: `#b5451b` (terracotta)  
**Background**: `#fdf6ef` (warm cream)  
**Typography**: Cormorant Garamond (display) + Poppins (body)  
All constants in `lib/screens/app_theme.dart` — never hardcode colors.

## Auth Flow

Email + password + email OTP (replaced mobile OTP / Twilio):
- Register: name → username → email → password → OTP sent to email → verify
- Login: email + password → JWT
- Forgot password: email → OTP → new password

## Key Files

```
lib/
├── constants/api.dart           # API base URL (single source)
├── providers/auth_provider.dart # Auth state + all API calls
├── screens/
│   ├── app_theme.dart           # ALL colors, fonts, styles
│   ├── login_screen.dart
│   ├── register_screen.dart
│   ├── forgot_password_screen.dart
│   ├── home_screen.dart
│   ├── upload_screen.dart
│   ├── processing_screen.dart
│   └── results_screen.dart
└── widgets/
    ├── skeleton_loader.dart     # Shimmer loading widgets
    └── ...
```

## Setup

```bash
flutter pub get
flutter run
```

## Rules

- Use `AppTheme.*` constants, never hardcode colors
- Use `ApiConstants.baseUrl`, never hardcode API URL  
- Use `.withValues(alpha:)` not `.withOpacity()` (Flutter 3.27+)
- No `developer.log` in production code

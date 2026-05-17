import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // ── Terracotta & Cream Palette ──────────────────────────────────────────
  static const Color background     = Color(0xFFFDF6EF);
  static const Color surface        = Color(0xFFFFFFFF);
  static const Color softSurface    = Color(0xFFF5EDE4);
  static const Color border         = Color(0xFFEDE0D4);
  static const Color primary        = Color(0xFFB5451B);
  static const Color primaryDark    = Color(0xFF8B3214);
  static const Color headingBrown   = Color(0xFF3D1F15);
  static const Color appNameBrown   = Color(0xFF6B3A2A);
  static const Color mutedText      = Color(0xFFA07060);
  static const Color errorColor     = Color(0xFFD32F2F);

  static const Color backgroundDark  = Color(0xFF1A0F09);
  static const Color surfaceDark     = Color(0xFF2C1A12);
  static const Color softSurfaceDark = Color(0xFF3A2318);
  static const Color borderDark      = Color(0xFF4A3228);
  static const Color mutedTextDark   = Color(0xFF8A6050);

  // ── Typography helpers ──────────────────────────────────────────────────
  static TextStyle get displayLarge => GoogleFonts.cormorantGaramond(
    fontSize: 28, fontWeight: FontWeight.w600, fontStyle: FontStyle.italic,
    color: appNameBrown, letterSpacing: 0.5,
  );

  static TextStyle get displayMedium => GoogleFonts.cormorantGaramond(
    fontSize: 22, fontWeight: FontWeight.w600, fontStyle: FontStyle.italic,
    color: headingBrown,
  );

  static TextStyle get titleStyle => GoogleFonts.cormorantGaramond(
    fontSize: 18, fontWeight: FontWeight.w600, fontStyle: FontStyle.italic,
    color: headingBrown,
  );

  static TextStyle get labelUppercase => GoogleFonts.poppins(
    fontSize: 9, fontWeight: FontWeight.w700, letterSpacing: 2.0, color: mutedText,
  );

  static TextStyle get bodyMedium => GoogleFonts.poppins(
    fontSize: 14, fontWeight: FontWeight.w400, color: headingBrown,
  );

  static TextStyle get bodySmall => GoogleFonts.poppins(
    fontSize: 12, fontWeight: FontWeight.w400, color: mutedText,
  );

  // ── Button Styles ────────────────────────────────────────────────────────
  static ButtonStyle get primaryButton => ElevatedButton.styleFrom(
    backgroundColor: primary,
    foregroundColor: Colors.white,
    elevation: 0,
    padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 24),
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
    textStyle: GoogleFonts.poppins(fontSize: 13, fontWeight: FontWeight.w700, letterSpacing: 1.5),
  );

  static ButtonStyle get outlineButton => OutlinedButton.styleFrom(
    foregroundColor: primary,
    side: const BorderSide(color: primary, width: 1.5),
    padding: const EdgeInsets.symmetric(vertical: 13, horizontal: 24),
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
    textStyle: GoogleFonts.poppins(fontSize: 13, fontWeight: FontWeight.w600, letterSpacing: 1.5),
  );

  // ── Card Decorations ─────────────────────────────────────────────────────
  static BoxDecoration get cardDecoration => BoxDecoration(
    color: surface,
    borderRadius: BorderRadius.circular(12),
    border: Border.all(color: border),
  );

  static BoxDecoration get softCardDecoration => BoxDecoration(
    color: softSurface,
    borderRadius: BorderRadius.circular(12),
    border: Border.all(color: border),
  );

  // ── ThemeData ────────────────────────────────────────────────────────────
  static ThemeData get lightTheme => ThemeData(
    useMaterial3: true,
    textTheme: GoogleFonts.poppinsTextTheme(),
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
    appBarTheme: AppBarTheme(
      backgroundColor: background,
      foregroundColor: appNameBrown,
      elevation: 0,
      centerTitle: false,
      titleTextStyle: GoogleFonts.cormorantGaramond(
        fontSize: 20, fontWeight: FontWeight.w600, fontStyle: FontStyle.italic,
        color: appNameBrown,
      ),
    ),
    cardTheme: CardThemeData(
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
      focusedErrorBorder: const UnderlineInputBorder(
        borderSide: BorderSide(color: errorColor, width: 1.5),
      ),
      labelStyle: GoogleFonts.poppins(fontSize: 9, fontWeight: FontWeight.w700, letterSpacing: 2.0, color: mutedText),
      hintStyle: GoogleFonts.poppins(fontSize: 14, color: mutedText),
    ),
    dividerTheme: const DividerThemeData(color: border, thickness: 1, space: 1),
  );

  static ThemeData get darkTheme => ThemeData(
    useMaterial3: true,
    textTheme: GoogleFonts.poppinsTextTheme(ThemeData.dark().textTheme),
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
    appBarTheme: AppBarTheme(
      backgroundColor: backgroundDark,
      foregroundColor: const Color(0xFFD4956B),
      elevation: 0,
      centerTitle: false,
      titleTextStyle: GoogleFonts.cormorantGaramond(
        fontSize: 20, fontWeight: FontWeight.w600, fontStyle: FontStyle.italic,
        color: const Color(0xFFD4956B),
      ),
    ),
    cardTheme: CardThemeData(
      color: surfaceDark,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: borderDark),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: false,
      enabledBorder: const UnderlineInputBorder(borderSide: BorderSide(color: borderDark, width: 1.5)),
      focusedBorder: const UnderlineInputBorder(borderSide: BorderSide(color: primary, width: 1.5)),
      hintStyle: GoogleFonts.poppins(fontSize: 14, color: mutedTextDark),
    ),
    dividerTheme: const DividerThemeData(color: borderDark, thickness: 1, space: 1),
  );
}

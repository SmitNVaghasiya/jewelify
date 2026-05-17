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
  final _detailsKey   = GlobalKey<FormState>();
  final _nameCtrl     = TextEditingController();
  final _usernameCtrl = TextEditingController();
  final _emailCtrl    = TextEditingController();
  final _passCtrl     = TextEditingController();
  final _otpCtrl      = TextEditingController();

  _Step _step         = _Step.details;
  bool _obscure       = true;
  bool _loading       = false;
  int  _resendSeconds = 0;

  @override
  void dispose() {
    _nameCtrl.dispose();
    _usernameCtrl.dispose();
    _emailCtrl.dispose();
    _passCtrl.dispose();
    _otpCtrl.dispose();
    super.dispose();
  }

  Future<void> _sendOtp() async {
    if (!_detailsKey.currentState!.validate()) return;
    setState(() => _loading = true);
    try {
      await context.read<AuthProvider>().sendRegistrationOtp(_emailCtrl.text.trim());
      if (!mounted) return;
      setState(() {
        _step = _Step.otp;
        _resendSeconds = 60;
      });
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
      setState(() {
        if (_resendSeconds > 0) _resendSeconds--;
      });
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
      final auth = context.read<AuthProvider>();
      await auth.verifyOtp(
        email: _emailCtrl.text.trim(),
        otp:   _otpCtrl.text.trim(),
      );
      await auth.register(
        name:     _nameCtrl.text.trim(),
        username: _usernameCtrl.text.trim(),
        email:    _emailCtrl.text.trim(),
        password: _passCtrl.text,
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
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppTheme.appNameBrown),
          onPressed: _step == _Step.otp
              ? () => setState(() {
                    _step = _Step.details;
                    _otpCtrl.clear();
                  })
              : () => Navigator.pop(context),
        ),
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
            validator: (v) =>
                (v == null || v.trim().isEmpty) ? 'Enter your name' : null,
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
                  _obscure
                      ? Icons.visibility_off_outlined
                      : Icons.visibility_outlined,
                  color: AppTheme.mutedText,
                  size: 20,
                ),
                onPressed: () => setState(() => _obscure = !_obscure),
              ),
            ),
            style: AppTheme.bodyMedium,
            validator: (v) {
              if (v == null || v.isEmpty) return 'Enter a password';
              if (v.length < 8) return 'At least 8 characters';
              if (!v.contains(RegExp(r'[A-Z]'))) return 'Need one uppercase letter';
              if (!v.contains(RegExp(r'\d'))) return 'Need one number';
              if (!v.contains(RegExp(r'[!@#$%^&*()\-=\[\]{};|,.<>/?@]'))) {
                return 'Need one special character';
              }
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
                  ? const SizedBox(
                      height: 18,
                      width: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Text('SEND OTP'),
            ),
          ),
          const SizedBox(height: 16),
          Center(
            child: TextButton(
              onPressed: () =>
                  Navigator.pushReplacementNamed(context, '/login'),
              style: TextButton.styleFrom(foregroundColor: AppTheme.primary),
              child: Text(
                'Already have an account? Sign in',
                style: AppTheme.bodySmall.copyWith(color: AppTheme.primary),
              ),
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
        Text(
          _emailCtrl.text.trim(),
          style: AppTheme.bodyMedium.copyWith(color: AppTheme.primary),
        ),
        const SizedBox(height: 36),
        TextFormField(
          controller: _otpCtrl,
          keyboardType: TextInputType.number,
          maxLength: 6,
          decoration:
              const InputDecoration(labelText: 'OTP CODE', counterText: ''),
          style: AppTheme.bodyMedium.copyWith(fontSize: 22, letterSpacing: 8),
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Text("Didn't receive it? ", style: AppTheme.bodySmall),
            TextButton(
              onPressed:
                  _resendSeconds == 0 && !_loading ? _resendOtp : null,
              style: TextButton.styleFrom(
                foregroundColor: AppTheme.primary,
                padding: EdgeInsets.zero,
                minimumSize: Size.zero,
              ),
              child: Text(
                _resendSeconds > 0
                    ? 'Resend in ${_resendSeconds}s'
                    : 'Resend now',
                style: AppTheme.bodySmall.copyWith(
                  color: _resendSeconds > 0
                      ? AppTheme.mutedText
                      : AppTheme.primary,
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
                ? const SizedBox(
                    height: 18,
                    width: 18,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white,
                    ),
                  )
                : const Text('VERIFY & CREATE ACCOUNT'),
          ),
        ),
        const SizedBox(height: 32),
      ],
    );
  }
}

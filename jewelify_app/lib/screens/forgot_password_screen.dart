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
  final _emailCtrl    = TextEditingController();
  final _otpCtrl      = TextEditingController();
  final _passCtrl     = TextEditingController();
  _FPStep _step       = _FPStep.email;
  bool _loading       = false;
  bool _obscure       = true;
  int  _resendSeconds = 0;

  @override
  void dispose() {
    _emailCtrl.dispose();
    _otpCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
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
      setState(() {
        _step = _FPStep.otp;
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

  void _verifyOtp() {
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
        email:       _emailCtrl.text.trim(),
        otp:         _otpCtrl.text.trim(),
        newPassword: _passCtrl.text,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Password reset successfully. Please sign in.'),
        ),
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
            if (_step == _FPStep.otp) {
              setState(() => _step = _FPStep.email);
            } else if (_step == _FPStep.newPass) {
              setState(() => _step = _FPStep.otp);
            } else {
              Navigator.pop(context);
            }
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
                Text(
                  'Enter the email linked to your account.',
                  style: AppTheme.bodySmall,
                ),
                const SizedBox(height: 24),
                TextFormField(
                  controller: _emailCtrl,
                  keyboardType: TextInputType.emailAddress,
                  decoration:
                      const InputDecoration(labelText: 'EMAIL ADDRESS'),
                  style: AppTheme.bodyMedium,
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
                        : const Text('SEND RESET CODE'),
                  ),
                ),
              ],
              if (_step == _FPStep.otp) ...[
                Text('Enter the 6-digit code sent to',
                    style: AppTheme.bodySmall),
                const SizedBox(height: 2),
                Text(
                  _emailCtrl.text.trim(),
                  style:
                      AppTheme.bodyMedium.copyWith(color: AppTheme.primary),
                ),
                const SizedBox(height: 24),
                TextFormField(
                  controller: _otpCtrl,
                  keyboardType: TextInputType.number,
                  maxLength: 6,
                  decoration: const InputDecoration(
                      labelText: 'RESET CODE', counterText: ''),
                  style: AppTheme.bodyMedium
                      .copyWith(fontSize: 22, letterSpacing: 8),
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Text("Didn't receive it? ", style: AppTheme.bodySmall),
                    TextButton(
                      onPressed: _resendSeconds == 0 ? _sendOtp : null,
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
                    onPressed: _loading ? null : _verifyOtp,
                    style: AppTheme.primaryButton,
                    child: const Text('CONTINUE'),
                  ),
                ),
              ],
              if (_step == _FPStep.newPass) ...[
                Text('Choose a new password for your account.',
                    style: AppTheme.bodySmall),
                const SizedBox(height: 24),
                TextFormField(
                  controller: _passCtrl,
                  obscureText: _obscure,
                  decoration: InputDecoration(
                    labelText: 'NEW PASSWORD',
                    suffixIcon: IconButton(
                      icon: Icon(
                        _obscure
                            ? Icons.visibility_off_outlined
                            : Icons.visibility_outlined,
                        color: AppTheme.mutedText,
                        size: 20,
                      ),
                      onPressed: () =>
                          setState(() => _obscure = !_obscure),
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
                        ? const SizedBox(
                            height: 18,
                            width: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
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

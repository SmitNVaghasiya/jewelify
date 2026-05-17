import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../constants/api.dart';

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
    final body = jsonDecode(res.body) as Map<String, dynamic>;
    final token = body['access_token'] as String;
    final userJson = body['user'] as Map<String, dynamic>? ?? body;
    await _saveUser(UserOut.fromJson(userJson), token);
  }

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
    final body = jsonDecode(res.body) as Map<String, dynamic>;
    final token = body['access_token'] as String;
    final userJson = body['user'] as Map<String, dynamic>? ?? body;
    await _saveUser(UserOut.fromJson(userJson), token);
  }

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

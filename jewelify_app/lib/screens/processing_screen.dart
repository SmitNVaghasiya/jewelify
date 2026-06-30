import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart' as http_parser;
import 'dart:io';
import 'package:provider/provider.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import '../providers/auth_provider.dart';
import '../screens/image_storage.dart';
import '../screens/results_screen.dart';
import '../constants/api.dart';
import '../screens/app_theme.dart';

class ProcessingScreen extends StatefulWidget {
  final Map<String, dynamic>? arguments;

  const ProcessingScreen({super.key, this.arguments});

  @override
  _ProcessingScreenState createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  bool _isError = false;
  String _errorMessage = "";
  bool _isCancelled = false;
  String _currentStatus = "Validating images locally...";
  bool _isLongWait = false;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();
    _processImages();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  Future<http.StreamedResponse> _sendRequestWithRetry(
    http.MultipartRequest request,
  ) async {
    const maxRetries = 2;
    var retries = 0;

    Future.delayed(const Duration(seconds: 8), () {
      if (mounted && !_isCancelled && !_isError && !_isLongWait) {
        setState(() {
          _isLongWait = true;
          _currentStatus = "Server is warming up, please wait a moment...";
        });
      }
    });

    while (retries < maxRetries) {
      try {
        final response = await request.send().timeout(
          const Duration(seconds: 60),
          onTimeout: () {
            throw Exception("Request timed out while initiating prediction");
          },
        );
        return response;
      } catch (e) {
        retries++;
        if (retries == maxRetries) {
          rethrow;
        }
        await Future.delayed(const Duration(seconds: 5));
      }
    }
    throw Exception("Failed to send request after $maxRetries attempts");
  }

  Future<void> _processImages() async {
    if (_isCancelled || widget.arguments == null) {
      if (mounted) Navigator.pop(context);
      return;
    }

    final faceFile = widget.arguments!['face'] as File;
    final jewelryFile = widget.arguments!['jewelry'] as File;
    final token = Provider.of<AuthProvider>(context, listen: false).token;

    if (token == null) {
      setState(() {
        _isError = true;
        _errorMessage = "Unauthorized: Please log in.";
      });
      Navigator.pushReplacementNamed(context, '/login');
      return;
    }

    setState(() => _currentStatus = "Validating images locally...");
    if (!await _validateImages(faceFile, jewelryFile)) {
      setState(() {
        _isError = true;
        _errorMessage =
            "Invalid images: Ensure a face is visible and jewelry is clear.";
      });
      return;
    }

    setState(() => _currentStatus = "Uploading images...");

    final apiUrl = '${ApiConstants.baseUrl}/predictions/predict';
    var request = http.MultipartRequest('POST', Uri.parse(apiUrl));

    try {
      request.headers['Authorization'] = 'Bearer $token';

      String getMimeType(String path) {
        final ext = path.split('.').last.toLowerCase();
        return ext == 'png' ? 'image/png' : 'image/jpeg';
      }

      final facePath = await ImageStorage.saveImage(
        faceFile,
        'face_${DateTime.now().millisecondsSinceEpoch}',
      );
      final jewelryPath = await ImageStorage.saveImage(
        jewelryFile,
        'jewelry_${DateTime.now().millisecondsSinceEpoch}',
      );

      if (facePath == null || jewelryPath == null) {
        throw Exception("Failed to save images locally");
      }

      request.files.add(
        await http.MultipartFile.fromPath(
          'face',
          faceFile.path,
          contentType: http_parser.MediaType.parse(getMimeType(faceFile.path)),
        ),
      );
      request.files.add(
        await http.MultipartFile.fromPath(
          'jewelry',
          jewelryFile.path,
          contentType: http_parser.MediaType.parse(
            getMimeType(jewelryFile.path),
          ),
        ),
      );
      request.fields['face_image_path'] = facePath;
      request.fields['jewelry_image_path'] = jewelryPath;

      setState(() => _currentStatus = "Initiating prediction tasks...");

      final response = await _sendRequestWithRetry(request);
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        try {
          final result = jsonDecode(responseBody);
          setState(() => _currentStatus = "Prediction tasks initiated!");
          if (mounted && !_isCancelled) {
            Navigator.of(context).pushReplacement(
              MaterialPageRoute(
                builder: (context) => ResultsScreen(initialResult: result),
              ),
            );
          }
        } catch (e) {
          setState(() {
            _isError = true;
            _errorMessage = "Failed to parse server response: $e";
          });
        }
      } else {
        setState(() {
          _isError = true;
          if (response.statusCode == 401) {
            _errorMessage =
                "Unauthorized: Invalid or expired token. Please log in again.";
            Navigator.pushReplacementNamed(context, '/login');
          } else if (response.statusCode == 404) {
            _errorMessage =
                "Server endpoint not found. Please check the backend configuration.";
          } else if (response.statusCode == 400 &&
              responseBody.contains("Validation failed")) {
            _errorMessage =
                "Validation failed: Ensure a face is visible and jewelry is clear.";
          } else if (response.statusCode == 500 &&
              responseBody.contains("Prediction failed")) {
            _errorMessage =
                "Prediction failed: Server error during prediction.";
          } else if (response.statusCode == 500) {
            _errorMessage =
                "Server error: An unexpected error occurred on the server.";
          } else {
            _errorMessage = "Error: ${response.statusCode} - $responseBody";
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isError = true;
          if (e.toString().contains("timed out")) {
            _errorMessage =
                "Request timed out. Please check your internet connection and try again.";
          } else if (e.toString().contains("Connection failed") ||
              e.toString().contains("Network is unreachable")) {
            _errorMessage =
                "Network error: Unable to connect to the server. Please check your internet connection.";
          } else {
            _errorMessage = "Failed to process: $e";
          }
        });
      }
    }
  }

  Future<bool> _validateImages(File faceFile, File jewelryFile) async {
    final inputImage = InputImage.fromFile(faceFile);
    final faceDetector = FaceDetector(options: FaceDetectorOptions());
    final faces = await faceDetector.processImage(inputImage);
    await faceDetector.close();

    if (faces.isEmpty) return false;

    final jewelryBytes = await jewelryFile.readAsBytes();
    if (jewelryBytes.length < 1024) return false;

    return true;
  }

  void _cancelProcessing() {
    setState(() {
      _isCancelled = true;
      _animationController.stop();
    });
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) {
        if (!didPop) _cancelProcessing();
      },
      child: Scaffold(
        appBar: AppBar(
          backgroundColor: AppTheme.background,
          foregroundColor: AppTheme.appNameBrown,
          elevation: 0,
          title: Text('Processing', style: AppTheme.titleStyle),
        ),
        body: SafeArea(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(32.0),
              child: _isError ? _buildErrorWidget() : _buildLoadingWidget(),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLoadingWidget() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        RotationTransition(
          turns: _animationController,
          child: Icon(Icons.diamond, size: 100, color: AppTheme.primary),
        ),
        const SizedBox(height: 48),
        Text(
          "Analyzing Your Style...",
          style: AppTheme.displayMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 16),
        Text(
          _currentStatus,
          style: AppTheme.bodyMedium.copyWith(color: AppTheme.mutedText),
          textAlign: TextAlign.center,
        ),
        if (_isLongWait) ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: AppTheme.softSurface,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: AppTheme.border),
            ),
            child: Text(
              "Render free tier may take up to 60 seconds on first request.",
              style: AppTheme.bodySmall.copyWith(color: AppTheme.mutedText),
              textAlign: TextAlign.center,
            ),
          ),
        ],
        const SizedBox(height: 32),
        OutlinedButton.icon(
          onPressed: _cancelProcessing,
          icon: const Icon(Icons.cancel),
          label: const Text("Cancel"),
          style: AppTheme.outlineButton,
        ),
      ],
    );
  }

  Widget _buildErrorWidget() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.error_outline, size: 100, color: Colors.red.shade700),
        const SizedBox(height: 48),
        Text(
          "Processing Error",
          style: AppTheme.displayMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 16),
        Text(
          _errorMessage,
          style: AppTheme.bodyMedium.copyWith(color: AppTheme.mutedText),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 32),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton.icon(
              onPressed: () {
                setState(() {
                  _isError = false;
                  _isLongWait = false;
                  _currentStatus = "Validating images locally...";
                  _animationController.repeat();
                });
                _processImages();
              },
              icon: const Icon(Icons.refresh),
              label: const Text("Retry"),
              style: AppTheme.primaryButton,
            ),
            const SizedBox(width: 16),
            OutlinedButton.icon(
              onPressed: _cancelProcessing,
              icon: const Icon(Icons.arrow_back),
              label: const Text("Go Back"),
              style: AppTheme.outlineButton,
            ),
          ],
        ),
      ],
    );
  }
}

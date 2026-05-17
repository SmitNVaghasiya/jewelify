import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../models/jewelry_recommendation.dart';
import '../widgets/prediction_module.dart';
import '../widgets/skeleton_loader.dart';
import '../constants/api.dart';
import '../screens/app_theme.dart';

class ResultsScreen extends StatefulWidget {
  final Map<String, dynamic>? initialResult;

  const ResultsScreen({super.key, this.initialResult});

  @override
  _ResultsScreenState createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  List<Map<String, dynamic>> _predictions = [];
  bool _isLoading = true;
  String _errorMessage = "";
  String? _predictionId;
  final Map<String, bool> _feedbackSubmitted = {
    'prediction1': false,
    'prediction2': false,
  };
  String _loadingMessage = "Processing your prediction...";
  bool _isLongPolling = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _fetchResult();
  }

  Future<void> _fetchResult() async {
    if (!mounted) return;

    setState(() {
      _isLoading = true;
      _errorMessage = "";
      _loadingMessage = "Processing your prediction...";
      _isLongPolling = false;
    });

    final authProvider = Provider.of<AuthProvider>(context, listen: false);
    final token = authProvider.token;

    if (token == null) {
      setState(() {
        _errorMessage = "Unauthorized: Please log in.";
        _isLoading = false;
      });
      Navigator.pushReplacementNamed(context, '/login');
      return;
    }

    final routeArgs =
        ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>?;
    final predictionData = widget.initialResult ?? routeArgs;
    final predictionId = predictionData?['prediction_id'] as String?;

    if (predictionId == null &&
        predictionData != null &&
        predictionData.containsKey('prediction1')) {
      setState(() {
        _predictionId = predictionData['prediction_id'];
        _predictions = [
          {
            'score':
                (predictionData['prediction1']['score'] is num
                    ? predictionData['prediction1']['score'].toDouble()
                    : 0.0),
            'category':
                predictionData['prediction1']['category']?.toString() ??
                'Not Assigned',
            'recommendations':
                (predictionData['prediction1']['recommendations']
                        as List<dynamic>?)
                    ?.map(
                      (rec) => JewelryRecommendation.fromJson(
                        rec as Map<String, dynamic>,
                      ),
                    )
                    .toList() ??
                [],
            'face_image_path': predictionData['face_image_path'],
            'jewelry_image_path': predictionData['jewelry_image_path'],
            'model': 'prediction1',
            'feedback_required':
                predictionData['prediction1']['feedback_required'] ?? false,
            'overall_feedback':
                double.tryParse(
                  predictionData['prediction1']['overall_feedback']
                          ?.toString() ??
                      '0.5',
                ) ??
                0.5,
          },
          {
            'score':
                (predictionData['prediction2']['score'] is num
                    ? predictionData['prediction2']['score'].toDouble()
                    : 0.0),
            'category':
                predictionData['prediction2']['category']?.toString() ??
                'Not Assigned',
            'recommendations':
                (predictionData['prediction2']['recommendations']
                        as List<dynamic>?)
                    ?.map(
                      (rec) => JewelryRecommendation.fromJson(
                        rec as Map<String, dynamic>,
                      ),
                    )
                    .toList() ??
                [],
            'face_image_path': predictionData['face_image_path'],
            'jewelry_image_path': predictionData['jewelry_image_path'],
            'model': 'prediction2',
            'feedback_required':
                predictionData['prediction2']['feedback_required'] ?? false,
            'overall_feedback':
                double.tryParse(
                  predictionData['prediction2']['overall_feedback']
                          ?.toString() ??
                      '0.5',
                ) ??
                0.5,
          },
        ];
        _isLoading = false;
      });

      if (_isFallbackPrediction()) {
        setState(() {
          _errorMessage = "Prediction failed on the server. Please try again.";
          _predictions = [];
        });
      }
      return;
    }

    if (predictionId == null) {
      setState(() {
        _isLoading = false;
        _errorMessage = "No prediction data provided";
      });
      return;
    }

    const maxAttempts = 30;
    const pollingInterval = Duration(seconds: 5);
    var attempts = 0;

    Future.delayed(const Duration(seconds: 60), () {
      if (mounted && _isLoading && !_isLongPolling) {
        setState(() {
          _isLongPolling = true;
          _loadingMessage = "Prediction is taking longer than expected...";
        });
      }
    });

    while (attempts < maxAttempts) {
      try {
        final apiUrl =
            '${ApiConstants.baseUrl}/predictions/get_prediction/$predictionId';
        final response = await http
            .get(Uri.parse(apiUrl), headers: {'Authorization': 'Bearer $token'})
            .timeout(
              const Duration(seconds: 30),
              onTimeout: () {
                throw Exception(
                  "Request timed out while fetching prediction results",
                );
              },
            );

        if (response.statusCode == 200) {
          final dynamic data = jsonDecode(response.body);
          if (data['validation_status'] == 'pending' ||
              data['prediction_status'] == 'pending') {
            attempts++;
            await Future.delayed(pollingInterval);
            continue;
          } else if (data['validation_status'] == 'failed') {
            setState(() {
              _isLoading = false;
              _errorMessage =
                  "Validation failed: Ensure a face is visible and jewelry is clear.";
            });
            return;
          } else if (data['prediction_status'] == 'failed') {
            setState(() {
              _isLoading = false;
              _errorMessage =
                  "Prediction failed: Server error during prediction.";
            });
            return;
          } else {
            setState(() {
              _predictionId = predictionId;
              _predictions = [
                {
                  'score':
                      (data['prediction1']['score'] is num
                          ? data['prediction1']['score'].toDouble()
                          : 0.0),
                  'category':
                      data['prediction1']['category']?.toString() ??
                      'Not Assigned',
                  'recommendations':
                      (data['prediction1']['recommendations'] as List<dynamic>?)
                          ?.map(
                            (rec) => JewelryRecommendation.fromJson(
                              rec as Map<String, dynamic>,
                            ),
                          )
                          .toList() ??
                      [],
                  'face_image_path': data['face_image_path'],
                  'jewelry_image_path': data['jewelry_image_path'],
                  'model': 'prediction1',
                  'feedback_required':
                      data['prediction1']['feedback_required'] ?? false,
                  'overall_feedback':
                      double.tryParse(
                        data['prediction1']['overall_feedback']?.toString() ??
                            '0.5',
                      ) ??
                      0.5,
                },
                {
                  'score':
                      (data['prediction2']['score'] is num
                          ? data['prediction2']['score'].toDouble()
                          : 0.0),
                  'category':
                      data['prediction2']['category']?.toString() ??
                      'Not Assigned',
                  'recommendations':
                      (data['prediction2']['recommendations'] as List<dynamic>?)
                          ?.map(
                            (rec) => JewelryRecommendation.fromJson(
                              rec as Map<String, dynamic>,
                            ),
                          )
                          .toList() ??
                      [],
                  'face_image_path': data['face_image_path'],
                  'jewelry_image_path': data['jewelry_image_path'],
                  'model': 'prediction2',
                  'feedback_required':
                      data['prediction2']['feedback_required'] ?? false,
                  'overall_feedback':
                      double.tryParse(
                        data['prediction2']['overall_feedback']?.toString() ??
                            '0.5',
                      ) ??
                      0.5,
                },
              ];
              _isLoading = false;
            });

            if (_isFallbackPrediction()) {
              setState(() {
                _errorMessage =
                    "Prediction failed on the server. Please try again.";
                _predictions = [];
              });
            }
            return;
          }
        } else {
          final errorDetail =
              jsonDecode(response.body)['detail'] ?? 'Unknown error';
          setState(() {
            _isLoading = false;
            if (response.statusCode == 401) {
              _errorMessage =
                  "Unauthorized: Invalid or expired token. Please log in again.";
              Navigator.pushReplacementNamed(context, '/login');
            } else if (response.statusCode == 400 &&
                errorDetail == "Failed validation") {
              _errorMessage =
                  "Validation failed: Ensure a face is visible and jewelry is clear.";
            } else if (response.statusCode == 500 &&
                errorDetail == "Failed prediction") {
              _errorMessage =
                  "Prediction failed: Server error during prediction.";
            } else if (response.statusCode == 408) {
              _errorMessage =
                  "Request timed out waiting for prediction to complete.";
            } else {
              _errorMessage = "Failed to fetch result: $errorDetail";
            }
          });
          return;
        }
      } catch (e) {
        setState(() {
          _isLoading = false;
          if (e.toString().contains("timed out")) {
            _errorMessage =
                "Request timed out. Please check your connection and try again.";
          } else if (e.toString().contains("Connection failed") ||
              e.toString().contains("Network is unreachable")) {
            _errorMessage =
                "Network error: Unable to connect to the server. Please check your internet connection.";
          } else {
            _errorMessage = "Error fetching result: $e";
          }
        });
        return;
      }
    }

    setState(() {
      _isLoading = false;
      _errorMessage = "Prediction took too long to complete. Please try again.";
    });
  }

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

  Future<void> _submitFeedback(
    String predictionId,
    String model, {
    String? recommendationName,
    String review = "yes",
  }) async {
    final authProvider = Provider.of<AuthProvider>(context, listen: false);
    final token = authProvider.token;

    if (token == null) {
      setState(() {
        _errorMessage = "Unauthorized: Please log in.";
      });
      Navigator.pushReplacementNamed(context, '/login');
      return;
    }

    final feedbackType =
        recommendationName == null ? "prediction" : "recommendation";
    final apiUrl =
        '${ApiConstants.baseUrl}/predictions/feedback/$feedbackType';

    try {
      var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
      request.headers['Authorization'] = 'Bearer $token';
      request.fields['prediction_id'] = predictionId;
      request.fields['model_type'] = model;
      if (recommendationName != null) {
        request.fields['recommendation_name'] = recommendationName;
      }
      request.fields['score'] = review;

      final response = await request.send().timeout(
        const Duration(seconds: 30),
        onTimeout: () {
          throw Exception("Request timed out while submitting feedback");
        },
      );
      await response.stream.bytesToString();

      if (!mounted) return;
      if (response.statusCode == 200) {
        if (recommendationName == null) {
          setState(() {
            _feedbackSubmitted[model] = true;
          });
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: const Text("Feedback submitted successfully!"),
              backgroundColor: AppTheme.primary,
            ),
          );
        }
      } else if (response.statusCode == 401) {
        setState(() {
          _errorMessage =
              "Unauthorized: Invalid or expired token. Please log in again.";
        });
        Navigator.pushReplacementNamed(context, '/login');
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Failed to submit feedback. Please try again."),
          ),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Error submitting feedback. Please try again."),
        ),
      );
    }
  }

  bool _canProceed() {
    return _feedbackSubmitted['prediction1'] == true &&
        _feedbackSubmitted['prediction2'] == true;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        foregroundColor: AppTheme.appNameBrown,
        elevation: 0,
        title: Text('Your Result', style: AppTheme.titleStyle),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child:
              _isLoading
                  ? _buildSkeletonWidget()
                  : _errorMessage.isNotEmpty
                  ? _buildErrorWidget()
                  : _buildResultWidget(),
        ),
      ),
      bottomNavigationBar:
          _predictions.isNotEmpty && !_canProceed()
              ? Container(
                padding: const EdgeInsets.all(16.0),
                color: AppTheme.softSurface,
                child: Text(
                  "Please provide feedback for both models to proceed.",
                  style: AppTheme.bodySmall.copyWith(color: AppTheme.mutedText),
                  textAlign: TextAlign.center,
                ),
              )
              : null,
    );
  }

  Widget _buildSkeletonWidget() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          _loadingMessage,
          style: AppTheme.bodyMedium.copyWith(color: AppTheme.mutedText),
        ),
        const SizedBox(height: 16),
        const SkeletonHistoryItem(),
        const SizedBox(height: 12),
        const SkeletonHistoryItem(),
        const SizedBox(height: 12),
        const SkeletonHistoryItem(),
      ],
    );
  }

  Widget _buildResultWidget() {
    if (_predictions.isEmpty) {
      return Center(
        child: Text(
          "No prediction data available",
          style: AppTheme.bodyMedium.copyWith(color: AppTheme.mutedText),
        ),
      );
    }

    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            itemCount: _predictions.length,
            itemBuilder: (context, index) {
              final prediction = _predictions[index];
              if (prediction['score'] == null ||
                  prediction['category'] == null ||
                  prediction['recommendations'] == null ||
                  prediction['face_image_path'] == null ||
                  prediction['jewelry_image_path'] == null ||
                  prediction['model'] == null ||
                  prediction['feedback_required'] == null ||
                  prediction['overall_feedback'] == null) {
                return const SizedBox.shrink();
              }

              return PredictionModule(
                modelName:
                    index == 0 ? "Model 1 (XGBoost)" : "Model 2 (MLP)",
                prediction: prediction,
                predictionId: _predictionId ?? '',
                onFeedbackSubmit: _submitFeedback,
              );
            },
          ),
        ),
        const SizedBox(height: 16),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed:
                    _canProceed()
                        ? () =>
                            Navigator.pushReplacementNamed(context, '/upload')
                        : null,
                icon: const Icon(Icons.refresh),
                label: const Text("Try Again"),
                style: AppTheme.outlineButton,
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: ElevatedButton.icon(
                onPressed:
                    _canProceed()
                        ? () => Navigator.popUntil(
                          context,
                          (route) => route.isFirst,
                        )
                        : null,
                icon: const Icon(Icons.home),
                label: const Text("Home"),
                style: AppTheme.primaryButton,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildErrorWidget() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.error_outline, size: 80, color: Colors.red.shade700),
        const SizedBox(height: 24),
        Text(
          "Error Fetching Result",
          style: AppTheme.displayMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 16),
        Text(
          _errorMessage,
          style: AppTheme.bodyMedium.copyWith(color: AppTheme.mutedText),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 24),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: _fetchResult,
              style: AppTheme.primaryButton,
              child: const Text("Retry"),
            ),
            const SizedBox(width: 16),
            OutlinedButton(
              onPressed:
                  () => Navigator.pushReplacementNamed(context, '/upload'),
              style: AppTheme.outlineButton,
              child: const Text("Go Back"),
            ),
          ],
        ),
      ],
    );
  }
}

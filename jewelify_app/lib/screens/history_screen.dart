import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../models/jewelry_recommendation.dart';
import '../widgets/expandable_item.dart';
import '../widgets/prediction_module.dart';
import '../constants/api.dart';
import '../screens/app_theme.dart';
import '../widgets/skeleton_loader.dart';

class HistoryItem {
  final String id;
  final String? userId;
  final String date;
  final Map<String, dynamic> prediction1;
  final Map<String, dynamic> prediction2;
  final String? faceImagePath;
  final String? jewelryImagePath;
  final String validationStatus;
  final String predictionStatus;
  bool isExpanded;

  HistoryItem({
    required this.id,
    this.userId,
    required this.date,
    required this.prediction1,
    required this.prediction2,
    this.faceImagePath,
    this.jewelryImagePath,
    required this.validationStatus,
    required this.predictionStatus,
    this.isExpanded = false,
  });

  factory HistoryItem.fromJson(Map<String, dynamic> json) {
    return HistoryItem(
      id: json['id']?.toString() ?? '',
      userId: json['user_id']?.toString(),
      date: json['timestamp']?.toString() ?? 'Unknown Date',
      prediction1: {
        'score':
            (json['prediction1']['score'] is num
                ? json['prediction1']['score'].toDouble()
                : 0.0),
        'category':
            json['prediction1']['category']?.toString() ?? 'Not Assigned',
        'recommendations':
            (json['prediction1']['recommendations'] as List<dynamic>?)
                ?.map(
                  (rec) => JewelryRecommendation.fromJson(
                    rec as Map<String, dynamic>,
                  ),
                )
                .toList() ??
            [],
        'overall_feedback':
            json['prediction1']['overall_feedback']?.toString() ??
            'Not Provided',
        'feedback_required':
            json['prediction1']['feedback_required']?.toString(),
        'face_image_path': json['face_image_path']?.toString(),
        'jewelry_image_path': json['jewelry_image_path']?.toString(),
        'model': 'prediction1',
      },
      prediction2: {
        'score':
            (json['prediction2']['score'] is num
                ? json['prediction2']['score'].toDouble()
                : 0.0),
        'category':
            json['prediction2']['category']?.toString() ?? 'Not Assigned',
        'recommendations':
            (json['prediction2']['recommendations'] as List<dynamic>?)
                ?.map(
                  (rec) => JewelryRecommendation.fromJson(
                    rec as Map<String, dynamic>,
                  ),
                )
                .toList() ??
            [],
        'overall_feedback':
            json['prediction2']['overall_feedback']?.toString() ??
            'Not Provided',
        'feedback_required':
            json['prediction2']['feedback_required']?.toString(),
        'face_image_path': json['face_image_path']?.toString(),
        'jewelry_image_path': json['jewelry_image_path']?.toString(),
        'model': 'prediction2',
      },
      faceImagePath: json['face_image_path']?.toString(),
      jewelryImagePath: json['jewelry_image_path']?.toString(),
      validationStatus: json['validation_status']?.toString() ?? 'unknown',
      predictionStatus: json['prediction_status']?.toString() ?? 'unknown',
    );
  }
}

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  _HistoryScreenState createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final List<HistoryItem> _history = [];
  bool _isLoading = true;
  String? _errorMessage;
  final int _limit = 10;
  int _skip = 0;
  bool _hasMore = true;
  final Map<String, Future<File?>> _imageFutures = {};

  Future<File?> _getImageFuture(String? path) {
    if (path == null || path.isEmpty) return Future.value(null);
    if (_imageFutures.containsKey(path)) return _imageFutures[path]!;
    Future<File?> f;
    if (path.startsWith('http')) {
      f = Future.value(null);
    } else {
      f = (() async {
        final file = File(path);
        return await file.exists() ? file : null;
      })();
    }
    _imageFutures[path] = f;
    return f;
  }

  Widget _buildThumb(String? path) {
    return FutureBuilder<File?>(
      future: _getImageFuture(path),
      builder: (ctx, snap) {
        if (snap.connectionState == ConnectionState.waiting) {
          return Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [Color(0xFFF5EDE4), Color(0xFFEDE0D4)],
              ),
              borderRadius: BorderRadius.circular(7),
            ),
          );
        }
        if (snap.hasData && snap.data != null) {
          return ClipRRect(
            borderRadius: BorderRadius.circular(7),
            child: Image.file(
              snap.data!,
              width: 42,
              height: 42,
              fit: BoxFit.cover,
            ),
          );
        }
        return Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [Color(0xFFF5EDE4), Color(0xFFEDE0D4)],
            ),
            borderRadius: BorderRadius.circular(7),
            border: Border.all(color: AppTheme.border),
          ),
          child: Icon(
            Icons.image_outlined,
            size: 18,
            color: AppTheme.mutedText,
          ),
        );
      },
    );
  }

  Widget _scoreBar(String label, double score) {
    return Row(
      children: [
        SizedBox(
          width: 18,
          child: Text(
            label,
            style: const TextStyle(
              fontSize: 8,
              fontWeight: FontWeight.w700,
              letterSpacing: 1,
              color: AppTheme.mutedText,
            ),
          ),
        ),
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(2),
            child: LinearProgressIndicator(
              value: score / 100,
              minHeight: 3,
              backgroundColor: AppTheme.border,
              valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.primary),
            ),
          ),
        ),
        const SizedBox(width: 4),
        SizedBox(
          width: 28,
          child: Text(
            '${score.toStringAsFixed(0)}%',
            textAlign: TextAlign.right,
            style: const TextStyle(
              fontSize: 9,
              fontWeight: FontWeight.w700,
              color: AppTheme.primary,
            ),
          ),
        ),
      ],
    );
  }

  @override
  void initState() {
    super.initState();
    _fetchHistory();
  }

  Future<void> _fetchHistory({bool loadMore = false}) async {
    if (loadMore) {
      setState(() {
        _skip += _limit;
      });
    } else {
      setState(() {
        _isLoading = true;
        _errorMessage = null;
        _skip = 0;
        _history.clear();
        _hasMore = true;
      });
    }

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

    try {
      final response = await http
          .get(
            Uri.parse(
              '${ApiConstants.baseUrl}/history/?limit=$_limit&skip=$_skip',
            ),
            headers: {'Authorization': 'Bearer $token'},
          )
          .timeout(
            const Duration(seconds: 100),
            onTimeout: () {
              throw Exception("Request timed out while fetching history");
            },
          );

      if (response.statusCode == 200) {
        final dynamic data = json.decode(response.body);
        if (data is List) {
          final newItems =
              data
                  .map(
                    (item) =>
                        HistoryItem.fromJson(item as Map<String, dynamic>),
                  )
                  .toList();
          setState(() {
            _history.addAll(newItems);
            _isLoading = false;
            _hasMore = newItems.length == _limit;
          });
        } else if (data is Map && data.containsKey('message')) {
          setState(() {
            _isLoading = false;
            _hasMore = false;
          });
        }
      } else if (response.statusCode == 401) {
        setState(() {
          _errorMessage = "Session expired. Please log in again.";
          _isLoading = false;
        });
        authProvider.logout();
        if (!mounted) return;
        Navigator.pushReplacementNamed(context, '/login');
      } else {
        setState(() {
          _errorMessage = "Failed to fetch history: ${response.statusCode}";
          _isLoading = false;
          _hasMore = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage =
            e.toString().contains("timed out")
                ? "Request timed out. Please check your connection and try again."
                : "Error fetching history: $e";
        _isLoading = false;
        _hasMore = false;
      });
    }
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
        const Duration(seconds: 100),
        onTimeout: () {
          throw Exception("Request timed out while submitting feedback");
        },
      );
      await response.stream.bytesToString();

      if (!mounted) return;
      if (response.statusCode != 200) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Failed to submit feedback. Please try again."),
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Feedback submitted successfully!")),
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: AppTheme.background,
        foregroundColor: AppTheme.appNameBrown,
        elevation: 0,
        title: Text('History', style: AppTheme.titleStyle),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: Stack(
        children: [
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'YOUR PREDICTION HISTORY',
                    style: AppTheme.labelUppercase,
                  ),
                  const SizedBox(height: 16),
                  Expanded(
                    child:
                        _isLoading && _history.isEmpty
                            ? ListView(
                              children: const [
                                SkeletonHistoryItem(),
                                SkeletonHistoryItem(),
                                SkeletonHistoryItem(),
                                SkeletonHistoryItem(),
                                SkeletonHistoryItem(),
                              ],
                            )
                            : _errorMessage != null
                            ? Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Text(
                                    _errorMessage!,
                                    style: const TextStyle(color: Colors.red),
                                    textAlign: TextAlign.center,
                                  ),
                                  const SizedBox(height: 20),
                                  ElevatedButton(
                                    onPressed: () => _fetchHistory(),
                                    style: AppTheme.primaryButton,
                                    child: const Text('Retry'),
                                  ),
                                ],
                              ),
                            )
                            : _history.isEmpty
                            ? Center(
                              child: Text(
                                'No history exists',
                                style: AppTheme.bodyMedium.copyWith(
                                  color: AppTheme.mutedText,
                                ),
                              ),
                            )
                            : RefreshIndicator(
                              color: AppTheme.primary,
                              onRefresh: () => _fetchHistory(),
                              child: ListView.builder(
                                cacheExtent: 1000,
                                itemCount: _history.length + (_hasMore ? 1 : 0),
                                itemBuilder: (context, index) {
                                  if (index == _history.length && _hasMore) {
                                    _fetchHistory(loadMore: true);
                                    return const Center(
                                      child: Padding(
                                        padding: EdgeInsets.all(16.0),
                                        child: CircularProgressIndicator(),
                                      ),
                                    );
                                  }

                                  final item = _history[index];
                                  return Container(
                                    margin: const EdgeInsets.only(bottom: 10),
                                    decoration: AppTheme.cardDecoration,
                                    child: ExpandableItem(
                                      header: Padding(
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 14,
                                          vertical: 12,
                                        ),
                                        child: Row(
                                          crossAxisAlignment:
                                              CrossAxisAlignment.center,
                                          children: [
                                            // Image thumbnails
                                            Row(
                                              children: [
                                                _buildThumb(item.faceImagePath),
                                                const SizedBox(width: 4),
                                                _buildThumb(
                                                  item.jewelryImagePath,
                                                ),
                                              ],
                                            ),
                                            const SizedBox(width: 10),
                                            // Date + score bars
                                            Expanded(
                                              child: Column(
                                                crossAxisAlignment:
                                                    CrossAxisAlignment.start,
                                                children: [
                                                  Text(
                                                    item.date,
                                                    style: const TextStyle(
                                                      fontSize: 11,
                                                      fontWeight: FontWeight.w600,
                                                      color: AppTheme.headingBrown,
                                                    ),
                                                  ),
                                                  const SizedBox(height: 6),
                                                  _scoreBar(
                                                    'M1',
                                                    (item.prediction1['score']
                                                            as num)
                                                        .toDouble(),
                                                  ),
                                                  const SizedBox(height: 3),
                                                  _scoreBar(
                                                    'M2',
                                                    (item.prediction2['score']
                                                            as num)
                                                        .toDouble(),
                                                  ),
                                                ],
                                              ),
                                            ),
                                            const SizedBox(width: 8),
                                            Icon(
                                              item.isExpanded
                                                  ? Icons.keyboard_arrow_up
                                                  : Icons.keyboard_arrow_down,
                                              color: AppTheme.mutedText,
                                              size: 18,
                                            ),
                                          ],
                                        ),
                                      ),
                                      content: Column(
                                        children: [
                                          PredictionModule(
                                            modelName: "Model 1 (XGBoost)",
                                            prediction: item.prediction1,
                                            predictionId: item.id,
                                            onFeedbackSubmit: _submitFeedback,
                                          ),
                                          const SizedBox(height: 16),
                                          PredictionModule(
                                            modelName: "Model 2 (MLP)",
                                            prediction: item.prediction2,
                                            predictionId: item.id,
                                            onFeedbackSubmit: _submitFeedback,
                                          ),
                                        ],
                                      ),
                                      isExpanded: item.isExpanded,
                                      onToggle:
                                          (expanded) => setState(
                                            () => item.isExpanded = expanded,
                                          ),
                                    ),
                                  );
                                },
                              ),
                            ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

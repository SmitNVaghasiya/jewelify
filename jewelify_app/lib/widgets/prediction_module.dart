import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/jewelry_recommendation.dart';
import '../screens/app_theme.dart';
import 'recommendation_card.dart';

class PredictionModule extends StatefulWidget {
  final String modelName;
  final Map<String, dynamic> prediction;
  final String predictionId;
  final Future<void> Function(
    String,
    String, {
    String? recommendationName,
    String review,
  })
  onFeedbackSubmit;

  const PredictionModule({
    super.key,
    required this.modelName,
    required this.prediction,
    required this.predictionId,
    required this.onFeedbackSubmit,
  });

  @override
  _PredictionModuleState createState() => _PredictionModuleState();
}

class _PredictionModuleState extends State<PredictionModule>
    with AutomaticKeepAliveClientMixin {
  late Future<File?> _faceImageFuture;
  late Future<File?> _jewelryImageFuture;
  int _feedbackStars = 0;
  bool _hasSubmittedFeedback = false;
  bool _isSubmittingFeedback = false;

  @override
  void initState() {
    super.initState();
    _faceImageFuture = _loadFile(widget.prediction['face_image_path']);
    _jewelryImageFuture = _loadFile(widget.prediction['jewelry_image_path']);

    // Only mark as submitted when backend explicitly cleared feedback_required.
    // Do NOT use overall_feedback value — it defaults to 0.5 on new records.
    final fr = widget.prediction['feedback_required'];
    final alreadySubmitted = fr == false || fr == 'false' || fr == 'False';
    if (alreadySubmitted) {
      _hasSubmittedFeedback = true;
      final rawFeedback = widget.prediction['overall_feedback'];
      if (rawFeedback != null && rawFeedback.toString() != 'Not Provided') {
        final f = double.tryParse(rawFeedback.toString()) ?? 0.0;
        _feedbackStars = (f * 5).round().clamp(1, 5);
      }
    }
  }

  Future<File?> _loadFile(dynamic path) async {
    if (path == null || path.toString().isEmpty) return null;
    final p = path.toString();
    if (p.startsWith('http')) return null;
    final file = File(p);
    return await file.exists() ? file : null;
  }

  Future<void> _submitStarFeedback(int stars) async {
    if (_hasSubmittedFeedback || _isSubmittingFeedback) return;
    setState(() {
      _feedbackStars = stars;
      _isSubmittingFeedback = true;
    });
    final review = (stars / 5.0).toStringAsFixed(2);
    await widget.onFeedbackSubmit(
      widget.predictionId,
      widget.prediction['model']?.toString() ?? '',
      review: review,
    );
    if (mounted) {
      setState(() {
        _hasSubmittedFeedback = true;
        _isSubmittingFeedback = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final score = (widget.prediction['score'] as num? ?? 0.0).toDouble();
    final category =
        widget.prediction['category']?.toString() ?? 'Not Assigned';
    final recommendations =
        widget.prediction['recommendations'] as List<JewelryRecommendation>? ??
        [];
    final fr = widget.prediction['feedback_required'];
    final feedbackRequired = !(fr == false || fr == 'false' || fr == 'False');

    return Container(
      decoration: BoxDecoration(
        border: Border(top: BorderSide(color: AppTheme.border)),
        color: AppTheme.background,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Model label header
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 10, 14, 0),
            child: Text(
              widget.modelName.toUpperCase(),
              style: const TextStyle(
                fontSize: 9,
                fontWeight: FontWeight.w700,
                letterSpacing: 2.5,
                color: AppTheme.mutedText,
              ),
            ),
          ),

          // Photos section
          _divider(),
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 10, 14, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _sectionLabel('YOUR PHOTOS'),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: _imageBox(_faceImageFuture, 'Face', Icons.face_outlined),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: _imageBox(_jewelryImageFuture, 'Jewelry', Icons.diamond_outlined),
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Score section
          _divider(),
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 10, 14, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _sectionLabel('COMPATIBILITY'),
                const SizedBox(height: 10),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    SizedBox(
                      width: 82,
                      height: 82,
                      child: CustomPaint(
                        painter: _ScoreArcPainter(score),
                      ),
                    ),
                    const SizedBox(width: 14),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          score.toStringAsFixed(0),
                          style: GoogleFonts.cormorantGaramond(
                            fontSize: 44,
                            fontWeight: FontWeight.w600,
                            fontStyle: FontStyle.italic,
                            color: AppTheme.primary,
                            height: 1,
                          ),
                        ),
                        Text(
                          '/ 100 %',
                          style: const TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                            color: AppTheme.mutedText,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 10,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: AppTheme.primary,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            category.toUpperCase(),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 8,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 1.5,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Recommendations section
          if (recommendations.isNotEmpty) ...[
            _divider(),
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 10, 14, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionLabel('RECOMMENDATIONS'),
                  const SizedBox(height: 8),
                  ...recommendations.map(
                    (rec) => Padding(
                      padding: const EdgeInsets.only(bottom: 7),
                      child: RecommendationCard(recommendation: rec),
                    ),
                  ),
                ],
              ),
            ),
          ],

          // Feedback section
          if (feedbackRequired) ...[
            _divider(),
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 10, 14, 14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionLabel('RATE THIS RESULT'),
                  const SizedBox(height: 8),
                  Row(
                    children: List.generate(5, (i) {
                      return GestureDetector(
                        onTap: () => _submitStarFeedback(i + 1),
                        child: Padding(
                          padding: const EdgeInsets.only(right: 6),
                          child: Icon(
                            i < _feedbackStars
                                ? Icons.star_rounded
                                : Icons.star_outline_rounded,
                            size: 28,
                            color:
                                i < _feedbackStars
                                    ? AppTheme.primary
                                    : AppTheme.border,
                          ),
                        ),
                      );
                    }),
                  ),
                  if (_isSubmittingFeedback)
                    const Padding(
                      padding: EdgeInsets.only(top: 8),
                      child: SizedBox(
                        height: 14,
                        width: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    ),
                  if (_hasSubmittedFeedback && !_isSubmittingFeedback)
                    Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Text(
                        'Feedback submitted',
                        style: TextStyle(
                          fontSize: 11,
                          color: AppTheme.mutedText,
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _divider() => const Divider(
    height: 1,
    thickness: 1,
    color: AppTheme.border,
  );

  Widget _sectionLabel(String text) => Text(
    text,
    style: const TextStyle(
      fontSize: 8,
      fontWeight: FontWeight.w700,
      letterSpacing: 2.5,
      color: AppTheme.mutedText,
    ),
  );

  static const _imgBoxDecoration = BoxDecoration(
    gradient: LinearGradient(
      begin: Alignment.topLeft,
      end: Alignment.bottomRight,
      colors: [Color(0xFFF5EDE4), Color(0xFFEDE0D4)],
    ),
    borderRadius: BorderRadius.all(Radius.circular(9)),
    border: Border.fromBorderSide(BorderSide(color: Color(0xFFEDE0D4))),
  );

  Widget _imageBox(Future<File?> future, String label, IconData icon) {
    return FutureBuilder<File?>(
      future: future,
      builder: (ctx, snap) {
        if (snap.connectionState == ConnectionState.waiting) {
          return Container(
            height: 80,
            decoration: _imgBoxDecoration,
            child: const Center(
              child: SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            ),
          );
        }
        if (snap.hasData && snap.data != null) {
          return ClipRRect(
            borderRadius: BorderRadius.circular(9),
            child: Image.file(
              snap.data!,
              width: double.infinity,
              height: 80,
              fit: BoxFit.cover,
            ),
          );
        }
        return Container(
          height: 80,
          decoration: _imgBoxDecoration,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 22, color: AppTheme.mutedText),
              const SizedBox(height: 3),
              Text(
                label.toUpperCase(),
                style: const TextStyle(
                  fontSize: 7,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.5,
                  color: AppTheme.mutedText,
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  bool get wantKeepAlive => true;
}

class _ScoreArcPainter extends CustomPainter {
  final double score;
  const _ScoreArcPainter(this.score);

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2 - 5;

    final trackPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 7
      ..color = const Color(0xFFEDE0D4);
    canvas.drawCircle(center, radius, trackPaint);

    final arcPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 7
      ..strokeCap = StrokeCap.round
      ..color = const Color(0xFFB5451B);
    final sweepAngle = 2 * math.pi * (score.clamp(0, 100) / 100);
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -math.pi / 2,
      sweepAngle,
      false,
      arcPaint,
    );
  }

  @override
  bool shouldRepaint(_ScoreArcPainter old) => old.score != score;
}

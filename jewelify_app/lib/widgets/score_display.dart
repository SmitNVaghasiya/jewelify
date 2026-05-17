// import 'package:flutter/material.dart';

// class ScoreDisplay extends StatelessWidget {
//   final double score;
//   final String category;

//   const ScoreDisplay({super.key, required this.score, required this.category});

//   String addEmojiToCategory(String category) {
//     switch (category.trim().toLowerCase()) {
//       case 'very good':
//         return '⭐ Very Good';
//       case 'good':
//         return '✅ Good';
//       case 'neutral':
//         return '😐 Neutral';
//       case 'bad':
//         return '⚠️ Bad';
//       case 'very bad':
//         return '❌ Very Bad';
//       default:
//         return category;
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     final theme = Theme.of(context);
//     return Column(
//       crossAxisAlignment: CrossAxisAlignment.start,
//       children: [
//         Text(
//           "Score: ${score.toStringAsFixed(2)}%",
//           style: theme.textTheme.bodyLarge,
//         ),
//         const SizedBox(height: 12),
//         Text(
//           "Category: ${addEmojiToCategory(category)}",
//           style: theme.textTheme.bodyLarge?.copyWith(
//             fontFamily: 'NotoColorEmoji',
//           ),
//         ),
//       ],
//     );
//   }
// }

import 'package:flutter/material.dart';

class ScoreDisplay extends StatelessWidget {
  final double score;
  final String category;

  const ScoreDisplay({super.key, required this.score, required this.category});

  String _getRating(double score) {
    if (score >= 80) return 'Very Good';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Average';
    if (score >= 20) return 'Poor';
    return 'Very Poor';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final rating = _getRating(score);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Flexible(
              child: Text(
                "Compatibility Score: ${score.toStringAsFixed(2)}%",
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '⭐ $rating',
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
                color: Colors.amber,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Text("Category: $category", style: theme.textTheme.titleMedium),
      ],
    );
  }
}

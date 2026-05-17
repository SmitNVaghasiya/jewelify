// import 'package:flutter/material.dart';
// import 'package:cached_network_image/cached_network_image.dart';
// import '../models/jewelry_recommendation.dart';

// class RecommendationCard extends StatelessWidget {
//   final JewelryRecommendation recommendation;
//   final Function(String?) onImageTap;

//   const RecommendationCard({
//     super.key,
//     required this.recommendation,
//     required this.onImageTap,
//   });

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
//     return Padding(
//       padding: const EdgeInsets.symmetric(vertical: 8.0),
//       child: Column(
//         crossAxisAlignment: CrossAxisAlignment.start,
//         children: [
//           Text(
//             recommendation.name,
//             style: theme.textTheme.bodyLarge?.copyWith(
//               fontWeight: FontWeight.w600,
//             ),
//           ),
//           const SizedBox(height: 8),
//           GestureDetector(
//             onTap: () => onImageTap(recommendation.displayUrl),
//             child: ClipRRect(
//               borderRadius: BorderRadius.circular(12),
//               child: CachedNetworkImage(
//                 imageUrl: recommendation.displayUrl,
//                 width: double.infinity,
//                 height: 150,
//                 fit: BoxFit.cover,
//                 placeholder:
//                     (context, url) =>
//                         const Center(child: CircularProgressIndicator()),
//                 errorWidget:
//                     (context, url, error) => Container(
//                       width: double.infinity,
//                       height: 150,
//                       color: Colors.grey[300],
//                       child: const Icon(Icons.error),
//                     ),
//               ),
//             ),
//           ),
//           const SizedBox(height: 8),
//           Text(
//             "Compatibility Score: ${(recommendation.score).toStringAsFixed(1)}%",
//             style: theme.textTheme.bodyLarge,
//           ),
//           const SizedBox(height: 4),
//           Text(
//             "Category: ${addEmojiToCategory(recommendation.category)}",
//             style: theme.textTheme.bodyLarge?.copyWith(
//               fontFamily: 'NotoColorEmoji',
//             ),
//           ),
//         ],
//       ),
//     );
//   }
// }

import 'package:flutter/material.dart';
import '../models/jewelry_recommendation.dart';
import '../screens/app_theme.dart';

class RecommendationCard extends StatelessWidget {
  final JewelryRecommendation recommendation;
  final void Function(String url)? onImageTap;

  const RecommendationCard({
    super.key,
    required this.recommendation,
    this.onImageTap,
  });

  @override
  Widget build(BuildContext context) {
    final hasUrl =
        recommendation.displayUrl != null &&
        recommendation.displayUrl!.isNotEmpty;

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(9),
        border: Border.all(color: AppTheme.border),
      ),
      padding: const EdgeInsets.all(10),
      child: Row(
        children: [
          GestureDetector(
            onTap:
                hasUrl && onImageTap != null
                    ? () => onImageTap!(recommendation.displayUrl!)
                    : null,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child:
                  hasUrl
                      ? Image.network(
                        recommendation.displayUrl!,
                        width: 52,
                        height: 52,
                        fit: BoxFit.cover,
                        loadingBuilder: (ctx, child, progress) {
                          if (progress == null) return child;
                          return Container(
                            width: 52,
                            height: 52,
                            color: AppTheme.softSurface,
                            child: Center(
                              child: SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  value: progress.expectedTotalBytes != null
                                      ? progress.cumulativeBytesLoaded /
                                          progress.expectedTotalBytes!
                                      : null,
                                  color: AppTheme.primary,
                                ),
                              ),
                            ),
                          );
                        },
                        errorBuilder: (ctx, err, _) => _placeholder(),
                      )
                      : _placeholder(),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  recommendation.name,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: AppTheme.headingBrown,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 2),
                Text(
                  recommendation.category,
                  style: const TextStyle(
                    fontSize: 10,
                    color: AppTheme.mutedText,
                  ),
                ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Expanded(
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(2),
                        child: LinearProgressIndicator(
                          value: recommendation.score / 100,
                          minHeight: 3,
                          backgroundColor: AppTheme.border,
                          valueColor: const AlwaysStoppedAnimation<Color>(
                            AppTheme.primary,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      '${recommendation.score.toStringAsFixed(0)}%',
                      style: const TextStyle(
                        fontSize: 9,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.primary,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _placeholder() {
    return Container(
      width: 52,
      height: 52,
      color: AppTheme.softSurface,
      child: const Icon(Icons.diamond_outlined, size: 20, color: AppTheme.mutedText),
    );
  }
}
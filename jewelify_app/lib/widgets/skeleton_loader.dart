import 'package:flutter/material.dart';
import '../screens/app_theme.dart';

class SkeletonLoader extends StatefulWidget {
  final double width;
  final double height;
  final double borderRadius;

  const SkeletonLoader({
    super.key,
    required this.width,
    required this.height,
    this.borderRadius = 8,
  });

  @override
  State<SkeletonLoader> createState() => _SkeletonLoaderState();
}

class _SkeletonLoaderState extends State<SkeletonLoader>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);
    _anim = Tween<double>(begin: 0.3, end: 1.0).animate(
      CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _anim,
      builder: (_, _) => Container(
        width: widget.width,
        height: widget.height,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(widget.borderRadius),
          color: AppTheme.border.withValues(alpha: _anim.value),
        ),
      ),
    );
  }
}

class SkeletonLine extends StatelessWidget {
  final double height;
  final double? width;
  final double borderRadius;

  const SkeletonLine({
    super.key,
    this.height = 14,
    this.width,
    this.borderRadius = 4,
  });

  @override
  Widget build(BuildContext context) {
    return SkeletonLoader(
      width: width ?? double.infinity,
      height: height,
      borderRadius: borderRadius,
    );
  }
}

class SkeletonHistoryItem extends StatelessWidget {
  const SkeletonHistoryItem({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: AppTheme.cardDecoration,
      child: const Row(
        children: [
          SkeletonLoader(width: 40, height: 40, borderRadius: 8),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SkeletonLine(height: 12, width: 120),
                SizedBox(height: 6),
                SkeletonLine(height: 10, width: 80),
              ],
            ),
          ),
          SkeletonLine(height: 10, width: 40),
        ],
      ),
    );
  }
}

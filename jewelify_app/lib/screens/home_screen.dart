import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../screens/app_theme.dart';

class HomeScreen extends StatelessWidget {
  final VoidCallback toggleTheme;
  final bool isDarkMode;

  const HomeScreen({
    super.key,
    required this.toggleTheme,
    required this.isDarkMode,
  });

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final displayName = auth.name ?? auth.username ?? 'there';

    return Scaffold(
      body: SafeArea(
        child: CustomScrollView(
          slivers: [
            SliverToBoxAdapter(
              child: _buildTopBar(context, auth, displayName),
            ),
            SliverToBoxAdapter(child: _buildHeroCard(context)),
            SliverToBoxAdapter(child: _buildSectionLabel('QUICK ACTIONS')),
            SliverToBoxAdapter(child: _buildActionGrid(context)),
            SliverToBoxAdapter(child: _buildSectionLabel('HOW IT WORKS')),
            SliverToBoxAdapter(child: _buildAboutCard()),
            const SliverToBoxAdapter(child: SizedBox(height: 32)),
          ],
        ),
      ),
    );
  }

  Widget _buildTopBar(
      BuildContext context, AuthProvider auth, String displayName) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 20, 8, 8),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('WELCOME BACK', style: AppTheme.labelUppercase),
                const SizedBox(height: 2),
                Text(
                  displayName,
                  style: AppTheme.displayLarge.copyWith(fontSize: 22),
                ),
              ],
            ),
          ),
          IconButton(
            icon: Icon(
              isDarkMode
                  ? Icons.light_mode_outlined
                  : Icons.dark_mode_outlined,
              color: AppTheme.appNameBrown,
            ),
            onPressed: toggleTheme,
          ),
          IconButton(
            icon: const Icon(Icons.logout_outlined, color: AppTheme.mutedText),
            onPressed: () {
              auth.logout();
              Navigator.pushNamedAndRemoveUntil(
                  context, '/login', (_) => false);
            },
          ),
        ],
      ),
    );
  }

  Widget _buildHeroCard(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(20, 12, 20, 0),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFFB5451B), Color(0xFFD2691E)],
        ),
        borderRadius: BorderRadius.circular(20),
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'JEWELRY MATCHING',
            style: AppTheme.labelUppercase.copyWith(color: Colors.white60),
          ),
          const SizedBox(height: 8),
          Text(
            'Adorned\nbeautifully.',
            style: AppTheme.displayMedium.copyWith(
              color: Colors.white,
              fontSize: 26,
            ),
          ),
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: () => Navigator.pushNamed(context, '/upload'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.white,
              foregroundColor: AppTheme.primary,
              elevation: 0,
              padding:
                  const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(6),
              ),
              textStyle: AppTheme.labelUppercase.copyWith(
                color: AppTheme.primary,
                letterSpacing: 1.5,
                fontSize: 12,
              ),
            ),
            child: const Text('UPLOAD PHOTO'),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionLabel(String label) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 24, 24, 12),
      child: Text(label, style: AppTheme.labelUppercase),
    );
  }

  Widget _buildActionGrid(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        children: [
          Expanded(
            child: _ActionCard(
              icon: Icons.camera_alt_outlined,
              title: 'New Analysis',
              subtitle: 'Upload face & jewelry photos',
              onTap: () => Navigator.pushNamed(context, '/upload'),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: _ActionCard(
              icon: Icons.history_outlined,
              title: 'History',
              subtitle: 'View past recommendations',
              onTap: () => Navigator.pushNamed(context, '/history'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAboutCard() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      padding: const EdgeInsets.all(18),
      decoration: AppTheme.softCardDecoration,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'How it works',
            style: AppTheme.titleStyle.copyWith(fontSize: 16),
          ),
          const SizedBox(height: 14),
          const _AboutStep(number: '01', text: 'Upload a clear face photo'),
          const _AboutStep(number: '02', text: 'Add a photo of the jewelry'),
          const _AboutStep(number: '03', text: 'AI analyzes compatibility'),
          const _AboutStep(
            number: '04',
            text: 'Get your match score + recommendations',
          ),
        ],
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _ActionCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(18),
        decoration: AppTheme.cardDecoration,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: AppTheme.primary, size: 26),
            const SizedBox(height: 12),
            Text(
              title,
              style: AppTheme.bodyMedium
                  .copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 4),
            Text(subtitle, style: AppTheme.bodySmall, maxLines: 2),
          ],
        ),
      ),
    );
  }
}

class _AboutStep extends StatelessWidget {
  final String number;
  final String text;

  const _AboutStep({required this.number, required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          Text(
            number,
            style: AppTheme.labelUppercase.copyWith(
              color: AppTheme.primary,
              letterSpacing: 1,
            ),
          ),
          const SizedBox(width: 14),
          Expanded(child: Text(text, style: AppTheme.bodySmall)),
        ],
      ),
    );
  }
}

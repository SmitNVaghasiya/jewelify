import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import '../screens/app_theme.dart';

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  _UploadScreenState createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  File? _faceImage;
  File? _jewelryImage;
  final ImagePicker _picker = ImagePicker();
  bool _isLoading = false;

  Future<void> _pickImage(String type) async {
    try {
      final source = await _showImageSourceBottomSheet();
      if (source == null) return;

      final pickedFile = await _picker.pickImage(source: source);

      if (pickedFile != null) {
        final file = File(pickedFile.path);
        if (await file.exists()) {
          final fileSize = await file.length();
          const maxSizeInBytes = 5 * 1024 * 1024;
          if (fileSize > maxSizeInBytes) {
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text(
                    "Image size too large. Please select an image under 5MB.",
                  ),
                ),
              );
            }
            return;
          }

          final extension = pickedFile.path.split('.').last.toLowerCase();
          if (!['jpg', 'jpeg', 'png'].contains(extension)) {
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text(
                    "Unsupported image format. Please use JPG or PNG.",
                  ),
                ),
              );
            }
            return;
          }

          if (mounted) {
            setState(() {
              if (type == 'face') {
                _faceImage = file;
              } else {
                _jewelryImage = file;
              }
            });
          }
        } else {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text("Failed to load image.")),
            );
          }
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Error picking image: $e")));
      }
    }
  }

  Future<ImageSource?> _showImageSourceBottomSheet() async {
    return showModalBottomSheet<ImageSource>(
      context: context,
      backgroundColor: AppTheme.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder:
          (context) => Container(
            padding: const EdgeInsets.symmetric(vertical: 25, horizontal: 16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const SizedBox(height: 8),
                Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: AppTheme.border,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(height: 20),
                ListTile(
                  leading: Icon(
                    Icons.photo_library,
                    size: 28,
                    color: AppTheme.primary,
                  ),
                  title: Text("Gallery", style: AppTheme.bodyMedium),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 24),
                  onTap: () => Navigator.pop(context, ImageSource.gallery),
                ),
                Divider(color: AppTheme.border),
                ListTile(
                  leading: Icon(
                    Icons.camera_alt,
                    size: 28,
                    color: AppTheme.primary,
                  ),
                  title: Text("Camera", style: AppTheme.bodyMedium),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 24),
                  onTap: () => Navigator.pop(context, ImageSource.camera),
                ),
                const SizedBox(height: 8),
              ],
            ),
          ),
    );
  }

  Future<void> _onPopInvoked(bool didPop) async {
    if (didPop) return;
    if (_isLoading) {
      final shouldPop = await showDialog<bool>(
            context: context,
            builder:
                (context) => AlertDialog(
                  backgroundColor: AppTheme.surface,
                  title: Text(
                    "Cancel Upload?",
                    style: AppTheme.bodyMedium.copyWith(
                      fontWeight: FontWeight.bold,
                      color: AppTheme.headingBrown,
                    ),
                  ),
                  content: Text(
                    "Are you sure you want to cancel the upload process?",
                    style: AppTheme.bodyMedium,
                  ),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(context, false),
                      child: Text(
                        "No",
                        style: TextStyle(color: AppTheme.mutedText),
                      ),
                    ),
                    TextButton(
                      onPressed: () => Navigator.pop(context, true),
                      child: Text(
                        "Yes",
                        style: TextStyle(color: AppTheme.primary),
                      ),
                    ),
                  ],
                ),
          ) ??
          false;
      if (shouldPop && mounted) Navigator.pop(context);
    } else {
      Navigator.pop(context);
    }
  }

  Future<void> _uploadImages() async {
    if (_isLoading) return;

    if (_faceImage == null || _jewelryImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please upload both images')),
      );
      return;
    }

    setState(() => _isLoading = true);

    try {
      if (mounted) {
        await Navigator.pushNamed(
          context,
          '/processing',
          arguments: {'face': _faceImage, 'jewelry': _jewelryImage},
        );
        if (mounted) {
          setState(() {
            _isLoading = false;
            _faceImage = null;
            _jewelryImage = null;
          });
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isLoading = false);
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Error during processing: $e")));
      }
    }
  }

  Widget _buildImageCard({
    required String label,
    required String title,
    required String subtitle,
    required IconData icon,
    required File? image,
    required VoidCallback onTap,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: AppTheme.labelUppercase),
        const SizedBox(height: 10),
        GestureDetector(
          onTap: onTap,
          child: Container(
            width: double.infinity,
            height: 180,
            decoration: image == null
                ? AppTheme.softCardDecoration
                : BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppTheme.primary, width: 2),
                  ),
            clipBehavior: Clip.antiAlias,
            child: image == null
                ? Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(icon, size: 48, color: AppTheme.primary),
                      const SizedBox(height: 12),
                      Text(title, style: AppTheme.bodyMedium.copyWith(color: AppTheme.headingBrown)),
                      const SizedBox(height: 4),
                      Text(
                        subtitle,
                        style: AppTheme.bodySmall.copyWith(color: AppTheme.mutedText),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  )
                : Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.file(image, fit: BoxFit.cover),
                      Positioned(
                        bottom: 8,
                        right: 8,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 10,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: AppTheme.primary,
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: Text(
                            "Change",
                            style: AppTheme.bodySmall.copyWith(
                              color: Colors.white,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final bool canSubmit = _faceImage != null && _jewelryImage != null;

    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) => _onPopInvoked(didPop),
      child: Scaffold(
        backgroundColor: AppTheme.background,
        appBar: AppBar(
          backgroundColor: AppTheme.background,
          foregroundColor: AppTheme.appNameBrown,
          elevation: 0,
          title: Text('Upload Images', style: AppTheme.titleStyle),
        ),
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Expanded(
                  child: SingleChildScrollView(
                    child: Column(
                      children: [
                        _buildImageCard(
                          label: "FACE PHOTO",
                          title: "Upload Face Photo",
                          subtitle: "Clear, frontal face image",
                          icon: Icons.face,
                          image: _faceImage,
                          onTap: () => _pickImage('face'),
                        ),
                        const SizedBox(height: 24),
                        _buildImageCard(
                          label: "JEWELRY PHOTO",
                          title: "Upload Jewelry Photo",
                          subtitle: "Clear image of the jewelry",
                          icon: Icons.diamond,
                          image: _jewelryImage,
                          onTap: () => _pickImage('jewelry'),
                        ),
                        const SizedBox(height: 24),
                      ],
                    ),
                  ),
                ),
                ElevatedButton(
                  onPressed: canSubmit && !_isLoading ? _uploadImages : null,
                  style: AppTheme.primaryButton,
                  child: _isLoading
                      ? const SizedBox(
                          height: 20,
                          width: 20,
                          child: CircularProgressIndicator(
                            color: Colors.white,
                            strokeWidth: 2,
                          ),
                        )
                      : const Text("Match Jewelry"),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

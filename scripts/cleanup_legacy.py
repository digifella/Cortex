#!/usr/bin/env python3
"""
Legacy File Cleanup Script

This script removes old documentation files, unused artifacts, and cleans up
the repository for the v4.0.0 centralized version management release.

IMPORTANT: Review the files it will remove before running!
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def cleanup_legacy_files():
    """Remove outdated documentation and temporary files"""
    
    files_to_remove = [
        # Old documentation that should be consolidated into CLAUDE.md
        "CHANGELOG_2025-08-26.md",
        "DEPENDENCY_RESOLUTION_GUIDE.md", 
        "DEPLOYMENT_COMPLETE.md",
        "DEPLOYMENT_GUIDE.md",
        "DISTRIBUTION_README.md",
        "DISTRIBUTION_VALIDATION_REPORT.md",
        "DOCKER_FIXES_DOCUMENTATION.md",
        "DOCKER_INGESTION_FIXES_2025-08-24.md",
        "DOCKER_MODEL_PERFORMANCE_ANALYSIS.md", 
        "DOCKER_SESSION_FIXES_2025-08-23.md",
        "GEMINI.md",
        "HYBRID_MODEL_ARCHITECTURE.md",
        "IMPROVEMENT_PROGRESS_REPORT.md",
        "INNOVATION_ENGINE_SPRINT_PLAN.md",
        "Idea_Generator.md",
        "KNOWLEDGE_SYNTHESIS_PLAN.md",
        "MISTRAL_UPGRADE_GUIDE.md",
        "PRODUCTIZATION_PRD.md",
        "PYTORCH_META_TENSOR_ISSUE.md",
        "WORK_PROGRESS.md",
        
        # Temporary and test files
        "=2.6.0",
        "all_code_context.txt",
        "current_env_freeze.txt",
        "staging_test.json",
        "query.log",  # This gets regenerated
        
        # Screenshots in root directory (should be in archive or removed)
        "Screenshot 2025-08-26 170735.jpg",
        "Screenshot 2025-08-26 172155.jpg", 
        "Screenshot 2025-08-26 173450.jpg",
        "Screenshot 2025-08-26 175009.jpg",
        "Screenshot 2025-08-26 214711.png",
        "Screenshot 2025-08-27 124435.jpg",
    ]
    
    # Zone.Identifier files (Windows metadata)
    zone_files = []
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith(":Zone.Identifier"):
                zone_files.append(os.path.relpath(os.path.join(root, file), project_root))
    
    files_to_remove.extend(zone_files)
    
    # Backup files
    backup_files = []
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.backup') and 'venv' not in root:
                backup_files.append(os.path.relpath(os.path.join(root, file), project_root))
    
    files_to_remove.extend(backup_files)
    
    print("🧹 Legacy File Cleanup for Cortex Suite v4.0.0")
    print("=" * 50)
    
    print(f"\nFiles to remove ({len(files_to_remove)} total):")
    for file_path in sorted(files_to_remove):
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️  {file_path} (not found)")
    
    response = input(f"\nProceed with removing {len(files_to_remove)} files? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Cleanup cancelled.")
        return False
    
    removed_count = 0
    errors = []
    
    for file_path in files_to_remove:
        full_path = project_root / file_path
        try:
            if full_path.exists():
                if full_path.is_file():
                    full_path.unlink()
                    removed_count += 1
                elif full_path.is_dir():
                    import shutil
                    shutil.rmtree(full_path)
                    removed_count += 1
        except Exception as e:
            errors.append(f"{file_path}: {e}")
    
    print(f"\n✅ Successfully removed {removed_count} files")
    
    if errors:
        print(f"\n⚠️  Errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    
    return True

def create_archive_structure():
    """Create organized archive structure for any files we want to keep"""
    
    archive_dir = project_root / "archive" 
    docs_archive = archive_dir / "legacy_docs"
    
    # Create archive directories if they don't exist
    docs_archive.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Archive structure ready at: {docs_archive}")
    
    # Key documentation that should be preserved in archive
    important_docs = [
        "DISTRIBUTION_SYNC_CHECKLIST.md",  # Keep this - still relevant
    ]
    
    print("📋 Important documentation preserved:")
    for doc in important_docs:
        if (project_root / doc).exists():
            print(f"  ✅ {doc} (kept in root)")
        else:
            print(f"  ⚠️  {doc} (not found)")

if __name__ == "__main__":
    print("Cortex Suite Legacy Cleanup")
    print("This will remove outdated documentation and temporary files")
    print("for the v4.0.0 centralized version management release.\n")
    
    create_archive_structure()
    
    if cleanup_legacy_files():
        print("\n🎉 Cleanup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python scripts/version_manager.py --check")
        print("2. Test the application")
        print("3. Commit the cleaned up codebase")
    else:
        print("\n❌ Cleanup was cancelled or failed")
        sys.exit(1)
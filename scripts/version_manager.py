#!/usr/bin/env python3
"""
Cortex Suite Version Management Utility

This script helps manage versions across the entire Cortex Suite codebase.
It ensures version consistency and automates version-related tasks.

Usage:
    python scripts/version_manager.py --help
    python scripts/version_manager.py --check              # Check version consistency
    python scripts/version_manager.py --update-changelog   # Update CHANGELOG.md
    python scripts/version_manager.py --sync-all           # Sync all components
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.version_config import (
    get_version_string,
    get_changelog_entry,
    get_full_version_info,
    CORTEX_VERSION,
    VERSION_METADATA
)

class VersionManager:
    """Manages version consistency across the Cortex Suite"""
    
    def __init__(self):
        self.project_root = project_root
        self.issues = []
        
    def check_version_consistency(self) -> bool:
        """Check version consistency across all files"""
        print("🔍 Checking version consistency...")
        
        current_version = get_version_string()
        print(f"📌 Central version: {current_version}")
        
        # Files to check for version consistency
        files_to_check = [
            "Cortex_Suite.py",
            "README.md",
        ]
        
        inconsistent_files = []
        
        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists():
                if not self._check_file_version(full_path, current_version):
                    inconsistent_files.append(file_path)
            else:
                print(f"⚠️  File not found: {file_path}")
        
        # Check all page files
        pages_dir = self.project_root / "pages"
        if pages_dir.exists():
            for page_file in pages_dir.glob("*.py"):
                if not self._check_file_version(page_file, current_version):
                    inconsistent_files.append(f"pages/{page_file.name}")
        
        if inconsistent_files:
            print(f"❌ Version inconsistencies found in {len(inconsistent_files)} files:")
            for file in inconsistent_files:
                print(f"   - {file}")
            return False
        else:
            print("✅ All files have consistent version numbers")
            return True
    
    def _check_file_version(self, file_path: Path, expected_version: str) -> bool:
        """Check if a file contains the expected version"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Look for version patterns
            version_patterns = [
                r'VERSION\s*=\s*["\']([^"\']+)["\']',
                r'Version:\s*([^\s\n]+)',
                r'version\s*[:=]\s*["\']([^"\']+)["\']',
                r'v\d+\.\d+\.\d+',
            ]
            
            found_versions = set()
            for pattern in version_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                found_versions.update(matches)
            
            # Remove version prefixes and normalize
            normalized_versions = set()
            for version in found_versions:
                normalized = version.lstrip('v')
                normalized_versions.add(normalized)
            
            expected_normalized = expected_version.lstrip('v')
            
            # If we find versions but none match expected
            if normalized_versions and expected_normalized not in normalized_versions:
                return False
                
            return True
            
        except Exception as e:
            print(f"⚠️  Error checking {file_path}: {e}")
            return True  # Don't fail the check for read errors
    
    def update_changelog(self) -> bool:
        """Update CHANGELOG.md with current version info"""
        print("📝 Updating CHANGELOG.md...")
        
        changelog_path = self.project_root / "CHANGELOG.md"
        if not changelog_path.exists():
            print("❌ CHANGELOG.md not found")
            return False
        
        try:
            current_content = changelog_path.read_text(encoding='utf-8')
            new_entry = get_changelog_entry()
            
            # Check if this version already exists in changelog
            version_string = get_version_string()
            if version_string in current_content:
                print(f"ℹ️  Version {version_string} already exists in changelog")
                return True
            
            # Find insertion point (after "## [Unreleased]")
            unreleased_pattern = r'## \[Unreleased\]\s*\n'
            match = re.search(unreleased_pattern, current_content)
            
            if match:
                # Insert new entry after unreleased section
                insert_pos = match.end()
                updated_content = (
                    current_content[:insert_pos] + 
                    "\n" + new_entry + 
                    current_content[insert_pos:]
                )
                
                changelog_path.write_text(updated_content, encoding='utf-8')
                print(f"✅ Added {version_string} to CHANGELOG.md")
                return True
            else:
                print("⚠️  Could not find insertion point in CHANGELOG.md")
                return False
                
        except Exception as e:
            print(f"❌ Error updating changelog: {e}")
            return False
    
    def sync_all_components(self) -> bool:
        """Sync version across all components"""
        print("🔄 Syncing version across all components...")
        
        success = True
        current_version = get_version_string()
        
        # Update the VERSION_INFO dict in version_config.py
        if not self._update_version_info():
            success = False

        # Update main app file
        if not self._update_main_app_version(current_version):
            success = False

        # Update page files
        if not self._update_page_files(current_version):
            success = False

        # Update README files
        if not self._update_readme_files(current_version):
            success = False

        return success

    def _update_version_info(self) -> bool:
        """Sync the VERSION_INFO major/minor/patch dict to CORTEX_VERSION.

        VERSION_INFO is a hand-maintained dict that duplicates CORTEX_VERSION.
        It used to be missed by --sync-all entirely, so every release left it
        stale and failed tests/unit/test_version_config.py.
        """
        config_file = self.project_root / "cortex_engine" / "version_config.py"
        if not config_file.exists():
            print("❌ version_config.py not found")
            return False

        try:
            major, minor, patch = (int(p) for p in CORTEX_VERSION.split("."))
        except ValueError:
            print(f"❌ CORTEX_VERSION is not major.minor.patch: {CORTEX_VERSION}")
            return False

        try:
            content = config_file.read_text(encoding='utf-8')
            updated = re.sub(
                r'(VERSION_INFO = \{\s*\n\s*"major":\s*)\d+(,\s*\n\s*"minor":\s*)\d+(,\s*\n\s*"patch":\s*)\d+',
                rf'\g<1>{major}\g<2>{minor}\g<3>{patch}',
                content,
            )
            if updated == content:
                print("⚠️ VERSION_INFO already current, or its shape changed — check by hand")
            else:
                config_file.write_text(updated, encoding='utf-8')
            print(f"✅ Updated VERSION_INFO to {major}.{minor}.{patch}")
            return True
        except Exception as e:
            print(f"❌ Error updating VERSION_INFO: {e}")
            return False
    
    def _update_main_app_version(self, version: str) -> bool:
        """Update version in main app file"""
        app_file = self.project_root / "Cortex_Suite.py"
        if not app_file.exists():
            return False
        
        try:
            content = app_file.read_text(encoding='utf-8')
            
            # Update APP_VERSION constant
            content = re.sub(
                r'APP_VERSION\s*=\s*["\'][^"\']*["\']',
                f'APP_VERSION = "{version}"',
                content
            )
            
            # Update version in comments and captions
            content = re.sub(
                r'# Version:\s*v?\d+\.\d+\.\d+',
                f'# Version: {version}',
                content
            )
            
            app_file.write_text(content, encoding='utf-8')
            print(f"✅ Updated {app_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating {app_file.name}: {e}")
            return False
    

    def _update_page_files(self, version: str) -> bool:
        """Update version in page files"""
        pages_dir = self.project_root / "pages"
        if not pages_dir.exists():
            return False
        
        success = True
        version_without_v = version.lstrip('v')
        
        for page_file in pages_dir.glob("*.py"):
            try:
                content = page_file.read_text(encoding='utf-8')
                
                # Update PAGE_VERSION constants
                content = re.sub(
                    r'PAGE_VERSION\s*=\s*["\'][^"\']*["\']',
                    f'PAGE_VERSION = "{version}"',
                    content
                )
                
                # Update version in comments
                content = re.sub(
                    r'# Version:\s*v?\d+\.\d+\.\d+[^\n]*',
                    f'# Version: {version}',
                    content
                )
                
                page_file.write_text(content, encoding='utf-8')
                
            except Exception as e:
                print(f"⚠️  Error updating {page_file.name}: {e}")
                success = False
        
        if success:
            print("✅ Updated all page files")
        
        return success
    
    def _update_readme_files(self, version: str) -> bool:
        """Update version in README files"""
        readme_files = ["README.md"]
        success = True
        
        for file_path in readme_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                continue
                
            try:
                content = full_path.read_text(encoding='utf-8')
                
                # Update version references (be careful not to break other version numbers)
                content = re.sub(
                    r'Cortex Suite v\d+\.\d+\.\d+',
                    f'Cortex Suite {version}',
                    content
                )
                
                full_path.write_text(content, encoding='utf-8')
                print(f"✅ Updated {file_path}")
                
            except Exception as e:
                print(f"❌ Error updating {file_path}: {e}")
                success = False
        
        return success
    

    def show_version_info(self):
        """Display current version information"""
        info = get_full_version_info()
        
        print("📋 Current Version Information:")
        print(f"   Version: {info['formatted_version']}")
        print(f"   Release Date: {info['release_date']}")
        print(f"   Release Name: {info['release_name']}")
        print(f"   Description: {info['description']}")
        
        if info.get('breaking_changes'):
            print("\n🔥 Breaking Changes:")
            for change in info['breaking_changes']:
                print(f"   - {change}")
        
        if info.get('new_features'):
            print("\n✨ New Features:")
            for feature in info['new_features']:
                print(f"   - {feature}")
        
        if info.get('improvements'):
            print("\n🚀 Improvements:")
            for improvement in info['improvements']:
                print(f"   - {improvement}")

def main():
    parser = argparse.ArgumentParser(description="Cortex Suite Version Manager")
    parser.add_argument('--check', action='store_true', 
                       help='Check version consistency across files')
    parser.add_argument('--update-changelog', action='store_true',
                       help='Update CHANGELOG.md with current version')
    parser.add_argument('--sync-all', action='store_true',
                       help='Sync version across all components')
    parser.add_argument('--info', action='store_true',
                       help='Show current version information')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    vm = VersionManager()
    
    if args.info:
        vm.show_version_info()
    
    if args.check:
        consistent = vm.check_version_consistency()
        if not consistent:
            sys.exit(1)
    
    if args.update_changelog:
        if not vm.update_changelog():
            sys.exit(1)
    
    if args.sync_all:
        if not vm.sync_all_components():
            sys.exit(1)
        print("🎉 All components synced successfully!")

if __name__ == "__main__":
    main()
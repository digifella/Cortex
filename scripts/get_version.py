#!/usr/bin/env python3
"""
Simple script to extract the latest version from CHANGELOG.md
For use by batch files and shell scripts to get current version
"""

import re
import sys
from pathlib import Path

def get_latest_version(changelog_path=None):
    """Extract latest version from CHANGELOG.md"""
    if changelog_path is None:
        # Try to find CHANGELOG.md relative to script location
        script_dir = Path(__file__).parent
        changelog_path = script_dir.parent / "CHANGELOG.md"
        
        # Also try in Docker directory if we're in a Docker context
        if not changelog_path.exists():
            changelog_path = script_dir.parent / "docker" / "CHANGELOG.md"
    
    try:
        with open(changelog_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find first version line (## vX.Y.Z - YYYY-MM-DD)
        version_match = re.search(r'^## (v\d+\.\d+\.\d+)', content, re.MULTILINE)
        
        if version_match:
            return version_match.group(1)
        else:
            return "v0.0.0"  # Fallback if no version found
            
    except FileNotFoundError:
        return "v0.0.0"  # Fallback if file not found
    except Exception as e:
        print(f"Error reading changelog: {e}", file=sys.stderr)
        return "v0.0.0"

if __name__ == "__main__":
    # Support optional changelog path argument
    changelog_path = sys.argv[1] if len(sys.argv) > 1 else None
    version = get_latest_version(changelog_path)
    print(version)
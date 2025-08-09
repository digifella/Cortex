#!/usr/bin/env python3
"""
Cortex Suite Distribution Creator

This script creates a clean distribution package ready to share with others.
It excludes development files and creates a ZIP archive with all necessary components.
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def create_distribution():
    """Create a clean distribution package"""
    print("📦 Creating Cortex Suite Distribution Package...")
    print("=" * 50)
    
    # Get current directory
    source_dir = Path.cwd()
    
    # Create distribution directory name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dist_name = f"cortex-suite-distribution-{timestamp}"
    dist_dir = source_dir / "dist" / dist_name
    
    # Clean up any existing dist directory
    if (source_dir / "dist").exists():
        shutil.rmtree(source_dir / "dist")
    
    # Create distribution directory
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Creating distribution in: {dist_dir}")
    
    # Essential files and directories to include
    essential_items = [
        # Core application
        "Cortex_Suite.py",
        "requirements.txt",
        
        # Core modules
        "cortex_engine/",
        "pages/",
        "api/",
        
        # Docker setup (the key to easy distribution)
        "docker/",
        
        # Documentation
        "DISTRIBUTION_README.md",
        "CLAUDE.md",
    ]
    
    # Optional items that are nice to have
    optional_items = [
        "tests/",
        "examples/",
        "scripts/",
        "run_tests.py",
        "pytest.ini"
    ]
    
    # Files to exclude (development/system files)
    exclude_patterns = [
        "__pycache__",
        "*.pyc",
        ".git",
        ".gitignore",
        "venv/",
        "env/",
        ".env",
        "logs/",
        "*.log",
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        "*~",
        ".DS_Store",
        "Thumbs.db",
        "ai_databases/",
        "knowledge_base/",
        "backups/",
        "temp/",
        "tmp/",
        "staging/",
        "htmlcov/",
        ".coverage",
        ".pytest_cache/",
        "*:Zone.Identifier"
    ]
    
    copied_files = 0
    
    # Copy essential items
    print("\n📋 Copying essential files...")
    for item in essential_items:
        source_path = source_dir / item
        
        if source_path.exists():
            dest_path = dist_dir / item
            
            if source_path.is_file():
                # Copy file
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
                print(f"✅ {item}")
                copied_files += 1
            else:
                # Copy directory
                shutil.copytree(source_path, dest_path, 
                              ignore=shutil.ignore_patterns(*exclude_patterns))
                file_count = len(list(dest_path.rglob('*')))
                print(f"✅ {item} ({file_count} files)")
                copied_files += file_count
        else:
            print(f"⚠️ {item} - not found")
    
    # Copy optional items if they exist
    print("\n📋 Copying optional files...")
    for item in optional_items:
        source_path = source_dir / item
        
        if source_path.exists():
            dest_path = dist_dir / item
            
            if source_path.is_file():
                shutil.copy2(source_path, dest_path)
                print(f"✅ {item}")
                copied_files += 1
            else:
                shutil.copytree(source_path, dest_path,
                              ignore=shutil.ignore_patterns(*exclude_patterns))
                file_count = len(list(dest_path.rglob('*')))
                print(f"✅ {item} ({file_count} files)")
                copied_files += file_count
    
    # Create a simple README for the distribution
    readme_content = """# Cortex Suite - AI Knowledge Management System

## Quick Start (3 steps):

1. **Install Docker Desktop**: https://www.docker.com/products/docker-desktop/

2. **Run Cortex Suite**:
   - Windows: Double-click `docker/run-cortex.bat`
   - Mac/Linux: Double-click `docker/run-cortex.sh`

3. **Open browser**: http://localhost:8501

First launch takes 5-10 minutes (downloads AI models).
Subsequent launches take ~30 seconds.

## Features:
- 📚 Document ingestion (PDFs, Word, PowerPoint, images)
- 🚀 **Batch ingest mode** - process hundreds of files automatically 
- 📖 **NEW: Comprehensive help system** - detailed guidance and tooltips
- 🔍 AI-powered search with GraphRAG technology
- 📝 Proposal generation using your knowledge base
- 🗂️ Advanced collection management
- 💾 Backup/restore with Windows path support
- 🔧 REST API for integrations

## System Requirements:
- Docker Desktop installed
- 8GB RAM recommended (4GB minimum)
- 10GB free storage space
- Internet connection (for initial setup only)

## Help:
- Read `docker/QUICK_START.md` for detailed instructions
- Read `DISTRIBUTION_README.md` for complete documentation
- System runs completely locally - your data stays private!

Enjoy your AI-powered knowledge management system! 🚀
"""
    
    with open(dist_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ README.md (distribution guide)")
    copied_files += 1
    
    # Calculate distribution size
    total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print(f"\n📊 Distribution created:")
    print(f"   Files: {copied_files}")
    print(f"   Size: {size_mb:.1f}MB")
    print(f"   Location: {dist_dir}")
    
    # Create ZIP archive
    print(f"\n📦 Creating ZIP archive...")
    zip_path = source_dir / "dist" / f"{dist_name}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in dist_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(dist_dir.parent)
                zipf.write(file_path, arcname)
    
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    
    print(f"✅ ZIP created: {zip_path}")
    print(f"📊 ZIP size: {zip_size_mb:.1f}MB")
    
    print("\n" + "=" * 50)
    print("🎉 DISTRIBUTION PACKAGE READY!")
    print(f"📦 Share this file: {zip_path}")
    print("\n📋 Instructions for recipients:")
    print("1. Extract the ZIP file")
    print("2. Install Docker Desktop")
    print("3. Run docker/run-cortex.bat (Windows) or docker/run-cortex.sh (Mac/Linux)")
    print("4. Open http://localhost:8501")
    print("\n💡 First launch downloads ~4GB of AI models")
    print("   Subsequent launches are much faster!")
    
    return zip_path

if __name__ == "__main__":
    try:
        zip_path = create_distribution()
        print(f"\n✅ Success! Distribution ready at: {zip_path}")
    except Exception as e:
        print(f"\n❌ Error creating distribution: {e}")
        import traceback
        traceback.print_exc()
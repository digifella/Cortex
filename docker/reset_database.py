#!/usr/bin/env python3
"""
Database Reset Utility for Docker Environment
Version: v1.0.0
Date: 2025-08-26

Fixes database schema issues by recreating fresh ChromaDB instance.
"""

import os
import shutil
from pathlib import Path

def reset_chromadb():
    """Reset ChromaDB database to fix schema issues."""
    
    # Standard Docker container paths
    db_paths = [
        "/data/ai_databases/knowledge_hub_db",
        "./cortex_data/ai_databases/knowledge_hub_db",
        # Fallback local paths  
        "../ai_databases/knowledge_hub_db"
    ]
    
    for db_path in db_paths:
        path = Path(db_path)
        if path.exists():
            print(f"🗑️ Removing old database at: {path}")
            shutil.rmtree(path)
            print(f"✅ Removed: {path}")
    
    print("🆕 Database will be recreated on next ingestion")
    print("📋 To rebuild your knowledge base:")
    print("   1. Go to Knowledge Ingest page")
    print("   2. Select your documents") 
    print("   3. Run ingestion process")

if __name__ == "__main__":
    print("🔧 Cortex Database Reset Utility")
    print("=" * 40)
    
    confirm = input("⚠️ This will DELETE all knowledge base data. Continue? (yes/no): ")
    if confirm.lower() in ['yes', 'y']:
        reset_chromadb()
        print("✅ Database reset complete!")
    else:
        print("❌ Database reset cancelled.")
from pathlib import Path
import shutil
from datetime import datetime
from typing import List


class BackupManager:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.chroma_dir = self.db_path / "knowledge_hub_db"
        self.backups_dir = self.db_path / "backups"
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, name: str = None) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = name or f"backup_{ts}"
        dest = self.backups_dir / backup_name
        dest.mkdir(parents=True, exist_ok=True)
        # Copy chroma dir if exists
        if self.chroma_dir.exists():
            shutil.copytree(self.chroma_dir, dest / "knowledge_hub_db", dirs_exist_ok=True)
        # Copy collections and graph if present
        for fname in ["working_collections.json", "knowledge_cortex.gpickle"]:
            src = self.db_path / fname
            if src.exists():
                shutil.copy2(src, dest / fname)
        return backup_name

    def list_backups(self) -> List[str]:
        if not self.backups_dir.exists():
            return []
        return sorted([p.name for p in self.backups_dir.iterdir() if p.is_dir()])

    def restore_backup(self, name: str) -> bool:
        src = self.backups_dir / name
        if not src.exists():
            return False
        # Restore chroma dir
        chroma_src = src / "knowledge_hub_db"
        if chroma_src.exists():
            shutil.copytree(chroma_src, self.chroma_dir, dirs_exist_ok=True)
        # Restore files
        for fname in ["working_collections.json", "knowledge_cortex.gpickle"]:
            file_src = src / fname
            if file_src.exists():
                shutil.copy2(file_src, self.db_path / fname)
        return True

    def delete_backup(self, name: str) -> bool:
        src = self.backups_dir / name
        if not src.exists():
            return False
        shutil.rmtree(src)
        return True


import json
import os
from datetime import datetime
from pathlib import Path
from .utils import convert_to_docker_mount_path, get_logger

logger = get_logger(__name__)

def _collections_file_path() -> str:
    db_path = os.environ.get('AI_DATABASE_PATH', '/data/ai_databases')
    safe_db = convert_to_docker_mount_path(db_path)
    return os.path.join(safe_db, 'working_collections.json')


class WorkingCollectionManager:
    def __init__(self):
        self.collections_file = _collections_file_path()
        self.collections = self._load()

    def _load(self):
        data = {}
        try:
            folder = os.path.dirname(self.collections_file)
            os.makedirs(folder, exist_ok=True)
            if os.path.exists(self.collections_file):
                with open(self.collections_file, 'r') as f:
                    data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load collections file: {e}")
            data = {}

        now = datetime.now().isoformat()
        if 'default' not in data:
            data['default'] = {"name": "default", "doc_ids": [], "created_at": now, "modified_at": now}
        return data

    def _save(self):
        try:
            Path(os.path.dirname(self.collections_file)).mkdir(parents=True, exist_ok=True)
            with open(self.collections_file, 'w') as f:
                json.dump(self.collections, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Failed to save collections: {e}")
            return False

    def get_collection_names(self):
        return list(self.collections.keys())

    def get_doc_ids_by_name(self, name: str):
        return self.collections.get(name, {}).get('doc_ids', [])

    def create_collection(self, name: str) -> bool:
        if name and name not in self.collections:
            now = datetime.now().isoformat()
            self.collections[name] = {"name": name, "doc_ids": [], "created_at": now, "modified_at": now}
            return self._save()
        return False

    def add_docs_by_id_to_collection(self, name: str, doc_ids: list):
        if not doc_ids:
            return
        if name not in self.collections:
            self.create_collection(name)
        seen = set(self.collections[name].get('doc_ids', []))
        added = False
        for did in doc_ids:
            if did not in seen:
                self.collections[name]['doc_ids'].append(did)
                seen.add(did)
                added = True
        if added:
            self.collections[name]['modified_at'] = datetime.now().isoformat()
            self._save()


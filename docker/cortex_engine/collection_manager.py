import json
import os
from datetime import datetime
from pathlib import Path
from .utils import convert_to_docker_mount_path, get_logger
from .config_manager import ConfigManager

logger = get_logger(__name__)

def _collections_file_path() -> str:
    """Resolve collections file under the user-selected DB path.
    Priority: ConfigManager.ai_database_path -> env AI_DATABASE_PATH -> sensible default.
    """
    try:
        cfg = ConfigManager().get_config()
        configured_db = cfg.get('ai_database_path') or ''
    except Exception:
        configured_db = ''

    db_path = configured_db or os.environ.get('AI_DATABASE_PATH', '/data/ai_databases')
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

    def rename_collection(self, old_name: str, new_name: str) -> bool:
        if old_name in self.collections and new_name not in self.collections and old_name != "default":
            self.collections[new_name] = self.collections.pop(old_name)
            self.collections[new_name]['name'] = new_name
            self.collections[new_name]['modified_at'] = datetime.now().isoformat()
            return self._save()
        return False

    def delete_collection(self, name: str) -> bool:
        if name in self.collections and name != "default":
            try:
                del self.collections[name]
                return self._save()
            except Exception:
                return False
        return False

    def merge_collections(self, source_name: str, dest_name: str) -> bool:
        if source_name not in self.collections or dest_name not in self.collections or source_name == dest_name:
            return False
        source_doc_ids = self.get_doc_ids_by_name(source_name)
        self.add_docs_by_id_to_collection(dest_name, source_doc_ids)
        return self.delete_collection(source_name)

    def export_collection_files(self, name: str, output_dir: str, vector_collection) -> tuple:
        doc_ids = self.get_doc_ids_by_name(name)
        if not doc_ids:
            logger.info(f"No documents found in collection '{name}'")
            return [], []
        try:
            safe_output_dir = convert_to_docker_mount_path(output_dir)
            Path(safe_output_dir).mkdir(parents=True, exist_ok=True)
            results = vector_collection.get(where={"doc_id": {"$in": doc_ids}}, include=["metadatas"])
            source_paths = set(meta['doc_posix_path'] for meta in results.get('metadatas', []) if 'doc_posix_path' in meta)
            copied_files, failed_files = [], []
            for src_path_str in source_paths:
                src_path = Path(src_path_str)
                dest_path = Path(safe_output_dir) / src_path.name
                try:
                    if src_path.exists():
                        import shutil
                        shutil.copy(src_path, dest_path)
                        copied_files.append(str(dest_path))
                    else:
                        raise FileNotFoundError(f"Source file not found: {src_path}")
                except Exception as e:
                    failed_files.append(f"{src_path_str} (Reason: {e})")
            return copied_files, failed_files
        except Exception as e:
            logger.error(f"Export collection failed: {e}")
            return [], [str(e)]

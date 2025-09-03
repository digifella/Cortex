import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict


class ProposalManager:
    """Minimal proposal manager for Docker distribution.
    Stores proposals in <AI_DATABASE_PATH>/proposals/proposals.json
    """

    def __init__(self):
        self.base_path = Path(os.environ.get("AI_DATABASE_PATH", "/data/ai_databases"))
        self.store_dir = self.base_path / "proposals"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.store_file = self.store_dir / "proposals.json"
        if not self.store_file.exists():
            self._write([])

    def _read(self) -> List[Dict]:
        try:
            with open(self.store_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write(self, data: List[Dict]) -> None:
        with open(self.store_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def create_proposal(self, name: str) -> str:
        data = self._read()
        pid = f"p_{int(datetime.now().timestamp())}"
        now = datetime.now().isoformat()
        data.append({
            "id": pid,
            "name": name,
            "status": "Drafting",
            "last_modified": now,
        })
        self._write(data)
        return pid

    def list_proposals(self) -> List[Dict]:
        items = self._read()
        # Convert last_modified to datetime for display consistency
        for p in items:
            try:
                p["last_modified"] = datetime.fromisoformat(p["last_modified"])
            except Exception:
                p["last_modified"] = datetime.now()
        return sorted(items, key=lambda p: p["last_modified"], reverse=True)

    def load_proposal(self, proposal_id: str) -> Dict:
        for p in self._read():
            if p["id"] == proposal_id:
                # Load persisted state/blobs if present
                state_file = self.store_dir / f"{proposal_id}_state.json"
                template_file = self.store_dir / f"{proposal_id}_template.bin"
                generated_file = self.store_dir / f"{proposal_id}_generated.bin"
                state = {}
                if state_file.exists():
                    try:
                        state = json.loads(state_file.read_text())
                    except Exception:
                        state = {}
                data = {
                    "meta": p,
                    "state": state,
                }
                if template_file.exists():
                    data["template_bytes"] = template_file.read_bytes()
                if generated_file.exists():
                    data["generated_doc_bytes"] = generated_file.read_bytes()
                return data
        return {}

    def save_proposal(self, proposal_id: str, session_state: Dict, template_bytes: bytes, generated_doc_bytes: bytes = None) -> None:
        data = self._read()
        for p in data:
            if p["id"] == proposal_id:
                p["last_modified"] = datetime.now().isoformat()
                break
        self._write(data)
        # Persist session_state and blobs
        (self.store_dir / f"{proposal_id}_state.json").write_text(json.dumps({"session_state": session_state}, indent=2))
        if template_bytes:
            (self.store_dir / f"{proposal_id}_template.bin").write_bytes(template_bytes)
        if generated_doc_bytes:
            (self.store_dir / f"{proposal_id}_generated.bin").write_bytes(generated_doc_bytes)

    def update_proposal_status(self, proposal_id: str, status: str) -> None:
        data = self._read()
        for p in data:
            if p["id"] == proposal_id:
                p["status"] = status
                p["last_modified"] = datetime.now().isoformat()
                break
        self._write(data)

    def delete_proposal(self, proposal_id: str) -> None:
        data = [p for p in self._read() if p["id"] != proposal_id]
        self._write(data)

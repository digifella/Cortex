# ## File: cortex_engine/proposal_manager.py
# Version: 1.1.0 (Save/Resume Fix)
# Date: 2025-07-13
# Purpose: Manages the lifecycle of proposals, including creating, saving,
#          loading, listing, and deleting.
#          - FIX (v1.1.0): Made the save_proposal function robust by handling
#            both bytes and io.BytesIO objects for generated documents. This
#            resolves a TypeError during the save process after document assembly.

import os
import json
import shutil
import uuid
import io
from pathlib import Path
from datetime import datetime

# Define the directory where all proposal data will be stored
PROPOSALS_DIR = Path(__file__).parent.parent / "proposals"

class ProposalManager:
    """Handles all file-based operations for managing proposal state."""

    def __init__(self):
        """Ensures the main proposals directory exists."""
        PROPOSALS_DIR.mkdir(exist_ok=True)

    def list_proposals(self) -> list:
        """
        Scans the proposals directory and returns a list of all proposals
        with their metadata.
        """
        proposals = []
        for proposal_dir in PROPOSALS_DIR.iterdir():
            if proposal_dir.is_dir():
                state_file = proposal_dir / "state.json"
                if state_file.exists():
                    try:
                        # <<<--- LOADING from "state.json"
                        with open(state_file, 'r') as f:
                            state_data = json.load(f)
                        proposals.append({
                            "id": proposal_dir.name,
                            "name": state_data.get("name", "Untitled Proposal"),
                            "status": state_data.get("status", "Drafting"),
                            "last_modified": datetime.fromtimestamp(state_file.stat().st_mtime),
                        })
                    except (json.JSONDecodeError, IOError):
                        continue # Skip corrupted state files
        proposals.sort(key=lambda p: p['last_modified'], reverse=True)
        return proposals

    def create_proposal(self, name: str) -> str:
        """
        Creates a new proposal directory and an initial state file.
        Returns the unique ID of the new proposal.
        """
        proposal_id = str(uuid.uuid4())
        proposal_dir = PROPOSALS_DIR / proposal_id
        proposal_dir.mkdir() # <<<--- CREATING directory on disk

        initial_state = {
            "name": name,
            "status": "Drafting",
            "created_at": datetime.now().isoformat(),
            "session_state": {}
        }
        # <<<--- SAVING to "state.json"
        with open(proposal_dir / "state.json", 'w') as f:
            json.dump(initial_state, f, indent=4)

        return proposal_id

    def save_proposal(self, proposal_id: str, session_data: dict, template_bytes: bytes, generated_doc_bytes: bytes = None):
        """
        Saves the complete state of a proposal, including the session data
        and the template/draft .docx files.
        """
        proposal_dir = PROPOSALS_DIR / proposal_id
        if not proposal_dir.exists():
            raise FileNotFoundError(f"Proposal with ID {proposal_id} not found.")

        state_file = proposal_dir / "state.json"

        with open(state_file, 'r') as f:
            state_data = json.load(f)

        state_data['session_state'] = session_data
        state_data['status'] = "Drafting"

        # <<<--- SAVING to "state.json"
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=4)

        if template_bytes:
            # <<<--- SAVING to "template.docx"
            with open(proposal_dir / "template.docx", "wb") as f:
                f.write(template_bytes)

        if generated_doc_bytes:
            # FIX: Check if the object is a BytesIO stream and get its value if so.
            doc_bytes_to_write = generated_doc_bytes.getvalue() if isinstance(generated_doc_bytes, io.BytesIO) else generated_doc_bytes
            
            if doc_bytes_to_write:
                # <<<--- SAVING to "draft.docx"
                with open(proposal_dir / "draft.docx", "wb") as f:
                    f.write(doc_bytes_to_write)

    def load_proposal(self, proposal_id: str) -> dict:
        """
        Loads all data for a given proposal ID and returns it in a dictionary.
        """
        proposal_dir = PROPOSALS_DIR / proposal_id
        if not proposal_dir.exists():
            return None

        state_file = proposal_dir / "state.json"
        template_file = proposal_dir / "template.docx"
        draft_file = proposal_dir / "draft.docx"

        loaded_data = {}
        # <<<--- LOADING from "state.json"
        with open(state_file, 'r') as f:
            loaded_data['state'] = json.load(f)

        if template_file.exists():
             # <<<--- LOADING from "template.docx"
             with open(template_file, "rb") as f:
                loaded_data['template_bytes'] = f.read()

        if draft_file.exists():
             # <<<--- LOADING from "draft.docx"
             with open(draft_file, "rb") as f:
                loaded_data['generated_doc_bytes'] = f.read()

        return loaded_data

    def delete_proposal(self, proposal_id: str):
        """Permanently deletes a proposal's directory."""
        proposal_dir = PROPOSALS_DIR / proposal_id
        if proposal_dir.exists():
            shutil.rmtree(proposal_dir) # <<<--- DELETING directory from disk

    def update_proposal_status(self, proposal_id: str, new_status: str):
        """Updates just the status of a proposal."""
        proposal_dir = PROPOSALS_DIR / proposal_id
        state_file = proposal_dir / "state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            state_data['status'] = new_status
            # <<<--- SAVING to "state.json"
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=4)
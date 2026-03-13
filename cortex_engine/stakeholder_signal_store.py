from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cortex_engine.config_manager import ConfigManager
from cortex_engine.stakeholder_signal_matcher import match_signal_to_profiles, normalize_lookup
from cortex_engine.target_update_detector import detect_profile_change_artifacts
from cortex_engine.utils import convert_windows_to_wsl_path

_ORG_STOPWORDS = {
    "pty",
    "ltd",
    "limited",
    "llc",
    "inc",
    "corp",
    "corporation",
    "company",
    "co",
    "group",
    "consulting",
    "consultancy",
    "services",
    "service",
    "holdings",
    "the",
    "and",
}
_DEFAULT_OLLAMA_WATCH_MODEL = "qwen3.5:9b"
_OLLAMA_WATCH_MODEL_FALLBACKS = (
    "qwen3.5:9b",
    "qwen3.5:9b-q8_0",
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen2.5:14b",
    "mistral-small3.2:latest",
    "mistral:latest",
)
_DEFAULT_OLLAMA_WATCH_TIMEOUT_SECONDS = 300

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_address(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, str] = {}
    for key in ("street", "city", "state", "postcode", "country"):
        item = str(value.get(key) or "").strip()
        if item:
            normalized[key] = item
    return normalized


def _normalize_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"active", "inactive", "watch", "former"}:
        return text
    return "active"


def _normalize_watch_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"off", "watch"}:
        return text
    return "off"


def _ollama_watch_timeout_seconds() -> int:
    raw_value = str(os.environ.get("CORTEX_WATCH_OLLAMA_TIMEOUT") or "").strip()
    if raw_value:
        try:
            value = int(raw_value)
            if value > 0:
                return value
        except Exception:
            logger.warning("Invalid CORTEX_WATCH_OLLAMA_TIMEOUT=%r; using default %s", raw_value, _DEFAULT_OLLAMA_WATCH_TIMEOUT_SECONDS)
    return _DEFAULT_OLLAMA_WATCH_TIMEOUT_SECONDS


def _resolve_storage_root(base_path: Optional[Path] = None) -> Path:
    if base_path is not None:
        root = Path(base_path)
        root.mkdir(parents=True, exist_ok=True)
        return root

    config = ConfigManager().get_config()
    raw_db_path = str(config.get("ai_database_path") or "").strip()
    if not raw_db_path:
        raise RuntimeError("ai_database_path is not configured; Cortex stakeholder signal store cannot initialize")

    safe_db_path = raw_db_path if os.path.exists("/.dockerenv") else convert_windows_to_wsl_path(raw_db_path)
    root = Path(safe_db_path) / "stakeholder_intel"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _org_tokens(value: str) -> set[str]:
    normalized = normalize_lookup(value)
    return {token for token in normalized.split() if token and token not in _ORG_STOPWORDS}


def orgs_compatible(left: str, right: str) -> bool:
    left_norm = normalize_lookup(left)
    right_norm = normalize_lookup(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    left_tokens = _org_tokens(left)
    right_tokens = _org_tokens(right)
    if not left_tokens or not right_tokens:
        return left_norm == right_norm

    if left_tokens == right_tokens:
        return True
    smaller, larger = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    return smaller.issubset(larger)


class StakeholderSignalStore:
    """JSON-backed store for stakeholder profiles, incoming signals, and generated artefacts."""

    def __init__(self, base_path: Optional[Path] = None):
        self.root = _resolve_storage_root(base_path)
        self.state_path = self.root / "state.json"
        self.raw_dir = self.root / "raw_signals"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write_state(self._initial_state())

    @staticmethod
    def _initial_state() -> Dict[str, Any]:
        return {
            "updated_at": _utc_now_iso(),
            "profiles": [],
            "signals": [],
            "observed_facts": [],
            "update_suggestions": [],
        }

    def _read_state(self) -> Dict[str, Any]:
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            payload = json.loads(raw) if raw.strip() else {}
            if not isinstance(payload, dict):
                return self._initial_state()
            payload.setdefault("profiles", [])
            payload.setdefault("signals", [])
            payload.setdefault("observed_facts", [])
            payload.setdefault("update_suggestions", [])
            payload.setdefault("updated_at", _utc_now_iso())
            return payload
        except Exception:
            return self._initial_state()

    def _write_state(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = _utc_now_iso()
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="stakeholder_intel_", suffix=".json", dir=str(self.root))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=True, indent=2, sort_keys=True)
            os.replace(tmp_path, self.state_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def get_state(self) -> Dict[str, Any]:
        return self._read_state()

    def list_profiles(self, org_name: str = "") -> List[Dict[str, Any]]:
        state = self._read_state()
        profiles = list(state.get("profiles") or [])
        if org_name:
            profiles = [p for p in profiles if orgs_compatible(p.get("org_name", ""), org_name)]
        return sorted(profiles, key=lambda item: (item.get("org_name", ""), item.get("canonical_name", "")))

    def list_signals(self, org_name: str = "", matched_only: bool = False, limit: int = 200) -> List[Dict[str, Any]]:
        state = self._read_state()
        signals = list(state.get("signals") or [])
        if org_name:
            signals = [s for s in signals if orgs_compatible(s.get("org_name", ""), org_name)]
        if matched_only:
            signals = [s for s in signals if s.get("matches")]
        signals = sorted(signals, key=lambda item: item.get("received_at", ""), reverse=True)
        return signals[:limit]

    def list_update_suggestions(self, org_name: str = "", status: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        state = self._read_state()
        suggestions = list(state.get("update_suggestions") or [])
        if org_name:
            suggestions = [item for item in suggestions if orgs_compatible(item.get("org_name", ""), org_name)]
        if status:
            status_norm = str(status).strip().lower()
            suggestions = [item for item in suggestions if str(item.get("status") or "").strip().lower() == status_norm]
        suggestions = sorted(suggestions, key=lambda item: item.get("created_at", ""), reverse=True)
        return suggestions[:limit]

    def list_observed_facts(self, org_name: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        state = self._read_state()
        facts = list(state.get("observed_facts") or [])
        if org_name:
            facts = [item for item in facts if orgs_compatible(item.get("org_name", ""), org_name)]
        return sorted(facts, key=lambda item: item.get("created_at", ""), reverse=True)[:limit]

    def get_profile(self, profile_key: str) -> Optional[Dict[str, Any]]:
        wanted = str(profile_key).strip()
        if not wanted:
            return None
        state = self._read_state()
        for profile in state.get("profiles") or []:
            if profile.get("profile_key") == wanted:
                return dict(profile)
        return None

    def save_profile(self, profile_key: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wanted = str(profile_key).strip()
        if not wanted:
            return None
        state = self._read_state()
        profiles = list(state.get("profiles") or [])
        updated_profile: Optional[Dict[str, Any]] = None
        for profile in profiles:
            if profile.get("profile_key") != wanted:
                continue
            for field_name, field_value in updates.items():
                if field_name == "external_profile_id":
                    next_value = str(field_value or "").strip()
                    if not next_value:
                        continue
                    profile[field_name] = next_value
                    continue
                if field_name == "address":
                    profile[field_name] = _normalize_address(field_value)
                    continue
                if field_name == "status":
                    profile[field_name] = _normalize_status(field_value)
                    continue
                if field_name == "watch_status":
                    profile[field_name] = _normalize_watch_status(field_value)
                    continue
                profile[field_name] = field_value
            profile["updated_at"] = _utc_now_iso()
            updated_profile = dict(profile)
            break
        if updated_profile is None:
            return None
        state["profiles"] = profiles
        self._write_state(state)
        return updated_profile

    def delete_profile(self, profile_key: str) -> bool:
        wanted = str(profile_key).strip()
        if not wanted:
            return False
        state = self._read_state()
        profiles = [profile for profile in state.get("profiles") or [] if profile.get("profile_key") != wanted]
        if len(profiles) == len(state.get("profiles") or []):
            return False

        suggestions = [
            item for item in state.get("update_suggestions") or []
            if item.get("profile_key") != wanted
        ]
        facts = [
            item for item in state.get("observed_facts") or []
            if item.get("profile_key") != wanted
        ]
        state["profiles"] = profiles
        state["update_suggestions"] = suggestions
        state["observed_facts"] = facts
        self._write_state(state)
        return True

    def delete_signal(self, signal_id: str) -> bool:
        wanted = str(signal_id).strip()
        if not wanted:
            return False
        state = self._read_state()
        signals = list(state.get("signals") or [])
        target_signal = next((signal for signal in signals if signal.get("signal_id") == wanted), None)
        if target_signal is None:
            return False

        state["signals"] = [signal for signal in signals if signal.get("signal_id") != wanted]
        fact_ids = set(target_signal.get("observed_fact_ids") or [])
        suggestion_ids = set(target_signal.get("update_suggestion_ids") or [])
        state["observed_facts"] = [
            item for item in state.get("observed_facts") or []
            if item.get("fact_id") not in fact_ids and item.get("signal_id") != wanted
        ]
        state["update_suggestions"] = [
            item for item in state.get("update_suggestions") or []
            if item.get("suggestion_id") not in suggestion_ids and item.get("signal_id") != wanted
        ]
        self._write_state(state)
        return True

    def review_update_suggestion(
        self,
        suggestion_id: str,
        status: str,
        reviewed_by: str = "streamlit",
        apply_to_profile: bool = False,
    ) -> Optional[Dict[str, Any]]:
        wanted = str(suggestion_id).strip()
        if not wanted:
            return None
        next_status = str(status).strip().lower()
        if next_status not in {"accepted", "rejected", "pending"}:
            raise ValueError(f"Unsupported suggestion status: {status!r}")

        state = self._read_state()
        suggestions = list(state.get("update_suggestions") or [])
        updated_item: Optional[Dict[str, Any]] = None
        for item in suggestions:
            if item.get("suggestion_id") != wanted:
                continue
            item["status"] = next_status
            item["reviewed_at"] = _utc_now_iso()
            item["reviewed_by"] = reviewed_by
            updated_item = item
            break
        if updated_item is None:
            return None
        if next_status == "accepted" and apply_to_profile and updated_item.get("profile_key"):
            for profile in state.get("profiles") or []:
                if profile.get("profile_key") != updated_item.get("profile_key"):
                    continue
                profile[updated_item["field_name"]] = updated_item.get("proposed_value", "")
                if updated_item["field_name"] == "current_employer":
                    employers = [str(item).strip() for item in profile.get("known_employers") or [] if str(item).strip()]
                    new_employer = str(updated_item.get("proposed_value") or "").strip()
                    if new_employer and new_employer not in employers:
                        employers.append(new_employer)
                    profile["known_employers"] = employers
                profile["updated_at"] = _utc_now_iso()
                break
        state["update_suggestions"] = suggestions
        self._write_state(state)
        return updated_item

    def upsert_profiles(
        self,
        org_name: str,
        profiles: List[Dict[str, Any]],
        source: str = "",
        trace_id: str = "",
        replace_org_scope: bool = False,
    ) -> Dict[str, Any]:
        state = self._read_state()
        existing_profiles = list(state.get("profiles") or [])
        by_profile_key = {
            profile["profile_key"]: profile for profile in existing_profiles if profile.get("profile_key")
        }
        by_external_profile_id = {
            str(profile.get("external_profile_id") or "").strip(): profile
            for profile in existing_profiles
            if str(profile.get("external_profile_id") or "").strip()
        }
        added = 0
        updated = 0
        synced_profile_keys: set[str] = set()

        for raw_profile in profiles:
            profile = self._normalize_profile(raw_profile, org_name=org_name, source=source, trace_id=trace_id)
            external_profile_id = str(profile.get("external_profile_id") or "").strip()
            current = None
            if external_profile_id:
                current = by_external_profile_id.get(external_profile_id)
            if current is None:
                current = by_profile_key.get(profile["profile_key"])
            if current is None:
                current = self._match_legacy_profile(existing_profiles, profile)
            if current:
                profile["profile_key"] = current.get("profile_key", profile["profile_key"])
                profile["created_at"] = current.get("created_at", profile["created_at"])
                if not external_profile_id:
                    profile["external_profile_id"] = str(current.get("external_profile_id") or "").strip()
                updated += 1
            else:
                added += 1
            by_profile_key[profile["profile_key"]] = profile
            synced_profile_keys.add(str(profile["profile_key"]))
            if str(profile.get("external_profile_id") or "").strip():
                by_external_profile_id[str(profile["external_profile_id"]).strip()] = profile

        if replace_org_scope:
            retained_keys = set(by_profile_key.keys())
            for existing in existing_profiles:
                same_org = orgs_compatible(existing.get("org_name", ""), org_name)
                same_source = str(existing.get("source") or "").strip() == str(source or "").strip()
                if same_org and same_source and existing.get("profile_key") not in synced_profile_keys:
                    retained_keys.discard(existing.get("profile_key"))
            state["profiles"] = sorted(
                [profile for key, profile in by_profile_key.items() if key in retained_keys],
                key=lambda item: (item.get("org_name", ""), item.get("canonical_name", "")),
            )
        else:
            state["profiles"] = sorted(
                by_profile_key.values(),
                key=lambda item: (item.get("org_name", ""), item.get("canonical_name", "")),
            )
        self._write_state(state)
        return {
            "org_name": org_name,
            "profile_count": len(state["profiles"]),
            "added": added,
            "updated": updated,
        }

    @staticmethod
    def _match_legacy_profile(existing_profiles: List[Dict[str, Any]], profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        target_type = str(profile.get("target_type") or "").strip().lower()
        canonical_name = normalize_lookup(profile.get("canonical_name") or "")
        linkedin_url = normalize_lookup(profile.get("linkedin_url") or "")
        website_url = normalize_lookup(profile.get("website_url") or "")
        current_employer = normalize_lookup(profile.get("current_employer") or "")
        candidates: List[tuple[int, Dict[str, Any]]] = []

        for existing in existing_profiles:
            if str(existing.get("target_type") or "").strip().lower() != target_type:
                continue
            score = 0
            if linkedin_url and normalize_lookup(existing.get("linkedin_url") or "") == linkedin_url:
                score += 4
            if website_url and normalize_lookup(existing.get("website_url") or "") == website_url:
                score += 4
            if canonical_name and normalize_lookup(existing.get("canonical_name") or "") == canonical_name:
                score += 3
            if current_employer and normalize_lookup(existing.get("current_employer") or "") == current_employer:
                score += 1
            if score > 0:
                candidates.append((score, existing))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1].get("updated_at", "")), reverse=True)
        return candidates[0][1]

    def ingest_signal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._read_state()
        signal = self._normalize_signal(payload)
        signals = list(state.get("signals") or [])

        for existing in signals:
            if existing.get("signal_hash") == signal["signal_hash"]:
                return existing

        profiles = [
            profile for profile in state.get("profiles") or []
            if orgs_compatible(profile.get("org_name", ""), signal.get("org_name", ""))
        ]
        signal["matches"] = match_signal_to_profiles(signal, profiles)
        signal["matched_profile_keys"] = [match["profile_key"] for match in signal["matches"]]
        signal["needs_review"] = bool(signal["matches"]) and signal["matches"][0]["score"] < 0.75
        signal["raw_file"] = str(self._write_raw_signal_markdown(signal))

        artifacts = detect_profile_change_artifacts(signal, signal["matches"])
        signal["observed_fact_ids"] = [item["fact_id"] for item in artifacts["observed_facts"]]
        signal["update_suggestion_ids"] = [item["suggestion_id"] for item in artifacts["update_suggestions"]]

        signals.append(signal)
        state["signals"] = sorted(signals, key=lambda item: item.get("received_at", ""), reverse=True)
        state["observed_facts"] = self._merge_unique_items(
            state.get("observed_facts") or [],
            artifacts["observed_facts"],
            key_field="fact_id",
            extra_fields={"org_name": signal["org_name"], "created_at": _utc_now_iso()},
        )
        state["update_suggestions"] = self._merge_unique_items(
            state.get("update_suggestions") or [],
            artifacts["update_suggestions"],
            key_field="suggestion_id",
            extra_fields={"org_name": signal["org_name"], "created_at": _utc_now_iso()},
        )
        self._write_state(state)
        return signal

    def generate_digest(
        self,
        org_name: str,
        since_ts: str = "",
        profile_keys: Optional[List[str]] = None,
        max_items: int = 25,
        include_needs_review: bool = True,
        matched_only: bool = True,
        llm_synthesis: bool = False,
        llm_provider: str = "ollama",
        llm_model: str = "",
    ) -> Dict[str, Any]:
        state = self._read_state()
        signals = self.list_signals(org_name=org_name, matched_only=matched_only, limit=1000)
        if since_ts:
            signals = [signal for signal in signals if str(signal.get("received_at", "")) >= since_ts]
        if profile_keys:
            wanted = set(profile_keys)
            signals = [
                signal
                for signal in signals
                if wanted.intersection(set(signal.get("matched_profile_keys") or []))
            ]
        if not include_needs_review:
            signals = [signal for signal in signals if not signal.get("needs_review")]

        signals = signals[:max_items]
        generated_at = _utc_now_iso()
        digest_id = f"digest_{hashlib.sha1(f'{org_name}|{since_ts}|{len(signals)}|{generated_at}'.encode('utf-8')).hexdigest()[:16]}"
        digest_path = self.root / "digests"
        digest_path.mkdir(parents=True, exist_ok=True)
        output_path = digest_path / f"{digest_id}.md"
        profiles_covered = len(
            {
                str(profile_key).strip()
                for signal in signals
                for profile_key in signal.get("matched_profile_keys") or []
                if str(profile_key).strip()
            }
        )
        period_start = since_ts or (signals[-1].get("received_at", "") if signals else "")
        period_end = generated_at
        llm_synthesised = False
        actual_llm_model = str(llm_model or "").strip()

        body_lines = self._build_mechanical_digest_lines(signals)
        if llm_synthesis and signals:
            raw_data = self._prepare_digest_data(signals, state=state, org_name=org_name)
            synthesised, actual_llm_model = self._llm_synthesise(
                raw_data=raw_data,
                org_name=org_name,
                provider=llm_provider,
                model=llm_model,
            )
            if synthesised:
                body_lines = synthesised.strip().splitlines()
                llm_synthesised = True

        lines = [
            "---",
            f"digest_id: {digest_id}",
            f"org_name: {org_name}",
            f"generated_at: {generated_at}",
            f"since_ts: {since_ts}",
            f"signal_count: {len(signals)}",
            f"llm_synthesised: {str(llm_synthesised).lower()}",
            f"llm_provider: {str(llm_provider or '').strip()}",
            f"llm_model: {actual_llm_model}",
            "---",
            "",
            f"# Stakeholder Intelligence Digest",
            "",
            f"Organisation: {org_name}",
            f"Generated: {generated_at}",
            "",
        ]
        lines.extend(body_lines)

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return {
            "digest_id": digest_id,
            "org_name": org_name,
            "signal_count": len(signals),
            "signals": signals,
            "output_path": str(output_path),
            "llm_synthesised": llm_synthesised,
            "profiles_covered": profiles_covered,
            "period_start": period_start,
            "period_end": period_end,
            "llm_provider": str(llm_provider or "").strip(),
            "llm_model": actual_llm_model,
        }

    def _build_mechanical_digest_lines(self, signals: List[Dict[str, Any]]) -> List[str]:
        if not signals:
            return ["No matching signals for this window."]

        lines: List[str] = []
        for idx, signal in enumerate(signals, start=1):
            top_match = (signal.get("matches") or [{}])[0]
            lines.extend(
                [
                    f"## {idx}. {signal.get('subject') or signal.get('signal_id')}",
                    f"- Received: {signal.get('received_at', '')}",
                    f"- Target type: {signal.get('target_type', '')}",
                    f"- Candidate: {signal.get('parsed_candidate_name', '')}",
                    f"- Top match: {top_match.get('canonical_name', 'No match')} ({top_match.get('score', 0):.2f})" if top_match else "- Top match: No match",
                    f"- Why it matters: {'; '.join(top_match.get('reasons') or []) or 'Pending analyst review'}",
                ]
            )
            if signal.get("primary_url"):
                lines.append(f"- Source URL: {signal['primary_url']}")
            if signal.get("text_note"):
                lines.extend(["", signal["text_note"]])
            elif signal.get("raw_text"):
                lines.extend(["", signal["raw_text"][:1200]])
            lines.append("")
        return lines

    def _prepare_digest_data(self, signals: List[Dict[str, Any]], state: Dict[str, Any], org_name: str) -> str:
        fact_map = {
            str(item.get("fact_id") or "").strip(): item
            for item in state.get("observed_facts") or []
            if str(item.get("fact_id") or "").strip()
        }
        suggestion_map = {
            str(item.get("suggestion_id") or "").strip(): item
            for item in state.get("update_suggestions") or []
            if str(item.get("suggestion_id") or "").strip()
        }
        entries: List[Dict[str, Any]] = []
        for signal in signals:
            top_match = (signal.get("matches") or [{}])[0]
            facts = [
                fact_map[fact_id]
                for fact_id in signal.get("observed_fact_ids") or []
                if fact_id in fact_map
            ]
            suggestions = [
                suggestion_map[suggestion_id]
                for suggestion_id in signal.get("update_suggestion_ids") or []
                if suggestion_id in suggestion_map
            ]
            entry: Dict[str, Any] = {
                "signal_id": signal.get("signal_id", ""),
                "candidate_name": signal.get("parsed_candidate_name", ""),
                "target_type": signal.get("target_type", ""),
                "signal_type": signal.get("signal_type", ""),
                "source_system": signal.get("source_system", ""),
                "subject": signal.get("subject", ""),
                "received_at": signal.get("received_at", ""),
                "matched_profile": top_match.get("canonical_name", ""),
                "match_score": top_match.get("score", 0),
                "match_reasons": top_match.get("reasons", []),
                "text_note": (signal.get("text_note") or signal.get("raw_text") or "")[:800],
                "primary_url": signal.get("primary_url", ""),
                "needs_review": signal.get("needs_review", False),
            }
            if facts:
                entry["observed_facts"] = facts
            if suggestions:
                entry["update_suggestions"] = suggestions
            entries.append(entry)
        return json.dumps({"org_name": org_name, "signals": entries}, ensure_ascii=True, indent=2)

    def _llm_synthesise(self, raw_data: str, org_name: str, provider: str, model: str) -> tuple[Optional[str], str]:
        provider_name = str(provider or "ollama").strip().lower()
        system_prompt = (
            "You are an intelligence analyst producing a concise stakeholder watch report.\n\n"
            "Given structured signal data, produce a markdown intelligence digest with these sections:\n\n"
            "## Organisation Updates\n"
            "Group signals about companies. Note expansions, board changes, strategic moves.\n\n"
            "## Individual Updates\n"
            "Group signals about people. Highlight:\n"
            "- Role changes (with confidence level and previous role if known)\n"
            "- Content/posts (brief summary of topic)\n"
            "- Engagement signals (reactions, comments — mention what they engaged with)\n\n"
            "## Pending Review Items\n"
            "List any signals flagged needs_review=true or with update_suggestions pending.\n"
            "Include the suggested change and confidence.\n\n"
            "## Summary\n"
            "One paragraph synthesis: key themes, notable patterns, items requiring attention.\n\n"
            "Rules:\n"
            "- Be concise — bullet points preferred over paragraphs\n"
            "- Include source attribution and dates\n"
            "- For role changes, explicitly note old→new when known\n"
            "- Flag high-confidence changes (>0.8) vs lower-confidence ones\n"
            "- If no signals exist for a section, omit it entirely\n"
            "- Do NOT fabricate information — only report what's in the data"
        )
        user_prompt = f"Generate a WATCH report for organisation: {org_name}\n\nSignal data:\n{raw_data}"

        if provider_name == "ollama":
            return self._call_ollama(system_prompt, user_prompt, model or _DEFAULT_OLLAMA_WATCH_MODEL)
        if provider_name == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt, model or "claude-haiku-4-5-20251001")
        logger.warning("Unsupported digest LLM provider: %s", provider_name)
        return None, str(model or "").strip()

    def _preferred_ollama_watch_models(self, requested_model: str) -> List[str]:
        candidates: List[str] = []
        for item in [
            str(os.environ.get("CORTEX_WATCH_OLLAMA_MODEL") or "").strip(),
            str(requested_model or "").strip(),
            *_OLLAMA_WATCH_MODEL_FALLBACKS,
        ]:
            if item and item not in candidates:
                candidates.append(item)
        return candidates

    def _get_ollama_installed_models(self) -> List[str]:
        import requests

        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models") or []
            return [
                str(item.get("name") or "").strip()
                for item in models
                if str(item.get("name") or "").strip()
            ]
        except Exception as exc:
            logger.warning("Could not query installed Ollama models: %s", exc)
            return []

    def _resolve_ollama_model(self, requested_model: str) -> str:
        candidates = self._preferred_ollama_watch_models(requested_model)
        installed = set(self._get_ollama_installed_models())
        if installed:
            for candidate in candidates:
                if candidate in installed:
                    return candidate
        return candidates[0] if candidates else _DEFAULT_OLLAMA_WATCH_MODEL

    def _call_ollama(self, system: str, user: str, model: str) -> tuple[Optional[str], str]:
        import requests

        candidates = self._preferred_ollama_watch_models(model)
        resolved_model = self._resolve_ollama_model(model)
        ordered_models = [resolved_model] + [candidate for candidate in candidates if candidate != resolved_model]
        timeout_seconds = _ollama_watch_timeout_seconds()

        for candidate in ordered_models:
            try:
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": candidate,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "stream": False,
                    },
                    timeout=timeout_seconds,
                )
                if response.status_code >= 400:
                    body = ""
                    try:
                        body = str(response.json())
                    except Exception:
                        body = response.text
                    if response.status_code == 404 or "not found" in body.lower():
                        logger.warning("Ollama model unavailable for digest synthesis: %s", candidate)
                        continue
                response.raise_for_status()
                logger.info("Using Ollama WATCH digest model: %s", candidate)
                return str(response.json().get("message", {}).get("content") or "").strip() or None, candidate
            except Exception as exc:
                message = str(exc).lower()
                if "not found" in message and candidate != ordered_models[-1]:
                    logger.warning("Ollama model unavailable for digest synthesis: %s", candidate)
                    continue
                logger.warning("Ollama digest synthesis failed with model %s: %s", candidate, exc)
                if candidate != ordered_models[-1]:
                    continue
                return None, candidate
        return None, resolved_model

    def _call_anthropic(self, system: str, user: str, model: str) -> tuple[Optional[str], str]:
        import requests

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set; skipping digest synthesis")
            return None, model
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": model,
                    "max_tokens": 4096,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=60,
            )
            response.raise_for_status()
            content = response.json().get("content") or []
            if not content:
                return None, model
            return str(content[0].get("text") or "").strip() or None, model
        except Exception as exc:
            logger.warning("Anthropic digest synthesis failed: %s", exc)
            return None, model

    def _normalize_profile(
        self,
        raw_profile: Dict[str, Any],
        org_name: str,
        source: str = "",
        trace_id: str = "",
    ) -> Dict[str, Any]:
        canonical_name = str(raw_profile.get("canonical_name") or raw_profile.get("name") or "").strip()
        if not canonical_name:
            raise ValueError("Stakeholder profile requires canonical_name or name")

        target_type = str(raw_profile.get("target_type") or "person").strip().lower() or "person"
        watch_status = _normalize_watch_status(raw_profile.get("watch_status") or "off")
        linkedin_url = str(raw_profile.get("linkedin_url") or "").strip()
        website_url = str(raw_profile.get("website_url") or "").strip()
        current_employer = str(raw_profile.get("current_employer") or raw_profile.get("org_name") or "").strip()
        external_profile_id = str(
            raw_profile.get("external_profile_id") or raw_profile.get("website_profile_id") or raw_profile.get("id") or ""
        ).strip()
        aliases = [str(item).strip() for item in raw_profile.get("aliases") or [] if str(item).strip()]
        known_employers = [str(item).strip() for item in raw_profile.get("known_employers") or [] if str(item).strip()]
        if current_employer and current_employer not in known_employers:
            known_employers.append(current_employer)

        key_basis = external_profile_id or linkedin_url or website_url or canonical_name
        key_material = "|".join([normalize_lookup(org_name), target_type, normalize_lookup(key_basis)])
        profile_key = hashlib.sha1(key_material.encode("utf-8")).hexdigest()[:16]

        return {
            "profile_key": profile_key,
            "external_profile_id": external_profile_id,
            "org_name": str(org_name).strip(),
            "target_type": target_type,
            "canonical_name": canonical_name,
            "email": str(raw_profile.get("email") or "").strip(),
            "industry": str(raw_profile.get("industry") or "").strip(),
            "function": str(raw_profile.get("function") or "").strip(),
            "status": _normalize_status(raw_profile.get("status") or "active"),
            "last_verified_at": str(raw_profile.get("last_verified_at") or "").strip(),
            "current_employer": current_employer,
            "current_role": str(raw_profile.get("current_role") or "").strip(),
            "linkedin_url": linkedin_url,
            "website_url": website_url,
            "address": _normalize_address(raw_profile.get("address") or {}),
            "acn_abn": str(raw_profile.get("acn_abn") or "").strip(),
            "phone": str(raw_profile.get("phone") or "").strip(),
            "parent_entity": str(raw_profile.get("parent_entity") or "").strip(),
            "notes": str(raw_profile.get("notes") or "").strip(),
            "watch_status": watch_status,
            "tags": [str(item).strip() for item in raw_profile.get("tags") or [] if str(item).strip()],
            "aliases": aliases,
            "known_employers": known_employers,
            "source": str(source or raw_profile.get("source") or "").strip(),
            "trace_id": str(trace_id or raw_profile.get("trace_id") or "").strip(),
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }

    def _normalize_signal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        org_name = str(payload.get("org_name") or payload.get("tenant_id") or "default").strip()
        raw_text = str(payload.get("raw_text") or payload.get("body") or "").strip()
        subject = str(payload.get("subject") or "").strip()
        primary_url = str(payload.get("primary_url") or payload.get("content") or "").strip()
        text_note = str(payload.get("text_note") or "").strip()
        received_at = str(payload.get("received_at") or _utc_now_iso()).strip()
        message_id = str(payload.get("message_id") or "").strip()

        signal_hash_source = "|".join(
            [
                org_name,
                message_id,
                subject,
                primary_url,
                raw_text,
                received_at,
            ]
        )
        signal_hash = hashlib.sha256(signal_hash_source.encode("utf-8")).hexdigest()
        signal_id = str(payload.get("signal_id") or f"sig_{signal_hash[:16]}")

        return {
            "signal_id": signal_id,
            "signal_hash": signal_hash,
            "trace_id": str(payload.get("trace_id") or "").strip(),
            "source_system": str(payload.get("source_system") or "market_radar").strip(),
            "signal_type": str(payload.get("signal_type") or "linkedin_notification").strip(),
            "target_type": str(payload.get("target_type") or "person").strip().lower() or "person",
            "org_name": org_name,
            "submitted_by": str(payload.get("submitted_by") or "").strip(),
            "received_at": received_at,
            "message_id": message_id,
            "subject": subject,
            "raw_text": raw_text,
            "primary_url": primary_url,
            "text_note": text_note,
            "notification_kind": str(payload.get("notification_kind") or "").strip(),
            "parsed_candidate_name": str(payload.get("parsed_candidate_name") or payload.get("stakeholder_name") or "").strip(),
            "parsed_candidate_employer": str(payload.get("parsed_candidate_employer") or payload.get("stakeholder_employer") or "").strip(),
            "tags": [str(item).strip() for item in payload.get("tags") or [] if str(item).strip()],
            "matches": [],
            "matched_profile_keys": [],
            "needs_review": False,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }

    def _write_raw_signal_markdown(self, signal: Dict[str, Any]) -> Path:
        output_path = self.raw_dir / f"{signal['signal_id']}.md"
        matched_names = ", ".join(match["canonical_name"] for match in signal.get("matches") or [])
        lines = [
            "---",
            f"signal_id: {signal['signal_id']}",
            f"trace_id: {signal.get('trace_id', '')}",
            f"org_name: {signal.get('org_name', '')}",
            f"received_at: {signal.get('received_at', '')}",
            f"signal_type: {signal.get('signal_type', '')}",
            f"target_type: {signal.get('target_type', '')}",
            f"primary_url: {signal.get('primary_url', '')}",
            f"matched_stakeholders: {matched_names}",
            "---",
            "",
            f"# {signal.get('subject') or signal['signal_id']}",
            "",
        ]
        if signal.get("text_note"):
            lines.extend(["## Note", signal["text_note"], ""])
        if signal.get("raw_text"):
            lines.extend(["## Raw Email Body", signal["raw_text"], ""])
        if signal.get("matches"):
            lines.append("## Top Matches")
            for match in signal["matches"]:
                lines.append(
                    f"- {match['canonical_name']} ({match['score']:.2f}) — {'; '.join(match.get('reasons') or [])}"
                )
            lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    @staticmethod
    def _merge_unique_items(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]], key_field: str, extra_fields: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        merged = {str(item.get(key_field) or ""): dict(item) for item in existing if item.get(key_field)}
        for item in incoming:
            key = str(item.get(key_field) or "")
            if not key:
                continue
            payload = dict(item)
            for field_name, field_value in (extra_fields or {}).items():
                payload.setdefault(field_name, field_value)
            payload.setdefault("created_at", _utc_now_iso())
            merged[key] = payload
        return sorted(merged.values(), key=lambda item: item.get("created_at", ""), reverse=True)

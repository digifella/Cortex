from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import tempfile
import threading
import time
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests


# Ensure repo root is importable when running `python worker/worker.py`
ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import handlers safely in both execution modes:
# - script: `python worker/worker.py`
# - module: `python -m worker.worker`
if __package__ in (None, ""):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from handlers import HANDLERS
else:
    from .handlers import HANDLERS
from cortex_engine.handoff_contract import normalize_handoff_metadata
from cortex_engine.queue_monitor import QueueMonitorStore


def load_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


@dataclass
class Config:
    server_url: str
    secret_key: str
    poll_interval: int
    worker_id: str
    supported_types: str
    log_level: str
    temp_dir: Path
    heartbeat_interval: int
    request_timeout: int
    queue_monitor_state_path: Path
    cortex_api_url: str
    cortex_tunnel_url: str
    cortex_meta_sync_interval: int


def read_config() -> Config:
    env_path = ROOT / "worker" / "config.env"
    file_vars = load_env_file(env_path)

    def get(name: str, default: str) -> str:
        return os.environ.get(name, file_vars.get(name, default))

    cfg = Config(
        server_url=get("QUEUE_SERVER_URL", "").strip(),
        secret_key=get("QUEUE_SECRET_KEY", "").strip(),
        poll_interval=int(get("POLL_INTERVAL", "15")),
        worker_id=get("WORKER_ID", "worker-local-1").strip(),
        supported_types=get("SUPPORTED_TYPES", "pdf_anonymise").strip(),
        log_level=get("LOG_LEVEL", "INFO").strip().upper(),
        temp_dir=Path(get("TEMP_DIR", str(ROOT / "worker" / "tmp"))),
        heartbeat_interval=int(get("HEARTBEAT_INTERVAL", "60")),
        request_timeout=int(get("REQUEST_TIMEOUT", "60")),
        queue_monitor_state_path=Path(
            get("QUEUE_MONITOR_STATE_PATH", str(ROOT / "worker" / "tmp" / "queue_monitor_state.json"))
        ),
        cortex_api_url=get("CORTEX_API_URL", "http://127.0.0.1:8000").strip(),
        cortex_tunnel_url=get("CORTEX_TUNNEL_URL", "").strip(),
        cortex_meta_sync_interval=int(get("CORTEX_META_SYNC_INTERVAL", "300")),
    )

    if not cfg.server_url or not cfg.secret_key:
        raise RuntimeError("Missing QUEUE_SERVER_URL or QUEUE_SECRET_KEY in worker/config.env or environment")
    return cfg


class QueueClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"X-Queue-Key": cfg.secret_key})

    def _request(self, method: str, params: Dict[str, str], **kwargs):
        query = dict(params)
        response = self.session.request(
            method=method,
            url=self.cfg.server_url,
            params=query,
            timeout=self.cfg.request_timeout,
            **kwargs,
        )
        # Some hosts/proxies strip custom headers. On 403, retry once with query key fallback.
        if response.status_code == 403 and "key" not in query:
            fallback_query = dict(query)
            fallback_query["key"] = self.cfg.secret_key
            response = self.session.request(
                method=method,
                url=self.cfg.server_url,
                params=fallback_query,
                timeout=self.cfg.request_timeout,
                **kwargs,
            )
        response.raise_for_status()
        ctype = response.headers.get("Content-Type", "").lower()
        if "application/json" in ctype:
            return response.json()
        return response

    def poll(self) -> Optional[dict]:
        payload = self._request(
            "GET",
            {
                "action": "poll",
                "types": self.cfg.supported_types,
                "worker_id": self.cfg.worker_id,
            },
        )
        return payload.get("job") if isinstance(payload, dict) else None

    def download_input(self, job_id: int, out_path: Path) -> Optional[Path]:
        response = self._request("GET", {"action": "download_input", "id": str(job_id)})
        if not isinstance(response, requests.Response):
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
        return out_path

    def heartbeat(self, job_id: int) -> None:
        self._request("POST", {"action": "heartbeat", "id": str(job_id)})

    def fail(self, job_id: int, error_message: str) -> None:
        self._request(
            "POST",
            {"action": "fail", "id": str(job_id)},
            data={"error": error_message[:5000]},
        )

    def complete(self, job_id: int, output_data: dict, output_file: Optional[Path]) -> None:
        data = {"output_data": json.dumps(output_data or {})}
        files = None
        if output_file and output_file.exists():
            files = {"file": (output_file.name, open(output_file, "rb"), "application/octet-stream")}
        try:
            self._request(
                "POST",
                {"action": "complete", "id": str(job_id)},
                data=data,
                files=files,
            )
        finally:
            if files:
                files["file"][1].close()


class HeartbeatThread(threading.Thread):
    def __init__(self, client: QueueClient, job_id: int, interval: int):
        super().__init__(daemon=True)
        self.client = client
        self.job_id = job_id
        self.interval = interval
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.wait(self.interval):
            try:
                self.client.heartbeat(self.job_id)
            except Exception:
                logging.exception("Heartbeat failed for job %s", self.job_id)


class JobCancelledError(RuntimeError):
    pass


def parse_input_data(raw_value) -> dict:
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    text = str(raw_value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def process_job(client: QueueClient, cfg: Config, store: QueueMonitorStore, job: dict) -> None:
    job_id = int(job["id"])
    job_type = str(job.get("type", ""))
    handler = HANDLERS.get(job_type)
    if handler is None:
        client.fail(job_id, f"Unsupported job type: {job_type}")
        return

    input_data = parse_input_data(job.get("input_data"))
    handoff = normalize_handoff_metadata(job=job, input_data=input_data)
    trace_id = handoff["trace_id"]

    logging.info(
        "Claimed job id=%s type=%s trace_id=%s source=%s scope=%s/%s",
        job_id,
        job_type,
        trace_id,
        handoff["source_system"],
        handoff["tenant_id"],
        handoff["project_id"],
    )
    store.upsert_job(
        job_id,
        job_type=job_type,
        trace_id=trace_id,
        source_system=handoff["source_system"],
        tenant_id=handoff["tenant_id"],
        project_id=handoff["project_id"],
        input_filename=str(job.get("input_filename") or ""),
        status="claimed",
        message="Job claimed by worker",
        progress_pct=1,
        stage="claimed",
        worker_id=cfg.worker_id,
    )
    store.append_event("Job claimed", job_id=job_id)
    work_dir = Path(tempfile.mkdtemp(prefix=f"queue_job_{job_id}_", dir=str(cfg.temp_dir)))
    input_path: Optional[Path] = None

    hb = HeartbeatThread(client, job_id, cfg.heartbeat_interval)
    hb.start()
    try:
        if store.is_cancel_requested(job_id):
            raise JobCancelledError("Cancelled before processing started")

        input_filename = str(job.get("input_filename", "") or "")
        if input_filename:
            store.upsert_job(job_id, status="processing", stage="download_input", message="Downloading input", progress_pct=5)
            input_path = work_dir / input_filename
            downloaded = client.download_input(job_id, input_path)
            if downloaded is None:
                raise RuntimeError("Failed to download input file")
            input_path = downloaded
            logging.info("Downloaded input to %s", input_path)
            store.append_event(f"Downloaded input: {input_path.name}", job_id=job_id)
            if store.is_cancel_requested(job_id):
                raise JobCancelledError("Cancelled after input download")

        progress_event_state = {"last_stage": "", "last_pct": -10, "last_message": ""}

        def _progress_cb(progress_pct: float, message: str, stage: Optional[str] = None) -> None:
            pct = max(0, min(100, int(progress_pct)))
            update_stage = stage or "processing"
            msg = str(message or "").strip()
            store.upsert_job(
                job_id,
                status="processing",
                stage=update_stage,
                message=msg[:500],
                progress_pct=pct,
            )
            stage_changed = update_stage != progress_event_state["last_stage"]
            pct_step = (pct - int(progress_event_state["last_pct"])) >= 10
            message_changed = bool(msg) and msg != progress_event_state["last_message"]
            if stage_changed or pct_step or message_changed:
                if msg:
                    store.append_event(
                        msg,
                        level="info",
                        job_id=job_id,
                        stage=update_stage,
                        progress_pct=pct,
                        source="worker.progress",
                    )
                progress_event_state["last_stage"] = update_stage
                progress_event_state["last_pct"] = pct
                progress_event_state["last_message"] = msg
            if store.is_cancel_requested(job_id):
                raise JobCancelledError("Cancelled by operator")

        store.upsert_job(job_id, status="processing", stage="handler_start", message="Running handler", progress_pct=10)

        handler_kwargs = {"input_path": input_path, "input_data": input_data, "job": job}
        handler_params = set(inspect.signature(handler).parameters.keys())
        if "progress_cb" in handler_params:
            handler_kwargs["progress_cb"] = _progress_cb
        if "is_cancelled_cb" in handler_params:
            handler_kwargs["is_cancelled_cb"] = lambda: store.is_cancel_requested(job_id)

        result = handler(**handler_kwargs)
        output_data = result.get("output_data", {}) if isinstance(result, dict) else {}
        output_file = result.get("output_file") if isinstance(result, dict) else None
        output_file = Path(output_file) if output_file else None
        if not isinstance(output_data, dict):
            output_data = {"result": output_data}
        output_data["_handoff"] = handoff

        if store.is_cancel_requested(job_id):
            raise JobCancelledError("Cancelled before completion upload")

        store.upsert_job(job_id, status="uploading_result", stage="complete", message="Uploading result", progress_pct=95)
        client.complete(job_id, output_data=output_data, output_file=output_file)
        logging.info("Completed job id=%s trace_id=%s", job_id, trace_id)
        store.upsert_job(job_id, status="completed", stage="done", message="Completed", progress_pct=100)
        store.append_event("Job completed", level="info", job_id=job_id)
    except JobCancelledError as e:
        logging.warning("Job id=%s trace_id=%s cancelled: %s", job_id, trace_id, e)
        store.upsert_job(job_id, status="cancelled", stage="cancelled", message=str(e), progress_pct=100)
        store.append_event(f"Job cancelled: {e}", level="warning", job_id=job_id)
        try:
            client.fail(job_id, f"[trace_id={trace_id}] Cancelled by operator: {str(e)}")
        except Exception:
            logging.exception("Failed to submit cancel status for job id=%s", job_id)
    except Exception as e:
        logging.exception("Job id=%s trace_id=%s failed", job_id, trace_id)
        store.upsert_job(job_id, status="failed", stage="failed", message=str(e)[:500], progress_pct=100)
        store.append_event(f"Job failed: {e}", level="error", job_id=job_id)
        try:
            client.fail(job_id, f"[trace_id={trace_id}] {str(e)}")
        except Exception:
            logging.exception("Failed to submit fail status for job id=%s", job_id)
    finally:
        hb.stop()
        hb.join(timeout=2)
        shutil.rmtree(work_dir, ignore_errors=True)


def sync_cortex_meta(cfg: Config) -> bool:
    """Push Cortex collection metadata to the website queue server."""
    try:
        # Check Cortex API health
        api_status = "offline"
        try:
            health_resp = requests.get(f"{cfg.cortex_api_url}/health", timeout=3)
            if health_resp.status_code == 200:
                api_status = "online"
        except Exception:
            pass

        # Get collection summary from WorkingCollectionManager
        collections = {}
        capabilities = []
        if api_status == "online":
            try:
                from cortex_engine.collection_manager import WorkingCollectionManager
                mgr = WorkingCollectionManager()
                collections = mgr.get_collections_summary()
                capabilities = ["vector", "graph", "hybrid"]
            except Exception as e:
                logging.warning("Failed to read Cortex collections: %s", e)

        # Push to website
        payload = {
            "collections": collections,
            "api_status": api_status,
            "tunnel_url": cfg.cortex_tunnel_url,
            "capabilities": capabilities,
        }

        resp = requests.post(
            cfg.server_url,
            params={"action": "cortex_meta"},
            headers={"X-Queue-Key": cfg.secret_key, "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        # Retry with query key fallback on 403
        if resp.status_code == 403:
            resp = requests.post(
                cfg.server_url,
                params={"action": "cortex_meta", "key": cfg.secret_key},
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10,
            )
        resp.raise_for_status()
        logging.info("Cortex meta sync pushed: api=%s, collections=%d", api_status, len(collections))
        return True
    except Exception as e:
        logging.warning("Cortex meta sync failed: %s", e)
        return False


class CortexMetaSyncThread(threading.Thread):
    """Background thread that periodically pushes Cortex metadata to the website."""

    def __init__(self, cfg: Config, interval: int):
        super().__init__(daemon=True, name="cortex-meta-sync")
        self.cfg = cfg
        self.interval = interval
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        # Initial sync on startup
        sync_cortex_meta(self.cfg)
        while not self._stop_event.wait(self.interval):
            sync_cortex_meta(self.cfg)


def main() -> int:
    cfg = read_config()
    cfg.temp_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info(
        "Queue worker started: worker_id=%s types=%s poll=%ss",
        cfg.worker_id,
        cfg.supported_types,
        cfg.poll_interval,
    )

    # Start Cortex metadata sync background thread
    meta_sync = CortexMetaSyncThread(cfg, cfg.cortex_meta_sync_interval)
    meta_sync.start()

    client = QueueClient(cfg)
    store = QueueMonitorStore(cfg.queue_monitor_state_path)
    store.set_worker(
        status="running",
        worker_id=cfg.worker_id,
        supported_types=cfg.supported_types,
        poll_interval=cfg.poll_interval,
        server_url=cfg.server_url,
        pid=os.getpid(),
    )
    store.append_event("Queue worker started", level="info")
    stop_event = threading.Event()

    def _stop(*_args):
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    consecutive_conn_errors = 0
    last_conn_error_log_ts = 0.0

    while not stop_event.is_set():
        try:
            job = client.poll()
            if consecutive_conn_errors > 0:
                logging.info("Queue connectivity restored after %s failed attempt(s)", consecutive_conn_errors)
                store.append_event(
                    f"Queue connectivity restored after {consecutive_conn_errors} failed attempt(s)",
                    level="info",
                    source="worker.connectivity",
                )
            consecutive_conn_errors = 0

            if not job:
                store.set_worker(status="idle", worker_id=cfg.worker_id)
                time.sleep(cfg.poll_interval)
                continue
            store.set_worker(status="processing", worker_id=cfg.worker_id)
            process_job(client, cfg, store, job)
        except requests.ConnectionError as e:
            consecutive_conn_errors += 1
            backoff_seconds = min(max(cfg.poll_interval, 5) * (2 ** min(consecutive_conn_errors - 1, 5)), 300)
            now = time.time()
            # Avoid flooding logs/events during sustained outages.
            if consecutive_conn_errors == 1 or (now - last_conn_error_log_ts) >= 30:
                err_text = str(e)
                if len(err_text) > 240:
                    err_text = err_text[:240] + "..."
                logging.warning(
                    "Queue server unreachable (attempt=%s, retry_in=%ss): %s",
                    consecutive_conn_errors,
                    int(backoff_seconds),
                    err_text,
                )
                store.append_event(
                    f"Queue server unreachable (attempt={consecutive_conn_errors}, retry_in={int(backoff_seconds)}s)",
                    level="warning",
                    source="worker.connectivity",
                )
                last_conn_error_log_ts = now
            store.set_worker(status="disconnected", worker_id=cfg.worker_id)
            time.sleep(backoff_seconds)
        except requests.Timeout as e:
            consecutive_conn_errors += 1
            backoff_seconds = min(max(cfg.poll_interval, 5) * (2 ** min(consecutive_conn_errors - 1, 5)), 300)
            now = time.time()
            if consecutive_conn_errors == 1 or (now - last_conn_error_log_ts) >= 30:
                logging.warning(
                    "Queue request timeout (attempt=%s, retry_in=%ss): %s",
                    consecutive_conn_errors,
                    int(backoff_seconds),
                    e,
                )
                store.append_event(
                    f"Queue request timeout (attempt={consecutive_conn_errors}, retry_in={int(backoff_seconds)}s)",
                    level="warning",
                    source="worker.connectivity",
                )
                last_conn_error_log_ts = now
            store.set_worker(status="degraded", worker_id=cfg.worker_id)
            time.sleep(backoff_seconds)
        except requests.HTTPError as e:
            consecutive_conn_errors = 0
            logging.error("HTTP error while polling/processing: %s", e)
            store.append_event(f"HTTP error: {e}", level="error")
            store.set_worker(status="error", worker_id=cfg.worker_id)
            time.sleep(cfg.poll_interval)
        except Exception:
            consecutive_conn_errors = 0
            logging.exception("Worker loop error")
            store.append_event("Worker loop error", level="error")
            store.set_worker(status="error", worker_id=cfg.worker_id)
            time.sleep(cfg.poll_interval)

    meta_sync.stop()
    meta_sync.join(timeout=3)
    logging.info("Queue worker stopped")
    store.set_worker(status="stopped", worker_id=cfg.worker_id)
    store.append_event("Queue worker stopped", level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

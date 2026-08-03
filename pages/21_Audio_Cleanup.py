"""Audio Cleanup — SAM-Audio voice separation via the sam-audio project.

Spawns sam-audio's clean_cli.py in its OWN venv (no torch/SAM deps in the
cortex venv). Progress arrives as JSON lines on the subprocess stdout.
"""
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import streamlit as st

from cortex_engine.utils import get_logger
from cortex_engine.version_config import VERSION_STRING

PAGE_VERSION = VERSION_STRING
SAM_ROOT = Path(os.environ.get("SAM_AUDIO_ROOT", "/home/longboardfella/sam-audio"))
SAM_PYTHON = SAM_ROOT / ".venv" / "bin" / "python"
CLEAN_CLI = SAM_ROOT / "clean_cli.py"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

st.set_page_config(page_title="Audio Cleanup", layout="wide", page_icon="🎙️")
logger = get_logger(__name__)


def init_state():
    ss = st.session_state
    ss.setdefault("ac_proc", None)
    ss.setdefault("ac_events", queue.Queue())
    ss.setdefault("ac_lines", [])
    ss.setdefault("ac_pct", 0)
    ss.setdefault("ac_status", "Idle")
    ss.setdefault("ac_out_dir", None)
    ss.setdefault("ac_error", None)
    ss.setdefault("ac_running", False)


def reader_thread(proc, events):
    """Runs off the Streamlit thread: only touches the queue, never session_state."""
    try:
        for raw in iter(proc.stdout.readline, ""):
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.put(json.loads(raw))
            except json.JSONDecodeError:
                events.put({"type": "log", "message": raw})
        proc.wait()
    finally:
        events.put({"type": "exit", "rc": proc.returncode})


def start_job(upload, description, opts):
    work = Path(tempfile.mkdtemp(prefix="cortex_audio_"))
    input_path = work / upload.name
    input_path.write_bytes(upload.getvalue())
    payload = {
        "description": description or "speech",
        "convert_to_mono": True,
        "chunk_duration": int(opts["chunk_duration"]),
        "overlap": float(opts["overlap"]),
        "loudness_normalize": bool(opts["loudness"]),
        "trial_seconds": int(opts["trial_seconds"]),
        "rerank": 1,
        "predict_spans": False,
        "device": "auto",
        "memory_fraction": 0.85,
        "allow_cpu_fallback": True,
    }
    job_json = work / "job.json"
    job_json.write_text(json.dumps(payload))
    out_dir = work / "out"
    proc = subprocess.Popen(
        [str(SAM_PYTHON), str(CLEAN_CLI), "--input", str(input_path),
         "--job-json", str(job_json), "--out-dir", str(out_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(SAM_ROOT),
    )
    st.session_state.ac_proc = proc
    st.session_state.ac_out_dir = out_dir
    st.session_state.ac_running = True
    st.session_state.ac_error = None
    st.session_state.ac_pct = 0
    st.session_state.ac_lines = []
    st.session_state.ac_status = "Starting SAM-Audio job"
    threading.Thread(target=reader_thread,
                     args=(proc, st.session_state.ac_events), daemon=True).start()


def drain_events():
    q = st.session_state.ac_events
    while True:
        try:
            evt = q.get_nowait()
        except queue.Empty:
            return
        if evt.get("type") == "progress":
            st.session_state.ac_pct = max(0, min(100, int(evt.get("pct", 0))))
            msg = f"[{evt.get('stage')}] {evt.get('message')}"
            st.session_state.ac_status = msg
            st.session_state.ac_lines.append(f"{evt.get('ts')} | {evt.get('pct'):>3}% | {msg}")
            st.session_state.ac_lines = st.session_state.ac_lines[-200:]
        elif evt.get("type") == "log":
            st.session_state.ac_lines.append(str(evt.get("message")))
        elif evt.get("type") == "exit":
            st.session_state.ac_running = False
            if evt.get("rc") != 0:
                out_dir = st.session_state.ac_out_dir
                err = "clean_cli exited with an error"
                try:
                    status = json.loads((Path(out_dir) / "status.json").read_text())
                    err = status.get("error", err)
                except Exception:
                    pass
                st.session_state.ac_error = err


def main():
    init_state()
    drain_events()

    st.title("🎙️ Audio Cleanup")
    st.caption(f"SAM-Audio voice separation — {PAGE_VERSION}. "
               "Describe the sound to EXTRACT (e.g. 'a man speaking over a radio'); "
               "everything else lands in residual.wav.")

    if not SAM_PYTHON.exists():
        st.error(f"sam-audio venv not found at {SAM_PYTHON}. Set SAM_AUDIO_ROOT.")
        return

    upload = st.file_uploader(
        "Audio or video file",
        type=["wav", "mp3", "flac", "ogg", "m4a", "aac", "mp4", "mkv", "mov"])
    description = st.text_input("What to extract", value="speech")

    with st.expander("Advanced options"):
        chunk_duration = st.number_input("Chunk duration (s)", 5, 600, 60, 5)
        overlap = st.number_input("Chunk overlap (s)", 0.0, 30.0, 2.0, 0.5)
        loudness = st.checkbox("Loudness-normalize target (-16 LUFS)", value=True)
        trial_seconds = st.number_input("Trial only first N seconds (0 = full)", 0, 86400, 0, 5)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Clean Audio", type="primary",
                     disabled=st.session_state.ac_running or upload is None):
            if upload.size > MAX_UPLOAD_BYTES:
                st.error("File exceeds 2GB limit")
            else:
                start_job(upload, description,
                          {"chunk_duration": chunk_duration, "overlap": overlap,
                           "loudness": loudness, "trial_seconds": trial_seconds})
                st.rerun()
    with col2:
        if st.session_state.ac_running and st.button("Stop"):
            proc = st.session_state.ac_proc
            if proc and proc.poll() is None:
                proc.terminate()

    st.progress(st.session_state.ac_pct)
    st.caption(st.session_state.ac_status)
    if st.session_state.ac_lines:
        st.code("\n".join(st.session_state.ac_lines[-30:]), language="text")

    if st.session_state.ac_error:
        st.error(st.session_state.ac_error)

    out_dir = st.session_state.ac_out_dir
    if (not st.session_state.ac_running) and out_dir:
        zip_path = Path(out_dir) / "result.zip"
        if zip_path.exists():
            zip_bytes = zip_path.read_bytes()
            st.download_button("Download ZIP (target + residual + metadata)",
                               data=zip_bytes, file_name="audio_cleanup_result.zip",
                               mime="application/zip")
            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Cleaned voice (target)**")
                    if "target.wav" in names:
                        st.audio(zf.read("target.wav"), format="audio/wav")
                with c2:
                    st.markdown("**Removed background (residual)**")
                    if "residual.wav" in names:
                        st.audio(zf.read("residual.wav"), format="audio/wav")

    if st.session_state.ac_running:
        time.sleep(1)
        st.rerun()


main()

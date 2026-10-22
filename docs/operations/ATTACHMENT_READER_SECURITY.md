# Attachment reader: reading untrusted files safely

`cortex_engine/attachment_reader.py` (v6.7.0, 2026-10-23) turns an untrusted attachment
(photo, screenshot, PDF, Word/Excel/PowerPoint, text) into a **summary, a risk verdict and,
only when safe, sanitised text**. It serves Hermes/SlowClaw's live Gmail (paul@project-143.com) and Microsoft 365
(paul@longboardfella.com.au) lookups. The
system-side wiring (bridge, sender gate, sp4 taint guard) is in
`~/nemoclaw_ops/docs/2026-10-23-email-attachment-security.md`.

```python
from cortex_engine.attachment_reader import read_attachment
r = read_attachment(path, filename, mime, allow_raw=True)   # allow_raw=False for unknown senders
r["risk"]["level"]   # low | medium | high | blocked
```
CLI: `venv/bin/python -m cortex_engine.attachment_reader <file> --name F [--no-raw]` prints JSON.

## Pipeline

1. **ClamAV.** `clamdscan --stream` runs before anything else. A signature hit blocks the file.
   A down daemon adds the flag `virus scan unavailable` (risk at least medium); it is never
   treated as clean. The daemon is `clamav-daemon` (about 1 GB RAM); signatures come from
   `clamav-freshclam`.
2. **Sandbox (`--sandbox` half).**
   - Runs under `prlimit` (3 GB address space, 240 s CPU, 500 MB file size), `nice 15`,
     `ionice -c3`, and `bwrap --unshare-all --clearenv`.
   - Read-only mounts: `/usr`, the Python install, this venv and the reader file.
   - Writable: only `/out` plus a tmpfs.
   - No network and no `$HOME`.
   - Inside it does:
     - **Triage:** libmagic real type vs an extension allowlist; double extensions; HTML/SVG;
       macro types and `vbaProject.bin`; DDE; legacy OLE (an oletools macro scan is reported,
       and the file is refused); OOXML zip bombs; encrypted PDFs; embedded files; PDF active
       content in the decompressed objects (`/JavaScript /JS /Launch /EmbeddedFile /RichMedia
       /XFA /SubmitForm /ImportData /GoToR /GoToE` are blocked; `/OpenAction /AA` are flagged).
     - **Extraction of VISIBLE text only.** PDF white, tiny or off-page spans, Word
       hidden/white/tiny runs and hidden Excel sheets are counted and flagged, not returned.
     - **Image rendering** to raw RGB for images and for scanned PDF pages (under 40 visible
       characters).
   - Audio, video and archives are refused outright, by extension and by real type (Paul's
     rule: media files can carry exploits; archives hide files from scanners).
3. **Host half.**
   - Rebuilds images from raw pixels (no image parser touches attacker bytes) and sends them
     to the loaded LM Studio Qwen VLM.
   - Strips zero-width and bidi characters and defangs URLs (`hxxps://x[.]com`).
   - **Decodes hidden instructions** before judging: invisible Unicode tag characters (ASCII
     smuggling), variation-selector payloads, Morse code, base64, hex, letter-spaced text,
     Cyrillic/Greek homoglyphs mixed into Latin words, ROT13, reversed text and leetspeak. Any
     finding is a risk flag. The decoded payload goes through the heuristics and into the judge,
     and raw text is then never returned.
   - Runs regex injection heuristics: strong patterns mean injection; credential and shell
     words are a flag only.
   - **Other languages:** the judge is told to translate any language (and decode any
     encoding) before deciding. Non-Latin scripts are reported in `notes`, not as a risk on
     their own.
   - Makes one **tool-less** Qwen call with nonce-delimited content. It returns a fixed JSON
     shape: summary, key_points, asks_reader_to, injection_attempt, phishing_signs, reason.
   - A summary that repeats instruction-like text is withheld.
   - Raw text is returned only when nothing is suspected and `allow_raw` is set.
   - `warning_for_paul` is a ready-made one-line warning (empty when the risk is low). Callers
     recompute it when they raise the risk level, e.g. for an untrusted sender.

LM Studio endpoint: `CORTEX_VLM_URL` (default `http://100.97.82.98:1234`); model `CORTEX_VLM_MODEL`.
As everywhere, `reasoning_effort: "none"` is mandatory for qwen3.6.

## Tests

```bash
venv/bin/python -m pytest tests/security -q      # 30 tests; model/clamd cases skip if those are down
venv/bin/python tests/security/make_attachment_corpus.py /tmp/corpus   # build the files to inspect
```
The corpus is harmless: EICAR test strings, fake audio/video/zip files, French, Morse, base64,
homoglyph, Unicode-tag, ROT13 and leetspeak injections, a benign German letter, a PDF with a JS alert, a fake macro blob, a DDE
field, white hidden text, an injection image and text, and a fake bank-phishing PDF.

## Limits

- QR codes are only mentioned by the VLM, not decoded (no zbar).
- `.doc`/`.xls`/`.ppt` are refused. LibreOffice conversion was removed until it can run inside
  the sandbox.
- Injection detection is best-effort. The hard guarantee is the caller's capability limit
  (the sp4 `untrusted-content-guard`), not this module.

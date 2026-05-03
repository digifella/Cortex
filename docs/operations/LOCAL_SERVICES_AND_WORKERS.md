# Local services and workers

Reference for the long-running processes that make up the Cortex / Longboardfella
queue + mailbox pipeline. Updated after the v4.6.90 textify-handler work, which
exposed a real-world failure mode where two copies of `worker.py` (one manual,
one systemd) raced for the same job.

---

## Overview

Three classes of process keep the system running:

1. **systemd user services** on the WSL host (`Fastfella`) — auto-start on
   reboot, restart on crash. Pollers, queue workers, and the API.
2. **cPanel cron jobs** on the production webserver (`longboardfella.com.au`) —
   periodic batch tasks: queue notifications, scheduled reports, etc.
3. **Manually started processes** in WSL terminal sessions — the cortex
   Streamlit UI, ad-hoc scripts, debugging.

The first two are managed; the third is for development only. **Don't manually
start a worker that already runs as a systemd service** — both copies will poll
the same queue / mailbox and race for jobs. That's exactly how stale-code
processed job 1599 on 2026-05-03.

---

## systemd user services

All units live in `~/.config/systemd/user/`. Backups checked into
`ops/systemd/` in this repo for recovery.

| Service | Source | What it does | Auto-start |
|---|---|---|---|
| `cortex-queue-worker.service` | `cortex_suite/worker/worker.py` | Polls the production queue API every ~15 s, claims and processes any job whose type appears in `SUPPORTED_TYPES` (pdf_textify, url_ingest, youtube_summarise, …). | yes |
| `cortex-uvicorn.service` | `cortex_suite/api/main.py` | Cortex Suite HTTP API on `0.0.0.0:8000`. | yes |
| `cortex-intel-mailbox.service` | `cortex_suite/worker/intel_mailbox_worker.py` | Polls the `intel@` mailbox (Market Radar intel notes). Continuous loop with internal poll interval. | yes |
| `cortex-notes-mailbox.service` | `cortex_suite/worker/notes_mailbox_worker.py` | Polls the `notes@` mailbox (note ingestion to AI-Vault). Continuous loop. | yes |
| `nemoclaw-lab-mailbox-worker.service` (+ `.timer`) | `cortex_suite/worker/lab_mailbox_worker.py --once` | Triggered by the timer **every 2 minutes**. Polls `lab@` mailbox, fetches attachments via Graph (when `LAB_FETCH_ATTACHMENTS=1`), and posts each email to the website's job webhook. | yes (timer) |
| `lbf-market-radar-worker.service` | `longboardfella_website/worker/worker.py` | Market Radar / signal_episode / portal_classify worker (separate `SUPPORTED_TYPES` from cortex-queue-worker — they don't overlap). | **disabled** by default |

`nemoclaw-telegram-bridge.service`, `nemoclaw-vault-query.service`,
`nemoclaw-voice.service`, `message-agent.service` also exist but are out of
scope for this doc.

### Common operations

```bash
# Status of every service in this group
systemctl --user --no-pager list-units 'cortex-*' 'nemoclaw-*' 'lbf-*'

# Status of one
systemctl --user status cortex-queue-worker --no-pager

# Tail logs of one (Ctrl+C to exit; doesn't affect the service)
journalctl --user -u cortex-queue-worker -f

# Restart after pulling new code
systemctl --user restart cortex-queue-worker

# Disable + stop (won't auto-start on reboot, won't run now)
systemctl --user disable --now cortex-uvicorn

# Enable + start (e.g. switching market_radar on)
systemctl --user enable --now lbf-market-radar-worker

# Reload the unit definitions after editing a .service file
systemctl --user daemon-reload
```

### After editing code in `cortex_suite/worker/`

Restart the relevant service so the new code is loaded into memory:

```bash
systemctl --user restart cortex-queue-worker      # for handler / worker.py changes
systemctl --user restart cortex-intel-mailbox     # for intel_mailbox_worker.py changes
systemctl --user restart cortex-notes-mailbox     # for notes_mailbox_worker.py changes
systemctl --user restart cortex-uvicorn           # for api/* changes
```

`nemoclaw-lab-mailbox-worker` is timer-driven — the next 2-minute fire picks
up the new code automatically. No restart needed.

---

## cPanel cron jobs (production webserver)

Live in cPanel → Cron Jobs. These run on the shared host, not on WSL.

| Schedule | Command | Purpose |
|---|---|---|
| `*/5 * * * *` | `php /home/longboar/public_html/admin/queue_notify.php` | **Delivers job-completion emails**, including the textify completion email to lab members. Reads `email_send_log` to know what to retry. |
| `*/5 * * * *` | `php /home/longboar/public_html/job_ingest.php` | Legacy IMAP poll (Graph poller is now the primary path via `lab_mailbox_worker`). |
| `*/5 * * * *` | `php /home/longboar/public_html/rip_strip_worker.php` | RIP / STRIP newsletter conversion. |
| `*/5 * * * *` | `php /home/longboar/public_html/market_radar_intel_ingest.php` | Market Radar intel ingestion. |
| `*/5 * * * *` | `php /home/longboar/public_html/subscribe/send_worker.php` | Newsletter / dispatch sender. |
| `*/15 * * * *` | `php /home/longboar/public_html/nova_kb_worker.php` | Nova knowledge-base worker. |
| `0 2 * * *` | `php /home/longboar/public_html/portal_intelligence_cron.php` | Daily portal intelligence run (2 am). |
| `0 7 * * *` | `php /home/longboar/public_html/page_insights_cron.php wiki` | Daily wiki page-insights (7 am). |
| `0 7 * * *` | `php /home/longboar/public_html/market_radar_watch_cron.php` | Daily market-radar watch (7 am). |
| `0 7 * * *` | `php /home/longboar/public_html/market_radar_intel_digest_cron.php` | Daily market-radar digest (7 am). |
| `0 7 * * *` | `php /home/longboar/public_html/nova_digest_cron.php` | Daily Nova digest (7 am). |
| `0 8 * * 1` | `php /home/longboar/public_html/market_radar_sector_report_cron.php` | Weekly sector report (Mondays 8 am). |

These do not need editing in normal operation. They are listed here so a
post-reboot or post-deploy verification has a single place to check.

---

## Critical: do not double-run

Each managed worker should have **exactly one** instance polling its source
(queue API or mailbox) at any time. Two instances cause:

- Race conditions where a slower worker claims a job a fresher worker would
  have processed correctly. (Real incident: 2026-05-03, job 1599 ran on
  pre-v4.6.90 handler code and produced the wrong file extension.)
- Duplicate Graph token requests / mailbox state contention.

If you *want* a manual worker for terminal-style visibility, **first** disable
the systemd one:

```bash
systemctl --user disable --now cortex-queue-worker
# now your manual `python worker/worker.py` is the only claimant
```

…and re-enable when you're done:

```bash
systemctl --user enable --now cortex-queue-worker
```

For routine debugging, prefer `journalctl --user -u <service> -f` over killing
the service.

---

## Cold-boot recovery

If the host loses systemd state (e.g. WSL distro reset), the unit files have
been backed up at `ops/systemd/` in this repo. Restore with:

```bash
mkdir -p ~/.config/systemd/user
cp ops/systemd/*.service ops/systemd/*.timer ~/.config/systemd/user/ 2>/dev/null
systemctl --user daemon-reload
systemctl --user enable --now \
    cortex-queue-worker \
    cortex-uvicorn \
    cortex-intel-mailbox \
    cortex-notes-mailbox \
    nemoclaw-lab-mailbox-worker.timer
# Optionally:
# systemctl --user enable --now lbf-market-radar-worker
```

---

## What's *not* a service

- **The Cortex Streamlit UI** (`Cortex_Suite.py`) is launched manually with
  `streamlit run Cortex_Suite.py` when you want to use the desktop UI. It's
  not a daemon and isn't exposed to anyone else, so there's no point
  systemd-managing it.
- **One-off scripts** (`scripts/version_manager.py`, etc.) — invoked on demand.
- **Docker containers** (NemoClaw sandboxes) — managed by docker, not systemd.

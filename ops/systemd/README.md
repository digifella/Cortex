# systemd unit backups

Snapshot copies of the systemd user-unit files for the Cortex / Lab pipeline.
Operational doc: [`docs/operations/LOCAL_SERVICES_AND_WORKERS.md`](../../docs/operations/LOCAL_SERVICES_AND_WORKERS.md).

To restore on a fresh host:

```bash
mkdir -p ~/.config/systemd/user
cp *.service *.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now \
    cortex-queue-worker \
    cortex-uvicorn \
    cortex-intel-mailbox \
    cortex-notes-mailbox \
    nemoclaw-lab-mailbox-worker.timer
# Optional (was disabled on this host because market_radar isn't actively used):
# systemctl --user enable --now lbf-market-radar-worker
```

Source paths and `User=` are hard-coded for the `longboardfella` WSL user. If
the username or repo paths change, edit the units before installing.

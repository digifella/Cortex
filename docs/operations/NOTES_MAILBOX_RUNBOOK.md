# Notes Mailbox Runbook

Purpose: operate the `notes@longboardfella.com.au` mailbox flow that captures email notes into the public or private Obsidian vaults.

## Scope

This runbook covers the Cortex-side `notes@` worker only.

It does not cover:

- Market Radar intel routing through `intel@longboardfella.com.au`
- Lab job routing through `lab@longboardfella.com.au`
- website webhook ingestion

## Mailbox Contract

Use `notes@longboardfella.com.au` for note capture.

Routing rules:

- Plain subject: public note.
- `NOTE:` or `MEMO:` subject: public note.
- Subject containing `PRIVATE`, `SENSITIVE`, or `CONFIDENTIAL`: private note.
- `PRIVATE:` or `VAULT:` subject: private note.
- `INTEL:` or `entity:` style subject: rejected as Market Radar-shaped content.

Use `intel@longboardfella.com.au` instead for Market Radar or stakeholder intelligence.

## Destinations

Public notes:

- Windows: `C:\Users\paul\Documents\AI-Vault\Inbox`
- WSL: `/mnt/c/Users/paul/Documents/AI-Vault/Inbox`

Private notes:

- Windows: `C:\Users\paul\OneDrive - VentraIP Australia\Vault_OneDrive\notes`
- WSL: `/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/notes`

Audit/debug JSON:

- Public: `~/cortex_suite/tmp/notes_public_outbox`
- Private: `~/cortex_suite/tmp/notes_private_outbox`

Local processed-message state:

- `~/cortex_suite/tmp/notes_mailbox_state.json`

## Configuration

Configuration lives in `worker/config.env`.

Required keys:

```env
NOTES_TRANSPORT_MODE=graph
NOTES_MAILBOX_IDENTITY=notes@longboardfella.com.au
NOTES_GRAPH_MAILBOX=notes@longboardfella.com.au
NOTES_WRITE_VAULT_MARKDOWN=1
NOTES_PUBLIC_VAULT_DIR=/mnt/c/Users/paul/Documents/AI-Vault/Inbox
NOTES_PRIVATE_VAULT_DIR=/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/notes
NOTES_GRAPH_TENANT_ID=...
NOTES_GRAPH_CLIENT_ID=...
NOTES_GRAPH_CLIENT_SECRET=...
```

Do not commit `worker/config.env`; it contains live secrets.

## Microsoft Graph Permissions

The current design uses Microsoft Graph app-only polling.

Expected permission:

- `Mail.Read` application permission

Required operational guardrail:

- Scope the app in Exchange Online so it can access only `notes@longboardfella.com.au`.

Do not grant `Mail.ReadWrite` unless there is a concrete operational need to mutate the mailbox.

With only `Mail.Read`, Microsoft Graph can reject mark-as-read calls with `403 Forbidden`. This is expected. The worker records processed Graph IDs locally, so repeated unread messages do not create duplicate vault notes.

## Run Manually

```bash
cd ~/cortex_suite
source venv/bin/activate
python worker/notes_mailbox_worker.py
```

Expected startup:

```text
Notes mailbox worker started: transport=graph mailbox=notes@longboardfella.com.au ...
Notes mailbox transport mode 'graph' selected.
```

Expected public-note processing:

```text
Notes mailbox processed subject=TEST PUBLIC NOTE route=public_stash path=... vault_path=/mnt/c/Users/paul/Documents/AI-Vault/Inbox/...
```

Expected private-note processing:

```text
Notes mailbox processed subject=PRIVATE: test vault note route=private_vault path=... vault_path=/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/notes/...
```

Expected least-privilege warning:

```text
Notes mailbox left message unread in Graph ... local dedupe recorded it. This is expected with least-privilege Mail.Read.
```

Treat that warning as acceptable when the same log cycle includes a successful `processed` line and a `vault_path`.

## Validation

Public test:

1. Send an email to `notes@longboardfella.com.au`.
2. Use a normal subject, for example `TEST PUBLIC NOTE`.
3. Confirm a markdown file appears in `C:\Users\paul\Documents\AI-Vault\Inbox`.
4. Confirm a JSON audit file appears in `~/cortex_suite/tmp/notes_public_outbox`.

Private test:

1. Send an email to `notes@longboardfella.com.au`.
2. Use a subject such as `PRIVATE: test vault note`.
3. Confirm a markdown file appears in `C:\Users\paul\OneDrive - VentraIP Australia\Vault_OneDrive\notes`.
4. Confirm a JSON audit file appears in `~/cortex_suite/tmp/notes_private_outbox`.

Duplicate-suppression test:

1. Leave the worker running for another polling cycle.
2. Confirm the same email does not create another markdown file.
3. Confirm its Graph ID is present in `~/cortex_suite/tmp/notes_mailbox_state.json`.

Unit test:

```bash
pytest -q tests/unit/test_notes_mailbox.py
```

## Troubleshooting

`Microsoft Graph token request failed (400)`:

- Check `NOTES_GRAPH_TENANT_ID`.
- A single transposed character will produce tenant-not-found or invalid-tenant errors.

`invalid_client`:

- Check `NOTES_GRAPH_CLIENT_SECRET`.
- Use the secret value, not the secret ID.

`403 Forbidden` while marking read:

- Expected with `Mail.Read`.
- Confirm the worker logs `local dedupe recorded it`.
- Do not grant `Mail.ReadWrite` unless duplicate suppression fails or server-side read-state becomes required.

No file appears in the public/private vault:

- Confirm `NOTES_WRITE_VAULT_MARKDOWN=1`.
- Confirm the WSL path exists.
- Confirm the worker log includes `vault_path=...`.
- Confirm the message was not rejected as `unsupported_market_intel`.

Repeated duplicate files:

- Confirm `~/cortex_suite/tmp/notes_mailbox_state.json` exists.
- Confirm the worker process was restarted after the local-dedupe implementation was deployed.
- Confirm each repeated log has the same Graph message ID; if Graph is returning different IDs for the same email, investigate Microsoft-side duplication.

## Security Notes

- `worker/config.env` is intentionally local-only and must not be committed.
- `NOTES_GRAPH_CLIENT_SECRET` is a live credential and should be rotated if exposed.
- The Graph app should stay scoped to the `notes@` mailbox.
- Private notes are written only to the private OneDrive-backed vault path and are marked `wiki-ready: false`.

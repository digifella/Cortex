# Notes Mailbox Runbook

Purpose: operate the `notes@longboardfella.com.au` mailbox flow that captures email notes into the private Obsidian vault.

## Scope

This runbook covers the Cortex-side `notes@` worker only.

It does not cover:

- Public candidate intake through `intel@longboardfella.com.au`
- Lab job routing through `lab@longboardfella.com.au`
- website webhook ingestion

## Mailbox Contract

Use `notes@longboardfella.com.au` for note capture.

Routing rules:

- Plain subject, `NOTE:`, or `MEMO:` subject: private note.
- Subject containing `PRIVATE`, `SENSITIVE`, or `CONFIDENTIAL`: private note.
- `PRIVATE:` or `VAULT:` subject: private note.
- `PUBLIC:` subject: private review candidate, never direct public publication.
- Legacy `entity:` or Market Radar-shaped content: rejected for manual routing.

Use `intel@longboardfella.com.au` for public knowledge candidates.

## Destinations

Public review candidates:

- Windows: `C:\Users\paul\OneDrive - VentraIP Australia\Vault_OneDrive\Public_Review\notes`
- WSL: `/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/Public_Review/notes`

Private notes:

- Windows: `C:\Users\paul\OneDrive - VentraIP Australia\Vault_OneDrive\notes`
- WSL: `/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/notes`

Audit/debug JSON: `~/vault-rag-db/notes-mailbox/private`.

Local processed-message state: `~/vault-rag-db/notes-mailbox/notes_mailbox_state.json`.

File attachments are preserved under `~/vault-rag-db/notes-mailbox/attachments`
and linked from the private note. Unsupported or oversized attachments leave
the message unprocessed and unread for manual handling; they are never silently
dropped. Attachment text extraction and indexing still need a separate check.

## Configuration

Configuration lives in `worker/config.env`.

Required keys:

```env
NOTES_TRANSPORT_MODE=graph
NOTES_MAILBOX_IDENTITY=notes@longboardfella.com.au
NOTES_GRAPH_MAILBOX=notes@longboardfella.com.au
NOTES_WRITE_VAULT_MARKDOWN=1
NOTES_PRIVATE_VAULT_DIR=/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/notes
NOTES_PUBLIC_REVIEW_VAULT_DIR=/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/Public_Review/notes
NOTES_GRAPH_TENANT_ID=...
NOTES_GRAPH_CLIENT_ID=...
NOTES_GRAPH_CLIENT_SECRET=...
```

Do not commit `worker/config.env`; it contains live secrets.

## Microsoft Graph Permissions

The current design uses Microsoft Graph app-only polling.

Expected permission:

- Exchange Application RBAC `Application Mail.Read` scoped to a group containing
  only `notes@`; remove any tenant-wide Entra Graph `Mail.Read` application role
  after verifying the scoped assignment.

Required operational guardrail:

- Verify a fresh live Graph probe gives notes@ 200 and lab@/intel@ 403 or 404
  before enabling the worker.

Do not grant `Mail.ReadWrite` unless there is a concrete operational need to mutate the mailbox.

With only `Mail.Read`, Microsoft Graph can reject mark-as-read calls with `403 Forbidden`. This is expected. The worker records processed Graph IDs locally and pages the unread inbox, so repeated unread messages do not create duplicate vault notes or hide later pages.
The restored worker starts at the 15 October 17:15 AEDT cutover and advances a
received-time cursor only after each message is filed or deliberately rejected.
Older mail requires a separate replay review to avoid duplicating pre-crash
notes. Each new note records `source_date`, its inference source and precision;
the email receipt date is the fallback.

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

Expected private-review processing for a `PUBLIC:` subject:

```text
Notes mailbox processed subject=PUBLIC: TEST NOTE route=public_review path=... vault_path=/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/Public_Review/notes/...
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

Private review test:

1. Send an email to `notes@longboardfella.com.au`.
2. Use a subject such as `PUBLIC: TEST NOTE`.
3. Confirm a markdown file appears in `C:\Users\paul\OneDrive - VentraIP Australia\Vault_OneDrive\Public_Review\notes` with `wiki-ready: false`.
4. Confirm no public AI-Vault note appears.

Private test:

1. Send an email to `notes@longboardfella.com.au`.
2. Use a subject such as `PRIVATE: test vault note`.
3. Confirm a markdown file appears in `C:\Users\paul\OneDrive - VentraIP Australia\Vault_OneDrive\notes`.
4. Confirm a JSON audit file appears in `~/vault-rag-db/notes-mailbox/private`.

Duplicate-suppression test:

1. Leave the worker running for another polling cycle.
2. Confirm the same email does not create another markdown file.
3. Confirm its Graph ID is present in `~/vault-rag-db/notes-mailbox/notes_mailbox_state.json`.

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

No file appears in the private vault or review folder:

- Confirm `NOTES_WRITE_VAULT_MARKDOWN=1`.
- Confirm the WSL path exists.
- Confirm the worker log includes `vault_path=...`.
- Confirm the message was not rejected as `unsupported_market_intel`.

Repeated duplicate files:

- Confirm `~/vault-rag-db/notes-mailbox/notes_mailbox_state.json` exists.
- Confirm the worker process was restarted after the local-dedupe implementation was deployed.
- Confirm each repeated log has the same Graph message ID; if Graph is returning different IDs for the same email, investigate Microsoft-side duplication.

## Security Notes

- `worker/config.env` is intentionally local-only and must not be committed.
- `NOTES_GRAPH_CLIENT_SECRET` is a live credential and should be rotated if exposed.
- The Graph app should stay scoped to the `notes@` mailbox.
- Private notes are written only to the private OneDrive-backed vault path and are marked `wiki-ready: false`.

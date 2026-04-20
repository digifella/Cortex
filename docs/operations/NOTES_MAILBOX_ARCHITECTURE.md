# Notes Mailbox Architecture

Purpose: define the separate Cortex-owned mailbox flow for `notes@longboardfella.com.au` so note capture is no longer mixed into the Market Radar `intel@` mailbox.

## Decision

Split the current mixed mailbox responsibilities into two explicit channels:

- `intel@longboardfella.com.au`
  - Market Radar / website intel only
  - stakeholder intel
  - org charts / strategy docs
  - CSV profile imports
  - website webhook delivery

- `notes@longboardfella.com.au`
  - Cortex note capture only
  - default public stash destination
  - private vault routing when subject contains `PRIVATE`

This replaces the old mixed behavior where the Gmail-backed intel mailbox was being used for both website intel ingestion and Nemoclaw/Cortex note stash behavior.

## Current Implementation Status

Implemented and live-validated on 2026-04-20:

- `notes@longboardfella.com.au` is read through Microsoft Graph app-only polling.
- The Graph app is mailbox-scoped in Exchange Online to `notes@longboardfella.com.au`.
- The app uses least-privilege `Mail.Read`; it does not require `Mail.ReadWrite`.
- Public notes are written as Obsidian markdown to:
  - Windows: `C:\Users\paul\Documents\AI-Vault\Inbox`
  - WSL: `/mnt/c/Users/paul/Documents/AI-Vault/Inbox`
- Private notes are written as Obsidian markdown to:
  - Windows: `C:\Users\paul\OneDrive - VentraIP Australia\Vault_OneDrive\notes`
  - WSL: `/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/notes`
- JSON audit copies are retained under the Cortex repo:
  - `tmp/notes_public_outbox`
  - `tmp/notes_private_outbox`
- Processed Graph message IDs are tracked locally in:
  - `tmp/notes_mailbox_state.json`

Live validation completed:

- public test note routed to the public Obsidian vault inbox
- `PRIVATE:` test note routed to the private Obsidian vault notes folder
- repeated unread Graph messages are suppressed locally by state tracking

## Why This Split Exists

The old `intel` mailbox had two conflicting jobs:

1. send structured intelligence into Market Radar
2. act as a personal note/stash mailbox for Cortex/Nemoclaw workflows

That created ambiguous operator behavior and brittle special cases.

Examples of unwanted coupling:

- a note intended for stash could be interpreted as Market Radar intel
- `private:` / `vault:` subjects were suppressing Market Radar processing inside the same mailbox flow
- website webhook delivery and note stash behavior were effectively competing routes

## Mailbox Roles

### `intel@longboardfella.com.au`

Canonical role:

- website-facing Market Radar intake mailbox

Allowed content:

- stakeholder updates
- organisation notes intended for Market Radar
- strategy / annual report / org chart attachments
- CSV profile imports
- explicit `INTEL:` submissions

Not allowed:

- general personal notes
- stash-to-vault messages
- memo capture unrelated to Market Radar

Delivery behavior:

- note/document/intel payloads post to the website intel webhook
- no Nemoclaw stash routing
- no Telegram private-note behavior

### `notes@longboardfella.com.au`

Canonical role:

- Cortex-owned note capture mailbox

Allowed content:

- quick notes
- meeting notes
- voice-dictated notes
- memo capture
- personal/public stash material
- private vault notes

Not allowed:

- Market Radar intel
- website subscriber-org intel deliveries
- CSV profile imports

Delivery behavior:

- default destination: public stash
- if subject contains `PRIVATE`, `SENSITIVE`, or `CONFIDENTIAL`: route to private vault
- optional future explicit subject aliases:
  - `NOTE:`
  - `MEMO:`
  - `VAULT:`

## Routing Rules For `notes@`

### Default route

If mail is sent to `notes@longboardfella.com.au` and no privacy marker is present:

- classify as `public_stash`
- write markdown into `AI-Vault/Inbox`
- write a JSON audit copy into `tmp/notes_public_outbox`
- mark the Graph message as processed in `tmp/notes_mailbox_state.json`

### Private route

If subject contains any of:

- `PRIVATE`
- `SENSITIVE`
- `CONFIDENTIAL`

or explicitly starts with:

- `VAULT:`
- `PRIVATE:`

then:

- classify as `private_vault`
- write markdown into `Vault_OneDrive/notes`
- write a JSON audit copy into `tmp/notes_private_outbox`
- mark the Graph message as processed in `tmp/notes_mailbox_state.json`
- do not forward to Market Radar
- do not send to public stash

### Unsupported route

If a message to `notes@...` clearly looks like Market Radar intel:

- reject or quarantine with a clear operator-visible reason
- advise sender to use `intel@longboardfella.com.au`

Examples:

- subject starts with `INTEL:`
- sender includes Market Radar org-scoping directives such as `entity:`
- attached CSV profile import intended for watchlist ingestion

## Transport

The implemented transport is Microsoft Graph polling with app-only client credentials.

This was chosen over IMAP because:

- `notes@...` is an M365 shared mailbox
- IMAP/basic-auth would recreate the Gmail workaround pattern
- Microsoft Graph is the usual Microsoft-supported API path
- the app can be scoped to one mailbox with an Exchange Online application access policy

The worker intentionally runs with `Mail.Read`, not `Mail.ReadWrite`.

Consequence:

- Graph can return `403 Forbidden` when the worker attempts to mark a message as read.
- This is expected under least privilege.
- The worker logs this as a warning and relies on `tmp/notes_mailbox_state.json` to prevent local duplicate processing.

Only grant `Mail.ReadWrite` if server-side mark-as-read becomes operationally necessary.

## Cortex Implementation Requirements

The `notes@` mailbox should not be bolted onto the existing `intel_mailbox_worker.py` as-is.

Implemented code structure:

- `cortex_engine/notes_mailbox.py`
  - owns public/private note route classification
- `worker/notes_mailbox_worker.py`
  - reads `worker/config.env`
  - polls Microsoft Graph when `NOTES_TRANSPORT_MODE=graph`
  - normalizes Graph messages
  - writes JSON audit files
  - writes markdown files into the configured public/private Obsidian vault folders
  - tracks processed Graph IDs locally
- `cortex_engine/intel_mailbox.py`
  - remains the Market Radar / website-intel mailbox path
  - suppresses explicit note-shaped subjects so notes do not leak into Market Radar

Configuration keys:

- `NOTES_TRANSPORT_MODE=graph`
- `NOTES_MAILBOX_IDENTITY=notes@longboardfella.com.au`
- `NOTES_GRAPH_TENANT_ID`
- `NOTES_GRAPH_CLIENT_ID`
- `NOTES_GRAPH_CLIENT_SECRET`
- `NOTES_GRAPH_MAILBOX=notes@longboardfella.com.au`
- `NOTES_WRITE_VAULT_MARKDOWN=1`
- `NOTES_PUBLIC_VAULT_DIR=/mnt/c/Users/paul/Documents/AI-Vault/Inbox`
- `NOTES_PRIVATE_VAULT_DIR=/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/notes`

Do not commit `worker/config.env`; it contains live secrets.

## Operator Guidance

Use:

- `intel@longboardfella.com.au` for Market Radar / stakeholder intelligence
- `notes@longboardfella.com.au` for Cortex notes

Subject guidance for `notes@`:

- plain subject = public stash
- `PRIVATE:` prefix = private vault

Run the worker manually:

```bash
cd ~/cortex_suite
source venv/bin/activate
python worker/notes_mailbox_worker.py
```

Expected successful log line:

```text
Notes mailbox processed subject=... route=public_stash path=... vault_path=...
```

or:

```text
Notes mailbox processed subject=PRIVATE: ... route=private_vault path=... vault_path=...
```

Expected least-privilege warning:

```text
Notes mailbox left message unread in Graph ... local dedupe recorded it. This is expected with least-privilege Mail.Read.
```

This warning is acceptable if the message was processed and a `vault_path` was logged.

## Definition Of Done

This architecture change is complete when:

- `intel@...` no longer triggers note-stash behavior: done
- `notes@...` is handled by a dedicated Cortex note flow: done
- private-note routing is mailbox-specific rather than piggybacked on the intel mailbox: done
- operator docs clearly separate `intel@` and `notes@`: done
- legacy forwarding rules into `intel.longboardfella@gmail.com` are no longer needed for notes: done
- public and private live tests write to the expected vault folders: done

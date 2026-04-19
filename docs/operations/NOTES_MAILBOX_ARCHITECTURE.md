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

- classify as `public_note`
- send to Cortex public stash
- send any secondary notification only if the note workflow already expects one

### Private route

If subject contains any of:

- `PRIVATE`
- `SENSITIVE`
- `CONFIDENTIAL`

or explicitly starts with:

- `VAULT:`
- `PRIVATE:`

then:

- classify as `private_note`
- route to private vault only
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

## Transport Options

### Short-term

Do not try to repoint the current Gmail/basic-auth intel mailbox worker directly at `notes@...`.

Reason:

- `notes@...` is an M365 shared mailbox
- the current worker transport is still configured for the legacy Gmail IMAP/SMTP path
- a direct swap would be a transport rewrite disguised as a config change

### Recommended long-term

Implement a dedicated `notes` transport with one of these options:

1. Microsoft Graph mailbox polling
2. Microsoft Graph subscription / webhook push
3. a small bridge process that normalizes M365 mail into the existing Cortex mailbox payload shape

Preferred direction:

- Microsoft Graph subscription or webhook push

Reason:

- avoids IMAP/basic-auth assumptions
- aligns with the website’s new transport-neutral webhook approach
- keeps `notes@` as a first-class M365 mailbox rather than a compatibility hack

## Cortex Implementation Requirements

The `notes@` mailbox should not be bolted onto the existing `intel_mailbox_worker.py` as-is.

Required implementation direction:

1. extract the stash/public-note/private-vault routing into a dedicated note mailbox processor
2. keep Market Radar delivery logic in the intel mailbox processor only
3. add mailbox-role-aware routing so `intel` and `notes` cannot silently overlap again

Recommended code structure:

- `cortex_engine/intel_mailbox.py`
  - remains website / Market Radar mailbox logic
- new module, recommended:
  - `cortex_engine/notes_mailbox.py`
  - owns public/private note routing
- optional worker entrypoint:
  - `worker/notes_mailbox_worker.py`

## Operator Guidance

Use:

- `intel@longboardfella.com.au` for Market Radar / stakeholder intelligence
- `notes@longboardfella.com.au` for Cortex notes

Subject guidance for `notes@`:

- plain subject = public stash
- `PRIVATE:` prefix = private vault

## Definition Of Done

This architecture change is complete when:

- `intel@...` no longer triggers note-stash behavior
- `notes@...` is handled by a dedicated Cortex note flow
- private-note routing is mailbox-specific rather than piggybacked on the intel mailbox
- operator docs clearly separate `intel@` and `notes@`
- legacy forwarding rules into `intel.longboardfella@gmail.com` are no longer needed for notes

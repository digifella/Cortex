# Hermes secure document delivery

**Status:** commissioned and operating as of 2026-08-05.

**Purpose:** let the Hermes agent on the Surface Pro 4 email an explicitly
requested document to Paul without giving the model a general-purpose mailbox,
arbitrary-recipient capability, Microsoft 365 credentials, or unattended send
permission.

## Architecture

```text
Paul in private Discord channel
  -> Hermes on SP4 obtains /tmp/kb-* document
  -> email-delivery-approval plugin requests one human approval
  -> kb-email-me validates and submits over Tailscale with bearer token
  -> hermes-delivery.service on Fastfella validates again and rate-limits
  -> Microsoft Graph client credentials
  -> Exchange Application RBAC permits Mail.Send only as hermes-delivery@
  -> fixed destination: paul@
```

The SP4 never receives the Entra client secret. Fastfella never accepts a
recipient from the request. Microsoft 365 independently restricts the
application to the dedicated shared mailbox, providing a second boundary if
the broker were misconfigured.

## Decision

Do not use `lab@longboardfella.com.au` or `intel@longboardfella.com.au` for
Hermes delivery. They are active production-ingest identities. Create a separate
shared mailbox, `hermes-delivery@longboardfella.com.au`, and a separate Entra
service principal with only the Exchange `Application Mail.Send` role, scoped to
that mailbox.

Do not add Microsoft Graph `Mail.Send` under **Entra API permissions**. An Entra
application permission is tenant-wide and is additive with Exchange Application
RBAC. Grant the scoped role only in Exchange Online.

`hermes-delivery@` is a sending identity, not a mailbox Paul needs to monitor.
It does not need to be added to desktop Outlook and Paul does not need Full
Access or Send As delegation for this workflow.

## Security properties and limits

- Discord access is limited to Paul's Discord user ID.
- Email is permitted only in the private knowledge-base channel and only after
  an explicit user request.
- Every send receives a tool-call-specific approval key; a previous approval
  cannot authorize a future message.
- The recipient lives only in Fastfella's protected `worker/config.env` and is
  absent from the request schema.
- `to`, `cc`, `bcc`, `recipient`, and `reply_to` request keys are rejected.
- SP4 accepts only regular, non-symlink files directly under `/tmp` whose names
  begin `kb-`.
- Allowed extensions are CSV, DOCX, JPEG, Markdown, PDF, PNG, PPTX, TXT, XLSX,
  and ZIP. The maximum attachment size is 20 MiB.
- The broker permits at most six provider delivery attempts per rolling hour.
- The bearer token and Entra secret are stored in mode-`0600` files and are
  never printed by the setup utilities.
- Audit records contain time, fixed recipient, filename, byte count, and
  SHA-256 only. They contain neither document content nor credentials.
- Graph/provider details are redacted from errors returned to SP4.
- The broker listens on Fastfella's Tailscale address, not a public interface.

## Microsoft 365 administrator steps

1. In Microsoft 365 Admin Center, create shared mailbox
   `hermes-delivery@longboardfella.com.au`.
2. In Entra Admin Center, create an app registration named `Hermes Delivery`.
3. Create a client secret and securely record:
   - Directory (tenant) ID
   - Application (client) ID
   - Client secret value
4. Open **Enterprise applications**, locate `Hermes Delivery`, and record its
   **Object ID**. This is the service-principal object ID; do not use the Object
   ID shown on the App registrations page.
5. Do not add Graph API permissions to this application.
6. In Exchange Admin Center, create a mail-enabled security group named
   `Hermes Delivery Scope` and add only the new shared mailbox as a direct member.
7. Connect with Exchange Online PowerShell and run the following, substituting
   the two IDs:

   ```powershell
   Connect-ExchangeOnline

   $AppId = '<application-client-id>'
   $ServicePrincipalObjectId = '<enterprise-application-object-id>'
   $ScopeGroup = Get-Group 'Hermes Delivery Scope'

   New-ServicePrincipal `
     -AppId $AppId `
     -ObjectId $ServicePrincipalObjectId `
     -DisplayName 'Hermes Delivery'

   New-ManagementScope `
     -Name 'Hermes Delivery Mailbox' `
     -RecipientRestrictionFilter "MemberOfGroup -eq '$($ScopeGroup.DistinguishedName)'"

   New-ManagementRoleAssignment `
     -Name 'Hermes Delivery Send' `
     -App $ServicePrincipalObjectId `
     -Role 'Application Mail.Send' `
     -CustomResourceScope 'Hermes Delivery Mailbox'

   Test-ServicePrincipalAuthorization `
     -Identity $ServicePrincipalObjectId `
     -Resource 'hermes-delivery@longboardfella.com.au' | Format-Table

   Test-ServicePrincipalAuthorization `
     -Identity $ServicePrincipalObjectId `
     -Resource 'lab@longboardfella.com.au' | Format-Table

   Test-ServicePrincipalAuthorization `
     -Identity $ServicePrincipalObjectId `
     -Resource 'intel@longboardfella.com.au' | Format-Table

   Test-ServicePrincipalAuthorization `
     -Identity $ServicePrincipalObjectId `
     -Resource 'notes@longboardfella.com.au' | Format-Table
   ```

   The first test must show `InScope=True`; every other test must show
   `InScope=False`.
   Exchange authorization caches can take 30 minutes to two hours to converge,
   although the test cmdlet evaluates the new assignment immediately.

## Fastfella configuration

The service is `worker/hermes_delivery_service.py`. Copy only the Hermes-specific
values from `worker/config.env.example` into the protected live worker config.
The recipient is fixed in `HERMES_DELIVERY_RECIPIENT`; the HTTP request cannot
override it. Generate `HERMES_DELIVERY_API_TOKEN` with at least 32 random bytes.

To enter the IDs, destination, and secret locally without placing the secret in
chat or shell history, run from the Cortex checkout:

```bash
venv/bin/python worker/configure_hermes_delivery.py
```

The secret prompt does not echo. The script atomically updates `worker/config.env`,
generates a 48-byte API token, and enforces mode `0600`.
It also refuses to configure the sender shared mailbox as its own fixed
destination; the destination must be the owner's actual inbox.

Recommended live settings:

```dotenv
HERMES_DELIVERY_SENDER=hermes-delivery@longboardfella.com.au
HERMES_DELIVERY_HOST=100.118.92.17
HERMES_DELIVERY_PORT=7341
HERMES_DELIVERY_MAX_BYTES=20971520
HERMES_DELIVERY_RATE_PER_HOUR=6
HERMES_DELIVERY_AUDIT_LOG=/home/longboardfella/vault-rag-db/hermes-delivery-audit.jsonl
```

The live recipient is `paul@longboardfella.com.au`. The commissioning process
initially entered the sender address in the recipient field, causing two tests
to be delivered back to the shared mailbox. This was corrected and the
configurator now rejects sender and recipient equality.

Install and start the hardened unit:

```bash
mkdir -p ~/.config/systemd/user
cp ops/systemd/hermes-delivery.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hermes-delivery.service
systemctl --user is-active hermes-delivery.service
curl -fsS http://100.118.92.17:7341/health
```

The unit uses `NoNewPrivileges`, a private temporary directory, read-only home
and system views, mode-`0077` creation, and grants write access only to the
external audit directory.

## SP4 and Discord controls

The installed SP4 client is `/home/paul/.local/bin/kb-email-me`. Its protected
configuration is `/home/paul/.hermes-delivery-client.env` (mode `0600`). The
client:

- accepts only regular, non-symlink `/tmp/kb-*` files;
- applies the same extension and 20 MiB limits as the broker;
- has no recipient option; and
- returns only a safe success or failure result.

The user plugin `~/.hermes/plugins/email-delivery-approval` is enabled. Its
`pre_tool_call` hook escalates every `kb-email-me` invocation to the Hermes
human-approval gate. The rule key includes the individual tool-call ID, so an
"always approve" response cannot authorize a later delivery. The same hook
blocks direct references to the credential file and broker endpoint.

The private Discord channel prompt permits email only after an explicit request
from Paul and forbids arbitrary recipients, alternative mail programs, direct
Graph calls, and retrying a denied delivery.

### SP4 recovery installation

From Fastfella, generate the minimal client configuration without exposing the
token:

```bash
venv/bin/python worker/export_hermes_delivery_client.py
```

Securely copy the generated file and the tracked recovery artifacts to SP4:

```bash
scp ~/.config/hermes-delivery/sp4-client.env \
  paul@100.71.168.3:/home/paul/.hermes-delivery-client.env
scp ops/hermes-delivery/sp4/kb-email-me \
  paul@100.71.168.3:/home/paul/.local/bin/kb-email-me
scp -r ops/hermes-delivery/sp4/email-delivery-approval \
  paul@100.71.168.3:/home/paul/.hermes/plugins/
```

On SP4:

```bash
chmod 600 ~/.hermes-delivery-client.env
chmod 700 ~/.local/bin/kb-email-me
chmod 700 ~/.hermes/plugins/email-delivery-approval
chmod 600 ~/.hermes/plugins/email-delivery-approval/*
hermes plugins enable email-delivery-approval
systemctl --user restart hermes-gateway.service
hermes plugins list --plain --no-bundled
curl -fsS http://100.118.92.17:7341/health
```

The plugin listing must show `email-delivery-approval` as enabled. Do not use
`--allow-tool-override`; this plugin registers a hook and does not need that
privilege.

## Authorization and commissioning checks

`Test-ServicePrincipalAuthorization` is the authoritative safe scope test. It
must report `InScope=True` for `hermes-delivery@` and `InScope=False` for every
mailbox outside the security group, including `lab@`, `intel@`, and `notes@`.

Do not use a Graph mailbox-read request as a scope probe: this application has
only `Application Mail.Send`, so mailbox reads are correctly denied even for
the in-scope sender. Likewise, an invalid `sendMail` payload can be rejected by
request validation before authorization and therefore cannot prove scope.

After the Exchange tests pass, commission the positive path with one harmless
fixed-recipient attachment. On 2026-08-05 the broker returned success and wrote
an audit entry whose byte count and SHA-256 matched the source attachment.
Thereafter, test from Discord: explicitly request one document by email and
confirm that delivery does not execute until the per-call approval is accepted.

### Commissioning evidence, 2026-08-05

- Exchange authorization: Hermes Delivery mailbox `True`; Lab, Intel, and Notes
  mailboxes `False`.
- Broker health returned `{"ok": true}` from SP4 over Tailscale.
- Plugin policy test returned approval for `kb-email-me`, block for direct
  broker access, and no directive for an unrelated command.
- Broker unit tests passed filename, recipient-field rejection, fixed-recipient
  audit, and rate-limit cases.
- SP4 policy tests passed restricted-file and per-call-approval cases.
- The correction test was accepted by Graph, audited with a SHA-256 matching
  the source file, and arrived in Paul's inbox.
- Paul then confirmed that the normal Hermes delivery workflow worked.

## Routine use

In the private Discord channel, ask Hermes to email the original document. When
the approval card appears, verify the filename and subject and choose **Approve
once**. Deny the request if the file is unexpected. No shared mailbox needs to
be visible in Outlook.

## Operations and troubleshooting

```bash
# Fastfella broker
systemctl --user status hermes-delivery.service --no-pager
journalctl --user -u hermes-delivery.service -n 100 --no-pager
curl -fsS http://100.118.92.17:7341/health

# Content-free delivery audit
tail -20 /home/longboardfella/vault-rag-db/hermes-delivery-audit.jsonl

# SP4 gateway and plugin
ssh paul@100.71.168.3 'systemctl --user status hermes-gateway.service --no-pager'
ssh paul@100.71.168.3 'hermes plugins list --plain --no-bundled'
```

Common failures:

| Symptom | Check | Resolution |
|---|---|---|
| Hermes says sent, but nothing arrives | Last audit record's `recipient` | It must be Paul's inbox, not the sender shared mailbox. |
| Client says provider rejected delivery | Broker journal and Exchange authorization | Wait for RBAC propagation; verify the group has only the Hermes mailbox. |
| HTTP 401 | SP4 and broker bearer tokens differ | Re-export and securely recopy the SP4 client configuration. |
| HTTP 429 | Six attempts occurred within one hour | Wait for the rolling window; do not raise the limit casually. |
| File rejected | `/tmp/kb-*`, regular file, extension and size | Fetch the original with the existing `kb-*` command and use its returned path. |
| No approval card | Plugin status and gateway restart | Enable the plugin and restart the gateway; do not send until approval works. |
| Shared mailbox absent from Outlook | Expected for this design | No action is required; Paul only needs the delivered message in his inbox. |

## Secret rotation

### Broker bearer token

1. Generate a new value of at least 32 random bytes.
2. Replace `HERMES_DELIVERY_API_TOKEN` in protected `worker/config.env`.
3. Restart `hermes-delivery.service`.
4. Run `worker/export_hermes_delivery_client.py` and securely copy the new
   protected client file to SP4.
5. Verify health and perform one approved delivery. The old token stops working
   as soon as the broker restarts.

### Entra client secret

1. Create a new secret value in the existing Hermes Delivery app registration.
2. Update only `HERMES_DELIVERY_CLIENT_SECRET` in protected `worker/config.env`.
3. Restart and test the broker.
4. Delete the old Entra secret only after the test succeeds.

## Emergency revocation

For immediate shutdown:

```bash
systemctl --user disable --now hermes-delivery.service
ssh paul@100.71.168.3 'hermes plugins disable email-delivery-approval'
```

Then remove or disable the Entra client secret. For complete Microsoft-side
revocation, remove the `Hermes Delivery Send` management-role assignment. Do
not add the Hermes app to a broader scope as a shortcut during recovery.

## Domain authentication follow-up

The 2026-08-05 public DNS check found Microsoft 365 MX and SPF, but no published
DMARC record or standard Microsoft 365 DKIM selector CNAMEs. Enable DKIM for the
domain in Microsoft 365, publish the two selector CNAMEs Microsoft supplies,
then introduce DMARC with monitoring (`p=none`) before moving to quarantine or
reject. Do not invent selector targets; copy the tenant-specific values shown by
Microsoft 365.

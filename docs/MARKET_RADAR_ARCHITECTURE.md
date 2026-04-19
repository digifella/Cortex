# Market Radar Architecture

Purpose: explain, in plain language, how the Market Radar and mailbox-driven intelligence system works today in Cortex.

## What This System Does

Market Radar is the intelligence loop for:

- gathering external signals about people and organisations
- accepting manually submitted notes and documents
- extracting structured intelligence from those submissions
- routing the results into the subscriber organisation's intelligence context
- returning useful summaries and suggested actions

In practice, there are two main ways intelligence enters the system:

1. scheduled or manual web scans through the `market_radar` workflow
2. email/mailbox submissions through the Intel mailbox workflow

These two paths are separate at the front door, but they converge on the same broad goal:

- create useful, scoped, organisation-aware intelligence

## The Main Components

### 1. Website Market Radar Worker

Primary role:

- search the web for target-specific signals
- synthesise a watch report

Relevant handler:

- [worker/handlers/market_radar.py](/home/longboardfella/cortex_suite/worker/handlers/market_radar.py)

What it receives:

- targets
- search scope
- subscriber company context
- intelligence focus
- optional prior report context

What it produces:

- a prioritised report
- raw signals suitable for later harvesting and ingest

Conceptually, this is the outward-looking research engine.

### 2. Intel Mailbox

Primary role:

- poll an IMAP inbox
- parse incoming emails and attachments
- classify what kind of submission each message is
- route it into the right downstream path

Core files:

- [intel_mailbox.py](/home/longboardfella/cortex_suite/cortex_engine/intel_mailbox.py)
- [intel_mailbox_worker.py](/home/longboardfella/cortex_suite/worker/intel_mailbox_worker.py)

This is the human input channel for:

- notes
- screenshots
- PDFs
- org charts
- annual reports
- strategic plans
- direct intelligence messages

### 3. Intel Note Classifier

Primary role:

- decide whether an email is:
  - a direct `intel_extract` message
  - a structured note
  - a document analysis submission
  - a CSV profile import

Relevant file:

- [intel_note_classifier.py](/home/longboardfella/cortex_suite/cortex_engine/intel_note_classifier.py)

This is the first branching decision after the mailbox parser.

### 4. Intel Note Processor

Primary role:

- process mailbox payloads once classified
- analyse attachments such as:
  - org charts
  - strategic plans
  - annual reports
- merge extracted people, organisations, signals, and document cues

Relevant file:

- [intel_note_processor.py](/home/longboardfella/cortex_suite/cortex_engine/intel_note_processor.py)

This is the content understanding layer for mailbox-driven intelligence.

### 5. Stakeholder Signal Store

Primary role:

- persist intelligence signals
- maintain organisation context
- support profile matching and relationship reasoning

This is the durable memory layer that lets intelligence accumulate instead of being one-off.

## The Two Main Intake Paths

## A. Web Watch Reports

This path begins from a Market Radar watch run.

Typical flow:

1. Website submits a watch configuration.
2. `market_radar` searches the web for each target.
3. Signals are synthesised into a report.
4. The report is harvested on the website side.
5. The raw signals can then be fed back into Cortex for longer-term use.

This is the machine-led discovery path.

## B. Mailbox / Human Submission Path

This path begins with an email arriving in the Intel mailbox.

Typical flow:

1. `intel_mailbox_worker.py` polls IMAP.
2. [parse_email_bytes()](/home/longboardfella/cortex_suite/cortex_engine/intel_mailbox.py) extracts:
   - subject
   - from/to addresses
   - text
   - html
   - attachments
3. The mailbox checks whether the sender is allowed.
4. The message is classified.
5. Routing logic decides:
   - which subscriber org it belongs to
   - whether subject overrides apply
   - whether document hints are present
6. The note processor analyses the content.
7. Cortex builds either:
   - a direct extraction result
   - or a structured note-ingest payload
8. The result is posted to the configured callback or stored locally in the outbox.

This is the human-led intelligence path.

## Subject Syntax You Use in Practice

### Direct extraction path

Use:

- `INTEL: <headline>`
- or include `[intel]` in the subject

Examples:

- `INTEL: Carolyn Bell update`
- `INTEL: Seqwater leadership changes`

Effect:

- triggers the `intel_extract` path
- Cortex extracts people, organisations, emails, and signals directly from the email and attachments

Use this when you want quick extraction rather than scoped note routing.

### Structured note or document path

Use:

- `entity: <subscriber org> | <headline>`

Optional modifiers:

- `depth:detailed`
- `force:yes`

Examples:

- `entity: Escient | Barwon Water annual report`
- `entity: Escient | depth:detailed | Yarra Valley Water strategic plan`
- `entity: Escient | force:yes | South East Water strategic plan`

Effect:

- routes the submission to the subscriber org scope
- preserves document/note context
- produces a structured note-ingest payload for the Market Radar side

Use this when you want routing and scope, not just raw extraction.

## Authorisation Model

Mailbox allowlist setting:

- `INTEL_ALLOWED_SENDERS`

Relevant logic:

- [_allowed_sender()](/home/longboardfella/cortex_suite/cortex_engine/intel_mailbox.py)

Current behaviour:

- if no allowlist is configured, mailbox submissions are accepted
- if an allowlist is configured, sender matching is normally by `from_email`
- trusted self-relay is explicitly allowed:
  - mail from `intel@longboardfella.com.au` to itself is treated as authorised
  - these messages are currently attributed as coming from `paul@longboardfella.com.au`
  - the legacy `intel.longboardfella@gmail.com` relay remains accepted during migration

This is the temporary Telegram relay rule now in place.

## Routing Logic

Relevant function:

- [_resolve_message_routing()](/home/longboardfella/cortex_suite/cortex_engine/intel_mailbox.py)

What routing determines:

- default org scope
- requested org override from the subject
- matched known org scope
- sender-domain-based org scope
- clean subject after removing routing modifiers
- document subject hints
- extraction depth
- force reingest

This is why a subject like:

- `entity: Escient | depth:detailed | Barwon Water annual report`

ends up as:

- subscriber org: `Escient`
- cleaned document subject: `Barwon Water annual report`
- subject org hint: `Barwon Water`
- extraction depth: `detailed`

## What Happens After Processing

There are two main output patterns.

### 1. Direct extraction result

Used for:

- `INTEL:` submissions

The system:

- extracts entities and signals
- ingests a signal into Cortex
- returns a structured extraction result

### 2. Structured note ingest

Used for:

- notes
- routed document submissions
- org chart and strategy analysis

The system builds a payload shaped for note ingestion, including:

- routed org scope
- note content
- document metadata
- primary entity
- referenced entities
- URLs
- graph enrichment
- fit assessment

The callback target is typically:

- the Market Radar website note-ingest endpoint

## Replies Back to the Sender

The mailbox system also builds reply emails.

Relevant logic:

- [_build_intel_reply()](/home/longboardfella/cortex_suite/cortex_engine/intel_mailbox.py)
- [_send_reply()](/home/longboardfella/cortex_suite/cortex_engine/intel_mailbox.py)

Replies summarise:

- which subscriber scope was used
- extraction depth
- primary entity
- fit assessment
- the main extracted intelligence

This gives the user immediate feedback without needing to inspect raw JSON.

## How This Connects to Market Radar

In plain terms:

- Market Radar web scans create new external signals
- mailbox submissions create user-supplied or document-derived signals
- both should enrich the same subscriber intelligence picture over time

The long-term design direction is:

- watch reports feed raw signals back into Cortex
- mailbox notes and documents enrich the same signal store
- later digest/reporting layers reason across both

That is the compounding intelligence model.

## Current Strengths

- mailbox path handles mixed human input well
- subject syntax is powerful but still lightweight
- scoped routing is explicit
- document analysis and note analysis share the same intake channel
- direct extraction and structured note ingestion are both supported

## Current Limitations

- website Market Radar scans and Cortex memory are not yet as tightly closed-loop as they should be
- relationship intelligence is stronger in the stored data model than in final report generation
- sender identity is still relatively simple and partly rule-based
- the Telegram relay path is currently a pragmatic exception, not a full identity model

## Recommended Mental Model

Think of the system as three layers:

1. Intake
   - web scans
   - email notes
   - documents

2. Understanding
   - classify
   - route
   - extract
   - enrich

3. Memory and action
   - store signals
   - feed Market Radar
   - reply to the user
   - produce usable intelligence

That is the current Market Radar architecture in human terms.

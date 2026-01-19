# Mention-Based Proposal System - Design Specification
**Version:** 1.0.0
**Date:** 2026-01-05
**Status:** Implementation Ready

## 🎯 Core Philosophy

**Human-Curated Data + AI-Assisted Assembly**

- Humans define entity data once (accurately, in YAML files)
- LLM suggests where to use data in tenders (first-pass markup)
- Humans review/refine (iterative approval workflow)
- System generates final proposal (auto-fill + creative content)

## 📁 Directory Structure

```
/mnt/f/ai_databases/
├── entity_profiles/                    # All entity data lives here
│   ├── longboardfella_consulting/      # Entity ID (URL-safe, lowercase_underscore)
│   │   ├── profile.yaml                # Core entity metadata & simple fields
│   │   ├── narrative.md                # Long-form content (company overview, capabilities)
│   │   ├── team/
│   │   │   ├── paul_smith.yaml         # Person ID = filename
│   │   │   └── jane_doe.yaml
│   │   ├── projects/
│   │   │   ├── health_transformation_2023.yaml
│   │   │   └── procurement_reform_2022.yaml
│   │   ├── references/
│   │   │   ├── sarah_johnson.yaml
│   │   │   ├── michael_chen.yaml
│   │   │   └── emma_williams.yaml
│   │   ├── capabilities/
│   │   │   └── iso_9001_2015.yaml
│   │   └── insurance/
│   │       ├── public_liability.yaml
│   │       ├── professional_indemnity.yaml
│   │       └── workers_compensation.yaml
│   │
│   └── consortium_partner_a/           # Future: multi-entity support
│       └── ...
│
└── workspaces/                         # Tender-specific workspaces
    └── workspace_RFT12493_2026-01-05/
        ├── metadata.yaml               # Workspace config
        ├── tender_original.docx        # Original tender document
        ├── tender_marked_up.docx       # With @mentions inserted
        ├── tender_filled.docx          # Final generated proposal
        ├── field_bindings.yaml         # @mention → entity field mappings
        ├── approval_status.yaml        # Multi-stage approval tracking
        ├── generation_log.json         # What was generated, when, by whom
        └── .git/                       # Git repo for version control
```

## 📋 YAML Schemas

### 1. Entity Profile Schema

**File:** `entity_profiles/{entity_id}/profile.yaml`

```yaml
---
# === METADATA ===
metadata:
  entity_id: longboardfella_consulting      # Unique ID (URL-safe)
  entity_name: Longboardfella Consulting Pty Ltd  # Display name
  entity_type: consulting_firm              # Type: consulting_firm, contractor, consortium
  created_date: 2026-01-05
  last_updated: 2026-01-05
  version: "1.0.0"
  status: active                            # active, archived
  tags: [consulting, digital_transformation, government]

# === SIMPLE FIELDS (Direct Substitution) ===
company:
  legal_name: Longboardfella Consulting Pty Ltd
  trading_names:
    - Longboardfella
    - LBF Consulting
  abn: "12 345 678 901"                     # String to preserve formatting
  acn: "123 456 789"
  registration_date: 2010-06-15

contact:
  registered_office:
    street: 123 Beach Road
    city: Sydney
    state: NSW
    postcode: "2000"
    country: Australia
  postal_address: null                      # If different from registered
  phone: "+61 2 1234 5678"
  email: info@longboardfella.com.au
  website: https://www.longboardfella.com.au

# === COMPLEX FIELDS (References to Other Files) ===
team:
  - paul_smith          # → team/paul_smith.yaml
  - jane_doe            # → team/jane_doe.yaml

projects:
  - health_transformation_2023
  - procurement_reform_2022

references:
  - sarah_johnson
  - michael_chen
  - emma_williams

capabilities:
  - iso_9001_2015

insurance:
  - public_liability
  - professional_indemnity
  - workers_compensation

# === FORMATTING PREFERENCES ===
formatting:
  abn_format: "XX XXX XXX XXX"              # How to format ABN in output
  acn_format: "XXX XXX XXX"
  phone_format: international               # international, local
  date_format: "DD/MM/YYYY"                 # ISO8601, DD/MM/YYYY, etc.
  currency_format: "AUD"

# === NARRATIVE SECTIONS (Keys map to narrative.md headings) ===
narrative_sections:
  - company_overview
  - core_capabilities
  - competitive_advantages
  - quality_assurance
  - sustainability_commitment
```

### 2. Team Member Schema

**File:** `entity_profiles/{entity_id}/team/{person_id}.yaml`

```yaml
---
person_id: paul_smith
full_name: Paul Smith
preferred_name: Paul
role: Senior Consultant
email: paul.smith@longboardfella.com.au
phone: "+61 400 123 456"

qualifications:
  - name: Master of Business Administration
    institution: University of Technology Sydney
    year: 2015
    specialization: Business Strategy
    credential_id: null
  - name: Certified Management Consultant
    institution: Institute of Management Consultants
    year: 2018
    certification_number: CMC-2018-12345

experience:
  - role: Senior Consultant
    organization: Longboardfella Consulting
    start_date: 2015-01-01
    end_date: null                          # null = current
    location: Sydney, Australia
    responsibilities:
      - Leading consulting engagements for government agencies
      - Strategic advisory on digital transformation
      - Project management for complex initiatives
    achievements:
      - Led $850K digital health transformation project
      - Delivered 30% improvement in client data processing
  - role: Business Analyst
    organization: Deloitte Consulting
    start_date: 2010-06-01
    end_date: 2014-12-31
    location: Sydney, Australia
    responsibilities:
      - Business process analysis
      - Requirements gathering
      - Stakeholder management

certifications:
  - name: Certified ScrumMaster
    issuer: Scrum Alliance
    date: 2016-03-15
    expiry: null
  - name: PRINCE2 Practitioner
    issuer: AXELOS
    date: 2014-09-20
    expiry: null

# Narrative bio (for CV generation)
bio:
  brief: |
    Paul Smith is a senior consultant with over 14 years of experience in
    strategic advisory and digital transformation for government agencies.

  full: |
    Paul Smith is a senior consultant specializing in digital transformation
    and process improvement for government agencies. With an MBA from UTS and
    over 14 years of experience, Paul has led numerous high-value engagements
    across health, finance, and veterans' affairs sectors.

    Paul's expertise includes strategic planning, change management, and
    stakeholder engagement for complex government IT initiatives. He has a
    proven track record of delivering measurable outcomes, including a recent
    $850K digital health transformation that achieved 30% improvement in
    data processing efficiency.

# Generation preferences
generation_preferences:
  cv_format: professional                   # professional, academic, brief
  include_photo: false
  max_cv_pages: 3
```

### 3. Project Schema

**File:** `entity_profiles/{entity_id}/projects/{project_id}.yaml`

```yaml
---
project_id: health_transformation_2023
project_name: Department of Health Digital Transformation
client: Australian Department of Health
sector: government                          # government, private, nfp
project_type: digital_transformation        # Tags for filtering

timeline:
  start_date: 2023-01-01
  end_date: 2024-06-30
  duration_months: 18

financials:
  contract_value: 850000.00
  currency: AUD
  payment_structure: milestone_based

team:
  size: 8
  our_staff:
    - paul_smith                            # References team members
  roles:
    - role: Lead Consultant
      person: paul_smith
    - role: Technical Architect
      person: jane_doe

description:
  brief: |
    Led digital transformation initiative to modernize health data systems
    and improve patient outcomes.

  full: |
    Longboardfella Consulting led a comprehensive digital transformation
    initiative for the Australian Department of Health, modernizing legacy
    data systems and implementing cloud-based analytics platforms. The
    18-month engagement involved strategic planning, system architecture,
    change management, and stakeholder engagement across multiple
    departmental divisions.

deliverables:
  - name: Strategic Roadmap
    description: 5-year digital transformation strategy
  - name: Implementation Plan
    description: Detailed 18-month delivery plan with milestones
  - name: Change Management Framework
    description: Stakeholder engagement and training program
  - name: Technical Architecture
    description: Cloud platform design and migration plan

outcomes:
  - metric: Data processing efficiency
    improvement: 30% improvement
    measurement: Processing time reduction
  - metric: Reporting time
    improvement: 50% reduction
    measurement: Time to generate monthly reports
  - metric: User satisfaction
    improvement: Increased from 65% to 89%
    measurement: Annual staff survey

technologies:
  - Cloud platforms (AWS)
  - Data analytics (Tableau, Power BI)
  - API integration (REST, GraphQL)
  - Agile delivery (Scrum, Kanban)

challenges_overcome:
  - Legacy system integration with 20+ year-old databases
  - Complex stakeholder landscape across 12 divisions
  - Strict government security and privacy requirements

reference:
  contact: sarah_johnson                    # References references/sarah_johnson.yaml
  available: true
  confidential: false                       # If true, don't include details

# Generation preferences
generation_preferences:
  summary_length: 200                       # Words for brief summary
  focus_areas:                              # What to emphasize
    - outcomes
    - scale
    - complexity
```

### 4. Reference Schema

**File:** `entity_profiles/{entity_id}/references/{reference_id}.yaml`

```yaml
---
reference_id: sarah_johnson
contact_name: Dr. Sarah Johnson
title: Director of Digital Strategy
organization: Australian Department of Health
email: sarah.johnson@health.gov.au
phone: "+61 2 6289 1234"

relationship:
  type: client                              # client, partner, subcontractor
  role: Project Sponsor
  projects:
    - health_transformation_2023            # Which projects they can speak to

availability:
  available: true
  preferred_contact: email                  # email, phone
  best_times: business_hours
  notes: |
    Sarah prefers email introduction before phone contact.
    Available for reference checks during business hours.

context:
  working_relationship: |
    Sarah was the Project Sponsor for our 2023-2024 digital health
    transformation engagement. She provided strategic oversight and
    executive stakeholder management.

  can_speak_to:
    - Strategic planning capabilities
    - Stakeholder management skills
    - Delivery quality and timeliness
    - Change management expertise

confidentiality:
  public: true                              # Can be listed in public proposals
  name_only: false                          # If true, only list name/org, not contact details
  on_request: false                         # If true, provide only upon request

# Pre-approved quote (if available)
quote: |
  "Longboardfella Consulting delivered exceptional results on our digital
  transformation initiative. Their strategic approach and stakeholder
  engagement skills were instrumental in achieving a 30% improvement in
  our data processing efficiency."
```

### 5. Insurance Policy Schema

**File:** `entity_profiles/{entity_id}/insurance/{policy_type}.yaml`

```yaml
---
policy_id: public_liability
policy_type: public_liability               # public_liability, professional_indemnity, workers_comp, cyber
insurer: Insurance Australia Group
policy_number: PL-2024-123456

coverage:
  amount: 20000000.00
  currency: AUD
  formatted: "$20,000,000"                  # How to display
  description: Public and products liability coverage

dates:
  effective_date: 2024-01-01
  expiry_date: 2025-12-31
  renewal_status: current                   # current, expiring_soon, expired

scope:
  geographic: Australia and New Zealand
  activities:
    - Management consulting services
    - Strategic advisory
    - Digital transformation consulting
  exclusions:
    - Software development
    - Direct employment services

broker:
  name: ABC Insurance Brokers
  contact: John Smith
  phone: "+61 2 1234 5678"
  email: john.smith@abcbrokers.com.au

documents:
  certificate_path: /path/to/certificate.pdf
  policy_document_path: /path/to/policy.pdf
```

### 6. Capability/Certification Schema

**File:** `entity_profiles/{entity_id}/capabilities/{capability_id}.yaml`

```yaml
---
capability_id: iso_9001_2015
capability_name: ISO 9001:2015 Quality Management
capability_type: certification              # certification, accreditation, membership

description:
  brief: Certified quality management system for consulting services
  full: |
    Longboardfella Consulting maintains ISO 9001:2015 certification for
    quality management in consulting and advisory services. Our certified
    QMS ensures consistent delivery quality, continuous improvement, and
    client satisfaction.

certification_body: SAI Global
certification_number: QMS-2024-AUS-12345

dates:
  obtained: 2020-03-15
  expiry: 2026-03-14
  next_audit: 2025-09-15

scope:
  services:
    - Management consulting
    - Strategic advisory
    - Digital transformation consulting
  locations:
    - Sydney, Australia
  standards:
    - ISO 9001:2015

evidence:
  certificate_path: /path/to/iso_certificate.pdf
  audit_reports:
    - /path/to/audit_2024.pdf
```

## 🔤 Mention Syntax Specification

### Syntax: `@{type}.{path}[parameters]`

**Components:**
- `@` - Mention prefix (required)
- `{type}` - Entity field type (optional if simple field)
- `.{path}` - Dot-notation path to field (required)
- `[parameters]` - Optional parameters for generation (key=value pairs)

### Categories

#### A. Simple Field Substitution
```
@companyname                                → profile.company.legal_name
@abn                                        → profile.company.abn (formatted)
@acn                                        → profile.company.acn (formatted)
@registered_office                          → Formatted address
@email                                      → profile.contact.email
@phone                                      → profile.contact.phone (formatted)
@website                                    → profile.contact.website
```

#### B. Structured Data Access (Dot Notation)
```
@insurance.public_liability.coverage        → "$20,000,000"
@insurance.public_liability.policy_number   → "PL-2024-123456"
@insurance.public_liability.expiry          → "31/12/2025" (formatted)

@team.paul_smith.role                       → "Senior Consultant"
@team.paul_smith.qualifications             → List of qualifications

@project.health_transformation_2023.value   → "$850,000"
@project.health_transformation_2023.client  → "Australian Department of Health"
```

#### C. Content Generation (Complex)
```
# CV Generation
@cv[paul_smith]                             → Full CV (default format)
@cv[paul_smith, format=brief]               → Brief CV (1 page)
@cv[paul_smith, format=brief, max_words=500] → Very brief CV
@cv[paul_smith, sections=experience,qualifications] → Specific sections only

# Project Summaries
@project_summary[health_transformation_2023] → Full project description
@project_summary[health_transformation_2023, focus=outcomes] → Outcome-focused
@project_summary[health_transformation_2023, length=200] → Max 200 words

# Team Qualifications
@team_qualifications                        → All team qualifications
@team_qualifications[person=paul_smith]     → Specific person
@team_qualifications[type=certification]    → Only certifications
@team_qualifications[format=table]          → Table format

# References
@reference[sarah_johnson]                   → Formatted reference with contact details
@reference[sarah_johnson, include_quote=true] → Include pre-approved quote
@references                                 → All references formatted
@references[type=government]                → Filtered references

# Insurance Summary
@insurance_summary                          → All policies formatted
@insurance_summary[type=public_liability]   → Specific policy
@insurance_summary[format=table]            → Table format

# Narrative Content
@narrative[company_overview]                → From narrative.md ## Company Overview section
@narrative[capabilities.digital_transformation] → Specific capability narrative
```

#### D. Creative Generation (LLM-Powered)
```
@generate[type=executive_summary, topic=digital health, length=500, tone=professional]
@generate[type=approach, context=procurement reform, methodology=agile]
@generate[type=risk_mitigation, project_type=government IT]
@generate[type=team_introduction, team_members=paul_smith|jane_doe]
@generate[type=innovation_statement, focus=AI and automation]

# With context from tender
@generate[type=response, section=selection_criteria_1, max_words=1000]
@generate[type=response, section=evaluation_question_3, include_evidence=true]
```

## 🔄 Workflow State Machine

### States

```
1. created           → Workspace created, tender uploaded
2. markup_suggested  → LLM completed first-pass markup
3. markup_reviewed   → Human reviewed and approved/modified markup
4. entity_bound      → Entity profile bound to workspace
5. content_generated → All @mentions filled
6. draft_ready       → Draft proposal ready for review
7. in_review         → Under review by reviewer
8. revisions_needed  → Reviewer requested changes
9. approved          → Final approval
10. exported         → Final document exported
11. submitted        → Submitted to client
```

### Transitions

```mermaid
created → markup_suggested:    LLM completes first pass
markup_suggested → markup_reviewed:   Human approves/modifies
markup_reviewed → entity_bound:       Entity profile selected
entity_bound → content_generated:     Generate all content
content_generated → draft_ready:      All sections filled
draft_ready → in_review:              Send for review
in_review → approved:                 Reviewer approves
in_review → revisions_needed:         Reviewer requests changes
revisions_needed → content_generated: Drafter makes revisions
approved → exported:                  Generate final .docx
exported → submitted:                 Mark as submitted
```

### Approval Workflow

**File:** `workspaces/{workspace_id}/approval_status.yaml`

```yaml
---
workspace_id: workspace_RFT12493_2026-01-05
tender_id: RFT12493
current_state: in_review

# Multi-stage approval
stages:
  - stage: draft
    role: drafter
    assigned_to: paul.smith@longboardfella.com.au
    status: completed
    completed_date: 2026-01-05T14:30:00Z
    comments: Initial draft completed

  - stage: review
    role: reviewer
    assigned_to: jane.doe@longboardfella.com.au
    status: in_progress
    started_date: 2026-01-05T15:00:00Z
    comments: null

  - stage: final_approval
    role: approver
    assigned_to: director@longboardfella.com.au
    status: pending
    started_date: null
    comments: null

# Section-level tracking
sections:
  - section_id: executive_summary
    status: approved                        # approved, needs_revision, pending
    reviewed_by: jane.doe@longboardfella.com.au
    reviewed_date: 2026-01-05T15:15:00Z
    comments: Excellent summary

  - section_id: team_qualifications
    status: needs_revision
    reviewed_by: jane.doe@longboardfella.com.au
    reviewed_date: 2026-01-05T15:20:00Z
    comments: Add more detail on cybersecurity certifications

  - section_id: approach_methodology
    status: pending
    reviewed_by: null
    reviewed_date: null
    comments: null

# Change history
history:
  - timestamp: 2026-01-05T10:00:00Z
    user: paul.smith@longboardfella.com.au
    action: created_workspace
    details: Uploaded tender RFT12493

  - timestamp: 2026-01-05T11:00:00Z
    user: system
    action: llm_markup_suggested
    details: 47 mentions suggested

  - timestamp: 2026-01-05T14:30:00Z
    user: paul.smith@longboardfella.com.au
    action: markup_reviewed
    details: 42 mentions approved, 5 modified

  - timestamp: 2026-01-05T15:00:00Z
    user: paul.smith@longboardfella.com.au
    action: submitted_for_review
    details: Assigned to jane.doe@longboardfella.com.au
```

## 🔧 Git Integration Strategy

### Repository Structure

Each workspace is its own Git repository:

```
/mnt/f/ai_databases/workspaces/workspace_RFT12493_2026-01-05/
├── .git/
├── .gitignore
├── README.md                   # Auto-generated workspace summary
├── metadata.yaml
├── tender_original.docx        # Binary, tracked
├── tender_marked_up.docx       # Binary, tracked
├── tender_filled.docx          # Binary, tracked
├── field_bindings.yaml         # Text, tracked
├── approval_status.yaml        # Text, tracked
└── generation_log.json         # Text, tracked
```

### Git Operations

```python
# On workspace creation
git init
git add .
git commit -m "Initial commit: Tender RFT12493 uploaded"

# On markup completion
git add tender_marked_up.docx field_bindings.yaml
git commit -m "LLM markup: 47 mentions suggested"

# On human review
git add tender_marked_up.docx field_bindings.yaml
git commit -m "Markup reviewed: 42 approved, 5 modified by paul.smith"

# On content generation
git add tender_filled.docx generation_log.json
git commit -m "Content generated: All sections filled"

# On reviewer feedback
git add approval_status.yaml tender_filled.docx
git commit -m "Review feedback: team_qualifications needs revision - jane.doe"

# On revision
git add tender_filled.docx
git commit -m "Revision: Added cybersecurity certifications - paul.smith"

# On approval
git add approval_status.yaml
git commit -m "Final approval granted - director"

# On export
git add tender_final.docx
git tag -a v1.0 -m "Final submission version"
git commit -m "Final export: Ready for submission"
```

### Branching Strategy (Optional, for complex tenders)

```
main                    → Production-ready versions
├── drafting           → Active drafting work
├── review/jane-doe    → Reviewer's branch for comments
└── archive/v1.0       → Historical versions
```

## 📊 Field Bindings

**File:** `workspaces/{workspace_id}/field_bindings.yaml`

Tracks all @mentions in the document and their resolution:

```yaml
---
# Auto-detected mentions
mentions:
  - mention_id: m001
    mention_text: "@companyname"
    location:
      document: tender_marked_up.docx
      section: Cover Page
      paragraph: 1

    binding:
      entity_id: longboardfella_consulting
      field_path: profile.company.legal_name
      resolved_value: "Longboardfella Consulting Pty Ltd"
      resolution_method: auto                # auto, manual, generated

    status: bound                            # detected, bound, generated, failed
    reviewed: true
    reviewed_by: paul.smith@longboardfella.com.au
    reviewed_date: 2026-01-05T14:15:00Z

  - mention_id: m002
    mention_text: "@cv[paul_smith, format=brief]"
    location:
      document: tender_marked_up.docx
      section: Team Qualifications
      paragraph: 15

    binding:
      entity_id: longboardfella_consulting
      field_path: team.paul_smith
      parameters:
        format: brief
      resolved_value: null                   # Generated on-demand
      resolution_method: generated

    status: pending_generation
    reviewed: true
    reviewed_by: paul.smith@longboardfella.com.au
    reviewed_date: 2026-01-05T14:20:00Z

  - mention_id: m003
    mention_text: "@generate[type=executive_summary, topic=digital health, length=500]"
    location:
      document: tender_marked_up.docx
      section: Executive Summary
      paragraph: 1

    binding:
      entity_id: null                        # Creative generation, no direct entity link
      field_path: null
      parameters:
        type: executive_summary
        topic: digital health
        length: 500
      resolved_value: null
      resolution_method: llm_generated
      llm_model: qwen2.5:14b-instruct-q4_K_M
      generation_context:
        tender_requirements: [selection_criteria_1, selection_criteria_2]
        entity_data: [company_overview, recent_projects]

    status: pending_generation
    reviewed: true
    reviewed_by: paul.smith@longboardfella.com.au
    reviewed_date: 2026-01-05T14:25:00Z

# Statistics
statistics:
  total_mentions: 47
  bound: 35
  pending_generation: 10
  failed: 2
  reviewed: 45
  unreviewed: 2
```

## 🎨 Template Library Structure

Templates provide common mention patterns for different tender types:

```
/mnt/f/ai_databases/templates/
├── government_it_services/
│   ├── template.yaml
│   └── common_sections.yaml
├── consulting_services/
│   ├── template.yaml
│   └── common_sections.yaml
└── construction_trades/
    ├── template.yaml
    └── common_sections.yaml
```

**Example:** `templates/government_it_services/template.yaml`

```yaml
---
template_id: government_it_services
template_name: Government IT Services Tender
version: "1.0.0"
description: Standard template for Australian government IT services tenders

# Common sections and their typical mentions
sections:
  - section_name: Cover Page
    typical_mentions:
      - "@companyname"
      - "@abn"
      - "@registered_office"
      - "@email"
      - "@phone"

  - section_name: Executive Summary
    typical_mentions:
      - "@generate[type=executive_summary, topic=<tender_topic>, length=500]"
      - "@narrative[company_overview]"

  - section_name: Company Profile
    typical_mentions:
      - "@companyname"
      - "@abn"
      - "@acn"
      - "@registered_office"
      - "@narrative[company_overview]"
      - "@narrative[core_capabilities]"

  - section_name: Insurance
    typical_mentions:
      - "@insurance_summary[format=table]"
      - "@insurance.public_liability.coverage"
      - "@insurance.professional_indemnity.coverage"
      - "@insurance.cyber.coverage"

  - section_name: Team Qualifications
    typical_mentions:
      - "@team_qualifications[format=table]"
      - "@cv[<key_person>, format=brief]"

  - section_name: Relevant Experience
    typical_mentions:
      - "@project_summary[<recent_project_1>]"
      - "@project_summary[<recent_project_2>]"
      - "@project_summary[<recent_project_3>]"

  - section_name: References
    typical_mentions:
      - "@references[type=government]"
      - "@reference[<specific_reference>, include_quote=true]"

  - section_name: Quality Management
    typical_mentions:
      - "@capability[iso_9001_2015]"
      - "@narrative[quality_assurance]"

# Common questions/criteria patterns
common_criteria:
  - pattern: "Demonstrate relevant experience"
    suggested_mentions:
      - "@project_summary[<relevant_project>, focus=outcomes]"
      - "@reference[<project_reference>, include_quote=true]"

  - pattern: "Describe your approach and methodology"
    suggested_mentions:
      - "@generate[type=approach, methodology=agile, length=800]"
      - "@narrative[capabilities.<relevant_capability>]"

  - pattern: "Team qualifications and experience"
    suggested_mentions:
      - "@team_qualifications"
      - "@cv[<key_person>, format=brief, max_words=500]"
```

## 🚀 Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
- [ ] Entity profile Pydantic schemas
- [ ] Entity profile manager (CRUD)
- [ ] Mention parser (regex + validation)
- [ ] Simple field substitution engine
- [ ] Basic UI for profile editing
- [ ] Git integration (auto-commit)

### Phase 2: LLM Integration (Week 2)
- [ ] Tender markup LLM (first-pass suggestions)
- [ ] Structured content generator (@cv, @project_summary)
- [ ] Creative content generator (@generate[...])
- [ ] Review UI (side-by-side view)
- [ ] Multi-stage approval workflow

### Phase 3: Polish & Production (Week 3)
- [ ] Validation & error handling
- [ ] Export to .docx with formatting preservation
- [ ] Template library management
- [ ] Analytics dashboard
- [ ] Documentation & training materials

### Phase 4: Future Enhancements
- [ ] Multi-entity consortium support
- [ ] AI learning from past successful responses
- [ ] Automated quality checks (completeness, tone, compliance)
- [ ] Integration with proposal submission portals
- [ ] Mobile review interface

## 📝 Key Design Principles

1. **Human Authority**: Humans define the data, AI suggests usage
2. **Transparency**: Everything is text files (YAML, MD), readable and editable
3. **Iteration-Friendly**: Save and resume at any point, multi-stage workflow
4. **Version Controlled**: Git tracks every change with clear attribution
5. **Modular**: Entity profiles are reusable across multiple tenders
6. **Fail-Safe**: Failed generation doesn't break the workflow
7. **Auditable**: Complete trail of who changed what and when
8. **Extensible**: Easy to add new mention types, entity types, templates

---

**Status:** ✅ Design approved - Ready for implementation
**Next:** Implement Phase 1 (Core Infrastructure)

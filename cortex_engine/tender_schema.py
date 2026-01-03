"""
Tender Data Schema - Structured Data Models
Version: 1.0.0
Date: 2026-01-03

Purpose: Define structured data models for extracting organizational information
from unstructured knowledge base documents (PDFs, Word docs, etc.) to auto-fill
tender document fields.

Key Models:
- OrganizationProfile: Legal entity details (ABN, ACN, address, contact)
- Insurance: Policy details, coverage, expiry dates
- Qualification: Team member certifications and education
- WorkExperience: Employment history and achievements
- ProjectExperience: Past projects, deliverables, outcomes
- Reference: Client/partner references with contact details
- Capability: Organizational capabilities and certifications

These models are populated by TenderDataExtractor and stored in structured_knowledge.json
for fast retrieval during tender field matching.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from enum import Enum


class InsuranceType(str, Enum):
    """Types of insurance commonly required in tenders."""
    PUBLIC_LIABILITY = "Public Liability"
    PROFESSIONAL_INDEMNITY = "Professional Indemnity"
    WORKERS_COMPENSATION = "Workers Compensation"
    CYBER_LIABILITY = "Cyber Liability"
    PRODUCT_LIABILITY = "Product Liability"
    OTHER = "Other"


class QualificationType(str, Enum):
    """Types of qualifications."""
    CERTIFICATION = "Certification"
    DEGREE = "Degree"
    DIPLOMA = "Diploma"
    LICENSE = "License"
    MEMBERSHIP = "Professional Membership"
    OTHER = "Other"


class OrganizationProfile(BaseModel):
    """Core organizational identity and contact information."""

    legal_name: str = Field(description="Official registered business name")
    trading_names: List[str] = Field(default_factory=list, description="Trading names or DBA names")

    # Legal identifiers
    abn: Optional[str] = Field(None, description="Australian Business Number (11 digits)")
    acn: Optional[str] = Field(None, description="Australian Company Number (9 digits)")

    # Address
    address: Dict[str, str] = Field(
        default_factory=dict,
        description="Address components: street, city, state, postcode, country"
    )
    postal_address: Optional[Dict[str, str]] = Field(None, description="Postal address if different")

    # Contact
    phone: Optional[str] = Field(None, description="Primary phone number")
    email: Optional[str] = Field(None, description="Primary email address")
    website: Optional[str] = Field(None, description="Organization website")

    # Metadata
    last_updated: Optional[datetime] = Field(None, description="When this data was last extracted")
    source_documents: List[str] = Field(default_factory=list, description="KB documents this was extracted from")

    @field_validator('abn')
    @classmethod
    def validate_abn(cls, v):
        """Validate ABN format (11 digits)."""
        if v and v.replace(' ', '').isdigit():
            digits = v.replace(' ', '')
            if len(digits) == 11:
                return digits
        return v

    @field_validator('acn')
    @classmethod
    def validate_acn(cls, v):
        """Validate ACN format (9 digits)."""
        if v and v.replace(' ', '').isdigit():
            digits = v.replace(' ', '')
            if len(digits) == 9:
                return digits
        return v


class Insurance(BaseModel):
    """Insurance policy details."""

    insurance_type: InsuranceType = Field(description="Type of insurance coverage")
    insurer: str = Field(description="Insurance company name")
    policy_number: str = Field(description="Policy number")

    coverage_amount: Optional[float] = Field(None, description="Coverage limit in AUD")
    coverage_description: Optional[str] = Field(None, description="What is covered")

    effective_date: Optional[date] = Field(None, description="Policy start date")
    expiry_date: Optional[date] = Field(None, description="Policy expiry date")

    # Metadata
    last_updated: Optional[datetime] = Field(None, description="When this data was last extracted")
    source_documents: List[str] = Field(default_factory=list, description="KB documents this was extracted from")

    @property
    def is_expired(self) -> bool:
        """Check if policy has expired."""
        if self.expiry_date:
            return date.today() > self.expiry_date
        return False

    @property
    def days_until_expiry(self) -> Optional[int]:
        """Calculate days until expiry."""
        if self.expiry_date:
            return (self.expiry_date - date.today()).days
        return None


class Qualification(BaseModel):
    """Individual qualification, certification, or professional membership."""

    person_name: str = Field(description="Name of person holding qualification")
    qualification_name: str = Field(description="Name of qualification/certification")
    qualification_type: QualificationType = Field(description="Type of qualification")

    institution: Optional[str] = Field(None, description="Issuing institution/body")
    date_obtained: Optional[date] = Field(None, description="Date qualification was obtained")
    expiry_date: Optional[date] = Field(None, description="Expiry date if applicable")

    credential_id: Optional[str] = Field(None, description="Credential/certificate number")
    description: Optional[str] = Field(None, description="Additional details")

    # Metadata
    last_updated: Optional[datetime] = Field(None, description="When this data was last extracted")
    source_documents: List[str] = Field(default_factory=list, description="KB documents this was extracted from")

    @property
    def is_expired(self) -> bool:
        """Check if qualification has expired."""
        if self.expiry_date:
            return date.today() > self.expiry_date
        return False


class WorkExperience(BaseModel):
    """Individual work experience entry."""

    person_name: str = Field(description="Name of person")
    role: str = Field(description="Job title/role")
    organization: str = Field(description="Employer/organization name")

    start_date: Optional[date] = Field(None, description="Start date")
    end_date: Optional[date] = Field(None, description="End date (None if current)")

    responsibilities: List[str] = Field(default_factory=list, description="Key responsibilities")
    achievements: List[str] = Field(default_factory=list, description="Notable achievements")
    technologies: List[str] = Field(default_factory=list, description="Technologies/tools used")

    # Metadata
    last_updated: Optional[datetime] = Field(None, description="When this data was last extracted")
    source_documents: List[str] = Field(default_factory=list, description="KB documents this was extracted from")

    @property
    def is_current(self) -> bool:
        """Check if this is current employment."""
        return self.end_date is None

    @property
    def duration_years(self) -> Optional[float]:
        """Calculate duration in years."""
        if self.start_date:
            end = self.end_date or date.today()
            days = (end - self.start_date).days
            return round(days / 365.25, 1)
        return None


class ProjectExperience(BaseModel):
    """Past project experience."""

    project_name: str = Field(description="Name of project")
    client: str = Field(description="Client/organization name")

    start_date: Optional[date] = Field(None, description="Project start date")
    end_date: Optional[date] = Field(None, description="Project end date (None if ongoing)")

    description: str = Field(description="Project description")
    role: Optional[str] = Field(None, description="Your organization's role in the project")
    value: Optional[float] = Field(None, description="Project value in AUD")

    deliverables: List[str] = Field(default_factory=list, description="Key deliverables")
    outcomes: List[str] = Field(default_factory=list, description="Project outcomes/benefits")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")

    team_size: Optional[int] = Field(None, description="Team size")

    # Metadata
    last_updated: Optional[datetime] = Field(None, description="When this data was last extracted")
    source_documents: List[str] = Field(default_factory=list, description="KB documents this was extracted from")

    @property
    def is_ongoing(self) -> bool:
        """Check if project is ongoing."""
        return self.end_date is None

    @property
    def duration_months(self) -> Optional[int]:
        """Calculate duration in months."""
        if self.start_date:
            end = self.end_date or date.today()
            days = (end - self.start_date).days
            return round(days / 30.44)
        return None


class Reference(BaseModel):
    """Client/partner reference."""

    contact_name: str = Field(description="Reference contact person name")
    contact_title: Optional[str] = Field(None, description="Contact's job title")

    organization: str = Field(description="Organization name")
    phone: Optional[str] = Field(None, description="Contact phone number")
    email: Optional[str] = Field(None, description="Contact email")

    relationship: str = Field(description="Nature of relationship (e.g., 'Client - Project Manager')")
    project_context: Optional[str] = Field(None, description="Which project(s) they can speak to")

    reference_date: Optional[date] = Field(None, description="When reference was obtained")

    # Metadata
    last_updated: Optional[datetime] = Field(None, description="When this data was last extracted")
    source_documents: List[str] = Field(default_factory=list, description="KB documents this was extracted from")


class Capability(BaseModel):
    """Organizational capability, certification, or accreditation."""

    capability_name: str = Field(description="Name of capability/certification")
    description: str = Field(description="What this capability enables")

    certification_body: Optional[str] = Field(None, description="Certifying organization")
    certification_number: Optional[str] = Field(None, description="Certificate/registration number")

    date_obtained: Optional[date] = Field(None, description="Date obtained")
    expiry_date: Optional[date] = Field(None, description="Expiry date if applicable")

    scope: Optional[str] = Field(None, description="Scope of certification")
    evidence: List[str] = Field(default_factory=list, description="Evidence documents")

    # Metadata
    last_updated: Optional[datetime] = Field(None, description="When this data was last extracted")
    source_documents: List[str] = Field(default_factory=list, description="KB documents this was extracted from")

    @property
    def is_expired(self) -> bool:
        """Check if capability certification has expired."""
        if self.expiry_date:
            return date.today() > self.expiry_date
        return False


class StructuredKnowledge(BaseModel):
    """
    Complete structured knowledge extracted from unstructured KB.
    Stored in structured_knowledge.json for fast retrieval.
    """

    # Core data
    organization: Optional[OrganizationProfile] = None
    insurances: List[Insurance] = Field(default_factory=list)
    team_qualifications: List[Qualification] = Field(default_factory=list)
    team_work_experience: List[WorkExperience] = Field(default_factory=list)
    projects: List[ProjectExperience] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)
    capabilities: List[Capability] = Field(default_factory=list)

    # Extraction metadata
    extraction_date: datetime = Field(default_factory=datetime.now)
    kb_version: Optional[str] = Field(None, description="Version/state of KB when extracted")
    total_documents_processed: int = Field(0, description="Number of KB documents processed")

    # Statistics
    @property
    def summary_stats(self) -> Dict[str, int]:
        """Get summary statistics."""
        return {
            "insurances": len(self.insurances),
            "qualifications": len(self.team_qualifications),
            "work_experiences": len(self.team_work_experience),
            "projects": len(self.projects),
            "references": len(self.references),
            "capabilities": len(self.capabilities)
        }

    def get_active_insurances(self) -> List[Insurance]:
        """Get non-expired insurance policies."""
        return [ins for ins in self.insurances if not ins.is_expired]

    def get_person_qualifications(self, person_name: str) -> List[Qualification]:
        """Get all qualifications for a specific person."""
        return [q for q in self.team_qualifications if person_name.lower() in q.person_name.lower()]

    def get_person_experience(self, person_name: str) -> List[WorkExperience]:
        """Get all work experience for a specific person."""
        return [exp for exp in self.team_work_experience if person_name.lower() in exp.person_name.lower()]

    def get_recent_projects(self, limit: int = 10) -> List[ProjectExperience]:
        """Get most recent projects."""
        sorted_projects = sorted(
            self.projects,
            key=lambda p: p.start_date or date.min,
            reverse=True
        )
        return sorted_projects[:limit]

    def to_json_serializable(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "organization": self.organization.model_dump() if self.organization else None,
            "insurances": [ins.model_dump() for ins in self.insurances],
            "team_qualifications": [q.model_dump() for q in self.team_qualifications],
            "team_work_experience": [exp.model_dump() for exp in self.team_work_experience],
            "projects": [p.model_dump() for p in self.projects],
            "references": [r.model_dump() for r in self.references],
            "capabilities": [c.model_dump() for c in self.capabilities],
            "extraction_date": self.extraction_date.isoformat(),
            "kb_version": self.kb_version,
            "total_documents_processed": self.total_documents_processed
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "StructuredKnowledge":
        """Load from JSON dict."""
        # Parse datetime fields
        if 'extraction_date' in data and isinstance(data['extraction_date'], str):
            data['extraction_date'] = datetime.fromisoformat(data['extraction_date'])

        return cls(**data)

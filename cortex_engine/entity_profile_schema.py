"""
Entity Profile Schema - Pydantic Models
Version: 1.0.0
Date: 2026-01-05

Purpose: Define Pydantic models for entity profiles used in mention-based proposal system.
All entity data is stored in YAML files and validated against these schemas.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import date, datetime
from enum import Enum
from pathlib import Path


# ============================================
# ENUMS
# ============================================

class EntityType(str, Enum):
    """Type of entity."""
    CONSULTING_FIRM = "consulting_firm"
    CONTRACTOR = "contractor"
    CONSORTIUM = "consortium"
    PARTNERSHIP = "partnership"
    SOLE_TRADER = "sole_trader"
    OTHER = "other"


class EntityStatus(str, Enum):
    """Status of entity profile."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DRAFT = "draft"


class PhoneFormat(str, Enum):
    """Phone number formatting preference."""
    INTERNATIONAL = "international"
    LOCAL = "local"


class DateFormat(str, Enum):
    """Date formatting preference."""
    ISO8601 = "ISO8601"
    DD_MM_YYYY = "DD/MM/YYYY"
    MM_DD_YYYY = "MM/DD/YYYY"
    YYYY_MM_DD = "YYYY-MM-DD"


class CVFormat(str, Enum):
    """CV generation format."""
    PROFESSIONAL = "professional"
    ACADEMIC = "academic"
    BRIEF = "brief"
    DETAILED = "detailed"


class RelationshipType(str, Enum):
    """Type of relationship with reference."""
    CLIENT = "client"
    PARTNER = "partner"
    SUBCONTRACTOR = "subcontractor"
    SUPPLIER = "supplier"
    PEER = "peer"


class InsuranceType(str, Enum):
    """Types of insurance policies."""
    PUBLIC_LIABILITY = "public_liability"
    PROFESSIONAL_INDEMNITY = "professional_indemnity"
    WORKERS_COMPENSATION = "workers_compensation"
    CYBER_LIABILITY = "cyber_liability"
    PRODUCT_LIABILITY = "product_liability"
    ENVIRONMENTAL = "environmental"
    OTHER = "other"


class RenewalStatus(str, Enum):
    """Insurance renewal status."""
    CURRENT = "current"
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"
    PENDING_RENEWAL = "pending_renewal"


class CapabilityType(str, Enum):
    """Type of capability/certification."""
    CERTIFICATION = "certification"
    ACCREDITATION = "accreditation"
    MEMBERSHIP = "membership"
    LICENSE = "license"
    AWARD = "award"


# ============================================
# NESTED MODELS
# ============================================

class Address(BaseModel):
    """Physical address."""
    street: str = Field(description="Street address")
    city: str = Field(description="City/suburb")
    state: str = Field(description="State/province")
    postcode: str = Field(description="Postcode/ZIP")
    country: str = Field(default="Australia", description="Country")

    def formatted(self, single_line: bool = False) -> str:
        """Return formatted address string."""
        if single_line:
            return f"{self.street}, {self.city} {self.state} {self.postcode}, {self.country}"
        else:
            return f"{self.street}\n{self.city} {self.state} {self.postcode}\n{self.country}"


class CompanyInfo(BaseModel):
    """Company legal and trading information."""
    legal_name: str = Field(description="Official registered business name")
    trading_names: List[str] = Field(default_factory=list, description="Trading names / DBAs")
    abn: Optional[str] = Field(None, description="Australian Business Number (11 digits)")
    acn: Optional[str] = Field(None, description="Australian Company Number (9 digits)")
    registration_date: Optional[date] = Field(None, description="Company registration date")

    @field_validator('abn')
    @classmethod
    def validate_abn(cls, v: Optional[str]) -> Optional[str]:
        """Validate ABN format (11 digits)."""
        if v:
            digits = ''.join(filter(str.isdigit, v))
            if len(digits) != 11:
                raise ValueError(f"ABN must be 11 digits, got {len(digits)}")
            return digits
        return v

    @field_validator('acn')
    @classmethod
    def validate_acn(cls, v: Optional[str]) -> Optional[str]:
        """Validate ACN format (9 digits)."""
        if v:
            digits = ''.join(filter(str.isdigit, v))
            if len(digits) != 9:
                raise ValueError(f"ACN must be 9 digits, got {len(digits)}")
            return digits
        return v


class ContactInfo(BaseModel):
    """Contact information."""
    registered_office: Address = Field(description="Registered office address")
    postal_address: Optional[Address] = Field(None, description="Postal address if different")
    phone: str = Field(description="Primary phone number")
    email: str = Field(description="Primary email address")
    website: Optional[str] = Field(None, description="Website URL")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation."""
        if '@' not in v or '.' not in v.split('@')[1]:
            raise ValueError(f"Invalid email address: {v}")
        return v.lower()


class FormattingPreferences(BaseModel):
    """Formatting preferences for output."""
    abn_format: str = Field(default="XX XXX XXX XXX", description="ABN formatting pattern")
    acn_format: str = Field(default="XXX XXX XXX", description="ACN formatting pattern")
    phone_format: PhoneFormat = Field(default=PhoneFormat.INTERNATIONAL)
    date_format: DateFormat = Field(default=DateFormat.DD_MM_YYYY)
    currency_format: str = Field(default="AUD", description="Currency code")


class ProfileMetadata(BaseModel):
    """Metadata for entity profile."""
    entity_id: str = Field(description="Unique entity ID (URL-safe)")
    entity_name: str = Field(description="Display name for entity")
    entity_type: EntityType = Field(description="Type of entity")
    created_date: date = Field(default_factory=date.today)
    last_updated: date = Field(default_factory=date.today)
    version: str = Field(default="1.0.0", description="Profile version")
    status: EntityStatus = Field(default=EntityStatus.ACTIVE)
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    @field_validator('entity_id')
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Ensure entity_id is URL-safe (lowercase, underscores)."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Entity ID must be alphanumeric with underscores/hyphens: {v}")
        return v.lower()


# ============================================
# TEAM MEMBER MODELS
# ============================================

class Qualification(BaseModel):
    """Educational qualification or certification."""
    name: str = Field(description="Qualification name")
    institution: str = Field(description="Issuing institution")
    year: int = Field(description="Year obtained")
    specialization: Optional[str] = Field(None, description="Area of specialization")
    credential_id: Optional[str] = Field(None, description="Credential/certificate number")


class Experience(BaseModel):
    """Work experience entry."""
    role: str = Field(description="Job title/role")
    organization: str = Field(description="Employer organization")
    start_date: date = Field(description="Start date")
    end_date: Optional[date] = Field(None, description="End date (null = current)")
    location: Optional[str] = Field(None, description="Work location")
    responsibilities: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)

    @property
    def is_current(self) -> bool:
        """Check if this is current employment."""
        return self.end_date is None

    @property
    def duration_years(self) -> float:
        """Calculate duration in years."""
        end = self.end_date or date.today()
        days = (end - self.start_date).days
        return round(days / 365.25, 1)


class Certification(BaseModel):
    """Professional certification."""
    name: str = Field(description="Certification name")
    issuer: str = Field(description="Issuing body")
    date_obtained: date = Field(description="Date obtained")
    expiry: Optional[date] = Field(None, description="Expiry date")
    credential_id: Optional[str] = Field(None, description="Credential number")

    @property
    def is_expired(self) -> bool:
        """Check if certification has expired."""
        if self.expiry:
            return date.today() > self.expiry
        return False


class Bio(BaseModel):
    """Biographical information."""
    brief: str = Field(description="Brief bio (2-3 sentences)")
    full: str = Field(description="Full bio (multiple paragraphs)")


class GenerationPreferences(BaseModel):
    """Preferences for content generation."""
    cv_format: CVFormat = Field(default=CVFormat.PROFESSIONAL)
    include_photo: bool = Field(default=False)
    max_cv_pages: int = Field(default=3, ge=1, le=10)


class TeamMember(BaseModel):
    """Team member profile."""
    person_id: str = Field(description="Unique person ID")
    full_name: str = Field(description="Full legal name")
    preferred_name: Optional[str] = Field(None, description="Preferred/nickname")
    role: str = Field(description="Current role")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")

    qualifications: List[Qualification] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)

    bio: Optional[Bio] = Field(None, description="Biographical information")
    generation_preferences: GenerationPreferences = Field(default_factory=GenerationPreferences)

    @field_validator('person_id')
    @classmethod
    def validate_person_id(cls, v: str) -> str:
        """Ensure person_id is URL-safe."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Person ID must be alphanumeric with underscores/hyphens: {v}")
        return v.lower()


# ============================================
# PROJECT MODELS
# ============================================

class Timeline(BaseModel):
    """Project timeline."""
    start_date: date
    end_date: Optional[date] = None
    duration_months: Optional[int] = Field(None, ge=1)

    @model_validator(mode='after')
    def calculate_duration(self):
        """Calculate duration if not provided."""
        if self.duration_months is None and self.end_date:
            days = (self.end_date - self.start_date).days
            self.duration_months = max(1, round(days / 30.44))
        return self


class Financials(BaseModel):
    """Project financial information."""
    contract_value: float = Field(ge=0, description="Contract value")
    currency: str = Field(default="AUD")
    payment_structure: Optional[str] = Field(None, description="e.g., milestone_based, monthly")


class TeamRole(BaseModel):
    """Team member role in project."""
    role: str = Field(description="Role title")
    person: str = Field(description="Person ID from team/")


class ProjectTeam(BaseModel):
    """Project team composition."""
    size: int = Field(ge=1, description="Total team size")
    our_staff: List[str] = Field(default_factory=list, description="List of person IDs")
    roles: List[TeamRole] = Field(default_factory=list)


class Deliverable(BaseModel):
    """Project deliverable."""
    name: str
    description: str


class Outcome(BaseModel):
    """Project outcome/benefit."""
    metric: str = Field(description="What was measured")
    improvement: str = Field(description="Improvement achieved")
    measurement: str = Field(description="How it was measured")


class ProjectReference(BaseModel):
    """Reference for this project."""
    contact: str = Field(description="Reference ID from references/")
    available: bool = Field(default=True)
    confidential: bool = Field(default=False)


class ProjectDescription(BaseModel):
    """Project description text."""
    brief: str = Field(description="Brief summary (1-2 sentences)")
    full: str = Field(description="Full description (multiple paragraphs)")


class ProjectGenerationPreferences(BaseModel):
    """Preferences for project content generation."""
    summary_length: int = Field(default=200, ge=50, le=1000, description="Words for summary")
    focus_areas: List[str] = Field(
        default_factory=lambda: ["outcomes", "scale", "complexity"],
        description="What to emphasize"
    )


class Project(BaseModel):
    """Project experience."""
    project_id: str = Field(description="Unique project ID")
    project_name: str = Field(description="Project name")
    client: str = Field(description="Client organization")
    sector: Optional[str] = Field(None, description="e.g., government, private, nfp")
    project_type: Optional[str] = Field(None, description="e.g., digital_transformation")

    timeline: Timeline
    financials: Financials
    team: ProjectTeam

    description: ProjectDescription
    deliverables: List[Deliverable] = Field(default_factory=list)
    outcomes: List[Outcome] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    challenges_overcome: List[str] = Field(default_factory=list)

    reference: Optional[ProjectReference] = None
    generation_preferences: ProjectGenerationPreferences = Field(default_factory=ProjectGenerationPreferences)

    @field_validator('project_id')
    @classmethod
    def validate_project_id(cls, v: str) -> str:
        """Ensure project_id is URL-safe."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Project ID must be alphanumeric with underscores/hyphens: {v}")
        return v.lower()


# ============================================
# REFERENCE MODELS
# ============================================

class ReferenceRelationship(BaseModel):
    """Relationship details with reference."""
    type: RelationshipType
    role: str = Field(description="Their role (e.g., Project Sponsor)")
    projects: List[str] = Field(default_factory=list, description="Project IDs they can speak to")


class ReferenceAvailability(BaseModel):
    """Reference availability information."""
    available: bool = Field(default=True)
    preferred_contact: Literal["email", "phone"] = Field(default="email")
    best_times: Optional[str] = Field(None, description="e.g., business_hours")
    notes: Optional[str] = Field(None)


class ReferenceContext(BaseModel):
    """Context about working relationship."""
    working_relationship: str = Field(description="Summary of working relationship")
    can_speak_to: List[str] = Field(default_factory=list, description="What they can speak to")


class ReferenceConfidentiality(BaseModel):
    """Confidentiality settings."""
    public: bool = Field(default=True, description="Can list in public proposals")
    name_only: bool = Field(default=False, description="Only show name/org, not contact details")
    on_request: bool = Field(default=False, description="Provide only upon request")


class Reference(BaseModel):
    """Client/partner reference."""
    reference_id: str = Field(description="Unique reference ID")
    contact_name: str = Field(description="Contact person name")
    title: Optional[str] = Field(None, description="Job title")
    organization: str = Field(description="Organization name")
    email: Optional[str] = Field(None)
    phone: Optional[str] = Field(None)

    relationship: ReferenceRelationship
    availability: ReferenceAvailability = Field(default_factory=ReferenceAvailability)
    context: Optional[ReferenceContext] = None
    confidentiality: ReferenceConfidentiality = Field(default_factory=ReferenceConfidentiality)

    quote: Optional[str] = Field(None, description="Pre-approved testimonial quote")

    @field_validator('reference_id')
    @classmethod
    def validate_reference_id(cls, v: str) -> str:
        """Ensure reference_id is URL-safe."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Reference ID must be alphanumeric with underscores/hyphens: {v}")
        return v.lower()


# ============================================
# INSURANCE MODELS
# ============================================

class Coverage(BaseModel):
    """Insurance coverage details."""
    amount: float = Field(ge=0, description="Coverage amount")
    currency: str = Field(default="AUD")
    formatted: str = Field(description="Formatted display (e.g., $20,000,000)")
    description: str = Field(description="What is covered")


class InsuranceDates(BaseModel):
    """Insurance policy dates."""
    effective_date: date
    expiry_date: date
    renewal_status: RenewalStatus = Field(default=RenewalStatus.CURRENT)

    @property
    def days_until_expiry(self) -> int:
        """Days until policy expires."""
        return (self.expiry_date - date.today()).days

    @property
    def is_expired(self) -> bool:
        """Check if policy has expired."""
        return date.today() > self.expiry_date


class InsuranceScope(BaseModel):
    """Insurance policy scope."""
    geographic: str = Field(description="Geographic coverage area")
    activities: List[str] = Field(default_factory=list, description="Covered activities")
    exclusions: List[str] = Field(default_factory=list, description="Excluded activities")


class InsuranceBroker(BaseModel):
    """Insurance broker details."""
    name: str
    contact: str
    phone: str
    email: str


class InsuranceDocuments(BaseModel):
    """Insurance document references."""
    certificate_path: Optional[str] = None
    policy_document_path: Optional[str] = None


class Insurance(BaseModel):
    """Insurance policy."""
    policy_id: str = Field(description="Unique policy ID")
    policy_type: InsuranceType
    insurer: str = Field(description="Insurance company")
    policy_number: str = Field(description="Policy number")

    coverage: Coverage
    dates: InsuranceDates
    scope: InsuranceScope
    broker: Optional[InsuranceBroker] = None
    documents: InsuranceDocuments = Field(default_factory=InsuranceDocuments)

    @field_validator('policy_id')
    @classmethod
    def validate_policy_id(cls, v: str) -> str:
        """Ensure policy_id is URL-safe."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Policy ID must be alphanumeric with underscores/hyphens: {v}")
        return v.lower()


# ============================================
# CAPABILITY MODELS
# ============================================

class CapabilityDescription(BaseModel):
    """Capability description."""
    brief: str = Field(description="Brief description (1 sentence)")
    full: str = Field(description="Full description (multiple paragraphs)")


class CapabilityDates(BaseModel):
    """Capability/certification dates."""
    obtained: date
    expiry: Optional[date] = None
    next_audit: Optional[date] = None

    @property
    def is_expired(self) -> bool:
        """Check if capability has expired."""
        if self.expiry:
            return date.today() > self.expiry
        return False


class CapabilityScope(BaseModel):
    """Scope of capability/certification."""
    services: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    standards: List[str] = Field(default_factory=list)


class CapabilityEvidence(BaseModel):
    """Evidence documents."""
    certificate_path: Optional[str] = None
    audit_reports: List[str] = Field(default_factory=list)


class Capability(BaseModel):
    """Organizational capability/certification."""
    capability_id: str = Field(description="Unique capability ID")
    capability_name: str = Field(description="Capability/certification name")
    capability_type: CapabilityType

    description: CapabilityDescription
    certification_body: Optional[str] = Field(None, description="Certifying organization")
    certification_number: Optional[str] = Field(None)

    dates: CapabilityDates
    scope: CapabilityScope = Field(default_factory=CapabilityScope)
    evidence: CapabilityEvidence = Field(default_factory=CapabilityEvidence)

    @field_validator('capability_id')
    @classmethod
    def validate_capability_id(cls, v: str) -> str:
        """Ensure capability_id is URL-safe."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Capability ID must be alphanumeric with underscores/hyphens: {v}")
        return v.lower()


# ============================================
# MAIN ENTITY PROFILE
# ============================================

class EntityProfile(BaseModel):
    """Complete entity profile."""
    metadata: ProfileMetadata
    company: CompanyInfo
    contact: ContactInfo

    # References to other files (IDs only)
    team: List[str] = Field(default_factory=list, description="List of person IDs")
    projects: List[str] = Field(default_factory=list, description="List of project IDs")
    references: List[str] = Field(default_factory=list, description="List of reference IDs")
    capabilities: List[str] = Field(default_factory=list, description="List of capability IDs")
    insurance: List[str] = Field(default_factory=list, description="List of insurance policy IDs")

    # Formatting and display preferences
    formatting: FormattingPreferences = Field(default_factory=FormattingPreferences)
    narrative_sections: List[str] = Field(
        default_factory=lambda: [
            "company_overview",
            "core_capabilities",
            "competitive_advantages",
            "quality_assurance"
        ],
        description="Sections available in narrative.md"
    )

    def get_directory_path(self, base_path: Path) -> Path:
        """Get the directory path for this entity profile."""
        return base_path / "entity_profiles" / self.metadata.entity_id

    def format_abn(self) -> str:
        """Format ABN according to preferences."""
        if not self.company.abn:
            return ""

        abn = self.company.abn
        pattern = self.formatting.abn_format

        # Simple XX XXX XXX XXX formatting
        if pattern == "XX XXX XXX XXX" and len(abn) == 11:
            return f"{abn[0:2]} {abn[2:5]} {abn[5:8]} {abn[8:11]}"

        return abn

    def format_acn(self) -> str:
        """Format ACN according to preferences."""
        if not self.company.acn:
            return ""

        acn = self.company.acn
        pattern = self.formatting.acn_format

        # Simple XXX XXX XXX formatting
        if pattern == "XXX XXX XXX" and len(acn) == 9:
            return f"{acn[0:3]} {acn[3:6]} {acn[6:9]}"

        return acn

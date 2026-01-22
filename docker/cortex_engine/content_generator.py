"""
Content Generator
Version: 1.0.0
Date: 2026-01-05

Purpose: Generate content for @mentions requiring LLM (CVs, project summaries, references, etc.)
"""

from typing import Dict, Any
from .entity_profile_manager import EntityProfileManager
from .llm_interface import LLMInterface
from .workspace_model import MentionBinding
from .utils import get_logger

logger = get_logger(__name__)


class ContentGenerator:
    """Generate content for complex @mentions using LLM."""

    def __init__(
        self,
        entity_manager: EntityProfileManager,
        llm: LLMInterface
    ):
        """
        Initialize content generator.

        Args:
            entity_manager: Entity profile manager
            llm: LLM interface
        """
        self.entity_manager = entity_manager
        self.llm = llm

        logger.info("ContentGenerator initialized")

    def generate_content(
        self,
        mention: MentionBinding,
        entity_id: str,
        generation_type: str
    ) -> str:
        """
        Generate content for a mention.

        Args:
            mention: Mention binding
            entity_id: Entity ID
            generation_type: Type of generation (cv, project_summary, reference, etc.)

        Returns:
            Generated content
        """
        if generation_type == "cv":
            return self._generate_cv(mention, entity_id)
        elif generation_type == "project_summary":
            return self._generate_project_summary(mention, entity_id)
        elif generation_type == "reference":
            return self._generate_reference(mention, entity_id)
        else:
            raise ValueError(f"Unknown generation type: {generation_type}")

    def _generate_cv(
        self,
        mention: MentionBinding,
        entity_id: str
    ) -> str:
        """Generate CV/resume for team member."""
        # Extract person_id from mention
        person_id = mention.mention_text.split('[')[1].split(']')[0].split(',')[0].strip()

        # Get team member data
        team_member = self.entity_manager.get_team_member(entity_id, person_id)

        if not team_member:
            raise ValueError(f"Team member not found: {person_id}")

        # Build context
        context = {
            "name": team_member.full_name,
            "role": team_member.role,
            "email": team_member.email or "",
            "phone": team_member.phone or "",
        }

        # Qualifications
        quals = []
        for qual in team_member.qualifications:
            q = f"{qual.name}, {qual.institution}"
            if qual.year:
                q += f" ({qual.year})"
            if qual.specialization:
                q += f" - {qual.specialization}"
            quals.append(q)
        context["qualifications"] = "\n- ".join(quals) if quals else "None listed"

        # Experience
        exp = []
        for e in team_member.experience:
            exp_str = f"{e.role} at {e.organization}"
            if e.start_date:
                exp_str += f" ({e.start_date.year}"
                if e.end_date:
                    exp_str += f"-{e.end_date.year})"
                else:
                    exp_str += "-Present)"

            if e.responsibilities:
                exp_str += "\n  Responsibilities:\n  " + "\n  ".join([f"• {r}" for r in e.responsibilities[:3]])

            if e.achievements:
                exp_str += "\n  Key Achievements:\n  " + "\n  ".join([f"• {a}" for a in e.achievements[:3]])

            exp.append(exp_str)

        context["experience"] = "\n\n".join(exp) if exp else "No experience listed"

        # Bio
        context["bio"] = team_member.bio.full if team_member.bio and team_member.bio.full else team_member.bio.brief if team_member.bio else ""

        # Generate CV
        system_prompt = """You are a professional CV writer specializing in tender responses and government proposals.
Generate a concise, professional CV summary suitable for inclusion in a tender document.
Focus on relevant experience, qualifications, and achievements.
Write in third person. Be specific and quantify achievements where possible.
Keep it concise (300-500 words) but impactful."""

        prompt = f"""Generate a professional CV summary for inclusion in a tender response document.

**Person Details:**
Name: {context['name']}
Current Role: {context['role']}

**Qualifications:**
{context['qualifications']}

**Professional Experience:**
{context['experience']}

**Professional Summary:**
{context['bio']}

Write a concise, compelling CV summary (300-500 words) that highlights this person's suitability for a government consulting engagement."""

        return self.llm.generate(prompt, system_prompt)

    def _generate_project_summary(
        self,
        mention: MentionBinding,
        entity_id: str
    ) -> str:
        """Generate project summary."""
        # Extract project_id from mention
        project_id = mention.mention_text.split('[')[1].split(']')[0].split(',')[0].strip()

        # Get project data
        project = self.entity_manager.get_project(entity_id, project_id)

        if not project:
            raise ValueError(f"Project not found: {project_id}")

        # Build context
        context = {
            "name": project.project_name,
            "client": project.client,
            "sector": project.sector or "Not specified",
            "start_date": project.timeline.start_date.strftime("%B %Y"),
            "end_date": project.timeline.end_date.strftime("%B %Y") if project.timeline.end_date else "Ongoing",
            "duration": f"{project.timeline.duration_months} months" if project.timeline.duration_months else "Not specified",
            "value": f"${project.financials.contract_value:,.0f} {project.financials.currency}",
            "description": project.description.full if project.description.full else project.description.brief,
        }

        # Deliverables
        if project.deliverables:
            context["deliverables"] = "\n- ".join([f"{d.name}: {d.description}" for d in project.deliverables])
        else:
            context["deliverables"] = "Not specified"

        # Outcomes
        if project.outcomes:
            context["outcomes"] = "\n- ".join([
                f"{o.metric}: {o.improvement}" + (f" ({o.measurement})" if o.measurement else "")
                for o in project.outcomes
            ])
        else:
            context["outcomes"] = "Not specified"

        # Generate summary
        system_prompt = """You are a professional proposal writer specializing in tender responses.
Generate a concise, compelling project summary suitable for inclusion in a tender document.
Focus on outcomes, value delivered, and relevance to similar projects.
Write in past tense for completed projects, present tense for ongoing projects.
Be specific and quantify results where possible.
Keep it concise (200-400 words) but demonstrate clear value."""

        prompt = f"""Generate a professional project summary for inclusion in a tender response document.

**Project Details:**
Project Name: {context['name']}
Client: {context['client']}
Sector: {context['sector']}
Duration: {context['start_date']} to {context['end_date']} ({context['duration']})
Contract Value: {context['value']}

**Project Description:**
{context['description']}

**Key Deliverables:**
{context['deliverables']}

**Outcomes Achieved:**
{context['outcomes']}

Write a concise, compelling project summary (200-400 words) that demonstrates our capability to deliver similar projects."""

        return self.llm.generate(prompt, system_prompt)

    def _generate_reference(
        self,
        mention: MentionBinding,
        entity_id: str
    ) -> str:
        """Generate reference/testimonial."""
        # Extract reference_id from mention
        reference_id = mention.mention_text.split('[')[1].split(']')[0].split(',')[0].strip()

        # Get reference data
        reference = self.entity_manager.get_reference(entity_id, reference_id)

        if not reference:
            raise ValueError(f"Reference not found: {reference_id}")

        # Build context
        context = {
            "name": reference.contact_name,
            "title": reference.title,
            "organization": reference.organization,
            "email": reference.email or "Available on request",
            "phone": reference.phone or "Available on request",
            "relationship": f"{reference.relationship.type.value.title()} - {reference.relationship.role}",
            "working_relationship": reference.context.working_relationship if reference.context else "",
            "quote": reference.quote or "",
        }

        # Can speak to
        if reference.context and reference.context.can_speak_to:
            context["can_speak_to"] = "\n- ".join(reference.context.can_speak_to)
        else:
            context["can_speak_to"] = "General project delivery"

        # Generate formatted reference
        system_prompt = """You are a professional proposal writer formatting client references for tender documents.
Create a well-formatted, professional reference entry suitable for a tender response.
Include all contact details, relationship context, and testimonials.
Be professional and factual.
Format clearly with appropriate headings and structure."""

        prompt = f"""Format a professional client reference for inclusion in a tender response document.

**Reference Details:**
Contact: {context['name']}
Title: {context['title']}
Organization: {context['organization']}
Email: {context['email']}
Phone: {context['phone']}

**Relationship:**
{context['relationship']}

**Working Relationship:**
{context['working_relationship']}

**Can Speak To:**
{context['can_speak_to']}

**Testimonial:**
{context['quote']}

Create a well-formatted reference section (150-250 words) that presents this information professionally and compellingly."""

        return self.llm.generate(prompt, system_prompt)

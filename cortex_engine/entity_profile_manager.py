"""
Entity Profile Manager - CRUD Operations
Version: 1.0.0
Date: 2026-01-05

Purpose: Manage entity profiles stored in YAML files.
Provides create, read, update, delete operations for:
- Entity profiles
- Team members
- Projects
- References
- Insurance policies
- Capabilities
- Narrative content
"""

import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import date, datetime

from .entity_profile_schema import (
    EntityProfile,
    TeamMember,
    Project,
    Reference,
    Insurance,
    Capability,
    ProfileMetadata,
    EntityType,
    EntityStatus
)
from .utils import get_logger

logger = get_logger(__name__)


class EntityProfileManager:
    """Manages entity profiles and all their components."""

    def __init__(self, base_path: Path):
        """
        Initialize manager.

        Args:
            base_path: Base path to ai_databases directory
        """
        self.base_path = Path(base_path)
        self.profiles_dir = self.base_path / "entity_profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"EntityProfileManager initialized at {self.profiles_dir}")

    # ========================================
    # ENTITY PROFILE OPERATIONS
    # ========================================

    def create_entity_profile(
        self,
        entity_id: str,
        entity_name: str,
        entity_type: EntityType,
        legal_name: str,
        abn: Optional[str] = None,
        acn: Optional[str] = None,
        **kwargs
    ) -> EntityProfile:
        """
        Create a new entity profile.

        Args:
            entity_id: Unique entity ID (URL-safe)
            entity_name: Display name
            entity_type: Type of entity
            legal_name: Legal company name
            abn: Australian Business Number
            acn: Australian Company Number
            **kwargs: Additional profile fields

        Returns:
            Created EntityProfile

        Raises:
            ValueError: If entity already exists
        """
        entity_dir = self.profiles_dir / entity_id

        if entity_dir.exists():
            raise ValueError(f"Entity profile already exists: {entity_id}")

        # Create directory structure
        entity_dir.mkdir(parents=True, exist_ok=True)
        (entity_dir / "team").mkdir(exist_ok=True)
        (entity_dir / "projects").mkdir(exist_ok=True)
        (entity_dir / "references").mkdir(exist_ok=True)
        (entity_dir / "capabilities").mkdir(exist_ok=True)
        (entity_dir / "insurance").mkdir(exist_ok=True)

        # Create minimal profile
        from .entity_profile_schema import CompanyInfo, ContactInfo, Address

        # Extract address from kwargs or create default
        address_data = kwargs.pop('address', {})
        if isinstance(address_data, dict):
            address = Address(
                street=address_data.get('street', ''),
                city=address_data.get('city', ''),
                state=address_data.get('state', ''),
                postcode=address_data.get('postcode', ''),
                country=address_data.get('country', 'Australia')
            )
        else:
            address = address_data

        profile = EntityProfile(
            metadata=ProfileMetadata(
                entity_id=entity_id,
                entity_name=entity_name,
                entity_type=entity_type
            ),
            company=CompanyInfo(
                legal_name=legal_name,
                abn=abn,
                acn=acn,
                trading_names=kwargs.get('trading_names', [])
            ),
            contact=ContactInfo(
                registered_office=address,
                phone=kwargs.get('phone', ''),
                email=kwargs.get('email', ''),
                website=kwargs.get('website')
            )
        )

        # Save to file
        self._save_profile(profile)

        # Create empty narrative.md
        narrative_path = entity_dir / "narrative.md"
        narrative_path.write_text(
            f"# {entity_name}\n\n## Company Overview\n\n[Add company overview here]\n\n## Core Capabilities\n\n[Add core capabilities here]\n"
        )

        logger.info(f"Created entity profile: {entity_id}")
        return profile

    def get_entity_profile(self, entity_id: str) -> Optional[EntityProfile]:
        """
        Load entity profile.

        Args:
            entity_id: Entity ID

        Returns:
            EntityProfile or None if not found
        """
        profile_path = self.profiles_dir / entity_id / "profile.yaml"

        if not profile_path.exists():
            logger.warning(f"Entity profile not found: {entity_id}")
            return None

        with open(profile_path, 'r') as f:
            data = yaml.safe_load(f)

        return EntityProfile(**data)

    def update_entity_profile(self, entity_id: str, updates: Dict[str, Any]) -> EntityProfile:
        """
        Update entity profile fields.

        Args:
            entity_id: Entity ID
            updates: Dict of fields to update

        Returns:
            Updated EntityProfile

        Raises:
            ValueError: If entity not found
        """
        profile = self.get_entity_profile(entity_id)
        if not profile:
            raise ValueError(f"Entity profile not found: {entity_id}")

        # Update fields
        data = profile.model_dump()
        data.update(updates)

        # Update last_updated date
        data['metadata']['last_updated'] = date.today()

        # Recreate profile with updated data
        updated_profile = EntityProfile(**data)

        # Save
        self._save_profile(updated_profile)

        logger.info(f"Updated entity profile: {entity_id}")
        return updated_profile

    def delete_entity_profile(self, entity_id: str) -> bool:
        """
        Delete entity profile and all associated data.

        Args:
            entity_id: Entity ID

        Returns:
            True if deleted, False if not found
        """
        entity_dir = self.profiles_dir / entity_id

        if not entity_dir.exists():
            logger.warning(f"Entity profile not found for deletion: {entity_id}")
            return False

        import shutil
        shutil.rmtree(entity_dir)

        logger.info(f"Deleted entity profile: {entity_id}")
        return True

    def list_entity_profiles(self) -> List[ProfileMetadata]:
        """
        List all entity profiles.

        Returns:
            List of ProfileMetadata for all entities
        """
        profiles = []

        for entity_dir in self.profiles_dir.iterdir():
            if entity_dir.is_dir():
                profile_path = entity_dir / "profile.yaml"
                if profile_path.exists():
                    with open(profile_path, 'r') as f:
                        data = yaml.safe_load(f)
                        profiles.append(ProfileMetadata(**data['metadata']))

        return sorted(profiles, key=lambda p: p.entity_name)

    def _save_profile(self, profile: EntityProfile):
        """Save profile to YAML file."""
        entity_dir = self.profiles_dir / profile.metadata.entity_id
        profile_path = entity_dir / "profile.yaml"

        # Convert to dict and save as YAML
        data = profile.model_dump(mode='json')

        with open(profile_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # ========================================
    # TEAM MEMBER OPERATIONS
    # ========================================

    def add_team_member(self, entity_id: str, team_member: TeamMember) -> bool:
        """
        Add team member to entity.

        Args:
            entity_id: Entity ID
            team_member: TeamMember object

        Returns:
            True if added successfully
        """
        entity_dir = self.profiles_dir / entity_id

        if not entity_dir.exists():
            raise ValueError(f"Entity not found: {entity_id}")

        # Save team member file
        team_path = entity_dir / "team" / f"{team_member.person_id}.yaml"
        data = team_member.model_dump(mode='json')

        with open(team_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Update profile's team list
        profile = self.get_entity_profile(entity_id)
        if team_member.person_id not in profile.team:
            profile.team.append(team_member.person_id)
            self._save_profile(profile)

        logger.info(f"Added team member {team_member.person_id} to {entity_id}")
        return True

    def get_team_member(self, entity_id: str, person_id: str) -> Optional[TeamMember]:
        """
        Get team member by ID.

        Args:
            entity_id: Entity ID
            person_id: Person ID

        Returns:
            TeamMember or None if not found
        """
        team_path = self.profiles_dir / entity_id / "team" / f"{person_id}.yaml"

        if not team_path.exists():
            return None

        with open(team_path, 'r') as f:
            data = yaml.safe_load(f)

        return TeamMember(**data)

    def list_team_members(self, entity_id: str) -> List[TeamMember]:
        """
        List all team members for entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of TeamMember objects
        """
        team_dir = self.profiles_dir / entity_id / "team"

        if not team_dir.exists():
            return []

        members = []
        for team_file in team_dir.glob("*.yaml"):
            with open(team_file, 'r') as f:
                data = yaml.safe_load(f)
                members.append(TeamMember(**data))

        return sorted(members, key=lambda m: m.full_name)

    def remove_team_member(self, entity_id: str, person_id: str) -> bool:
        """
        Remove team member from entity.

        Args:
            entity_id: Entity ID
            person_id: Person ID

        Returns:
            True if removed, False if not found
        """
        team_path = self.profiles_dir / entity_id / "team" / f"{person_id}.yaml"

        if not team_path.exists():
            return False

        team_path.unlink()

        # Update profile's team list
        profile = self.get_entity_profile(entity_id)
        if person_id in profile.team:
            profile.team.remove(person_id)
            self._save_profile(profile)

        logger.info(f"Removed team member {person_id} from {entity_id}")
        return True

    # ========================================
    # PROJECT OPERATIONS
    # ========================================

    def add_project(self, entity_id: str, project: Project) -> bool:
        """Add project to entity."""
        entity_dir = self.profiles_dir / entity_id

        if not entity_dir.exists():
            raise ValueError(f"Entity not found: {entity_id}")

        # Save project file
        project_path = entity_dir / "projects" / f"{project.project_id}.yaml"
        data = project.model_dump(mode='json')

        with open(project_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Update profile's projects list
        profile = self.get_entity_profile(entity_id)
        if project.project_id not in profile.projects:
            profile.projects.append(project.project_id)
            self._save_profile(profile)

        logger.info(f"Added project {project.project_id} to {entity_id}")
        return True

    def get_project(self, entity_id: str, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        project_path = self.profiles_dir / entity_id / "projects" / f"{project_id}.yaml"

        if not project_path.exists():
            return None

        with open(project_path, 'r') as f:
            data = yaml.safe_load(f)

        return Project(**data)

    def list_projects(self, entity_id: str) -> List[Project]:
        """List all projects for entity."""
        projects_dir = self.profiles_dir / entity_id / "projects"

        if not projects_dir.exists():
            return []

        projects = []
        for project_file in projects_dir.glob("*.yaml"):
            with open(project_file, 'r') as f:
                data = yaml.safe_load(f)
                projects.append(Project(**data))

        return sorted(projects, key=lambda p: p.timeline.start_date, reverse=True)

    def remove_project(self, entity_id: str, project_id: str) -> bool:
        """Remove project from entity."""
        project_path = self.profiles_dir / entity_id / "projects" / f"{project_id}.yaml"

        if not project_path.exists():
            return False

        project_path.unlink()

        # Update profile's projects list
        profile = self.get_entity_profile(entity_id)
        if project_id in profile.projects:
            profile.projects.remove(project_id)
            self._save_profile(profile)

        logger.info(f"Removed project {project_id} from {entity_id}")
        return True

    # ========================================
    # REFERENCE OPERATIONS
    # ========================================

    def add_reference(self, entity_id: str, reference: Reference) -> bool:
        """Add reference to entity."""
        entity_dir = self.profiles_dir / entity_id

        if not entity_dir.exists():
            raise ValueError(f"Entity not found: {entity_id}")

        # Save reference file
        ref_path = entity_dir / "references" / f"{reference.reference_id}.yaml"
        data = reference.model_dump(mode='json')

        with open(ref_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Update profile's references list
        profile = self.get_entity_profile(entity_id)
        if reference.reference_id not in profile.references:
            profile.references.append(reference.reference_id)
            self._save_profile(profile)

        logger.info(f"Added reference {reference.reference_id} to {entity_id}")
        return True

    def get_reference(self, entity_id: str, reference_id: str) -> Optional[Reference]:
        """Get reference by ID."""
        ref_path = self.profiles_dir / entity_id / "references" / f"{reference_id}.yaml"

        if not ref_path.exists():
            return None

        with open(ref_path, 'r') as f:
            data = yaml.safe_load(f)

        return Reference(**data)

    def list_references(self, entity_id: str) -> List[Reference]:
        """List all references for entity."""
        refs_dir = self.profiles_dir / entity_id / "references"

        if not refs_dir.exists():
            return []

        references = []
        for ref_file in refs_dir.glob("*.yaml"):
            with open(ref_file, 'r') as f:
                data = yaml.safe_load(f)
                references.append(Reference(**data))

        return sorted(references, key=lambda r: r.organization)

    def remove_reference(self, entity_id: str, reference_id: str) -> bool:
        """Remove reference from entity."""
        ref_path = self.profiles_dir / entity_id / "references" / f"{reference_id}.yaml"

        if not ref_path.exists():
            return False

        ref_path.unlink()

        # Update profile's references list
        profile = self.get_entity_profile(entity_id)
        if reference_id in profile.references:
            profile.references.remove(reference_id)
            self._save_profile(profile)

        logger.info(f"Removed reference {reference_id} from {entity_id}")
        return True

    # ========================================
    # INSURANCE OPERATIONS
    # ========================================

    def add_insurance(self, entity_id: str, insurance: Insurance) -> bool:
        """Add insurance policy to entity."""
        entity_dir = self.profiles_dir / entity_id

        if not entity_dir.exists():
            raise ValueError(f"Entity not found: {entity_id}")

        # Save insurance file
        ins_path = entity_dir / "insurance" / f"{insurance.policy_id}.yaml"
        data = insurance.model_dump(mode='json')

        with open(ins_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Update profile's insurance list
        profile = self.get_entity_profile(entity_id)
        if insurance.policy_id not in profile.insurance:
            profile.insurance.append(insurance.policy_id)
            self._save_profile(profile)

        logger.info(f"Added insurance {insurance.policy_id} to {entity_id}")
        return True

    def get_insurance(self, entity_id: str, policy_id: str) -> Optional[Insurance]:
        """Get insurance policy by ID."""
        ins_path = self.profiles_dir / entity_id / "insurance" / f"{policy_id}.yaml"

        if not ins_path.exists():
            return None

        with open(ins_path, 'r') as f:
            data = yaml.safe_load(f)

        return Insurance(**data)

    def list_insurance(self, entity_id: str) -> List[Insurance]:
        """List all insurance policies for entity."""
        ins_dir = self.profiles_dir / entity_id / "insurance"

        if not ins_dir.exists():
            return []

        policies = []
        for ins_file in ins_dir.glob("*.yaml"):
            with open(ins_file, 'r') as f:
                data = yaml.safe_load(f)
                policies.append(Insurance(**data))

        return sorted(policies, key=lambda i: i.policy_type.value)

    def remove_insurance(self, entity_id: str, policy_id: str) -> bool:
        """Remove insurance policy from entity."""
        ins_path = self.profiles_dir / entity_id / "insurance" / f"{policy_id}.yaml"

        if not ins_path.exists():
            return False

        ins_path.unlink()

        # Update profile's insurance list
        profile = self.get_entity_profile(entity_id)
        if policy_id in profile.insurance:
            profile.insurance.remove(policy_id)
            self._save_profile(profile)

        logger.info(f"Removed insurance {policy_id} from {entity_id}")
        return True

    # ========================================
    # CAPABILITY OPERATIONS
    # ========================================

    def add_capability(self, entity_id: str, capability: Capability) -> bool:
        """Add capability to entity."""
        entity_dir = self.profiles_dir / entity_id

        if not entity_dir.exists():
            raise ValueError(f"Entity not found: {entity_id}")

        # Save capability file
        cap_path = entity_dir / "capabilities" / f"{capability.capability_id}.yaml"
        data = capability.model_dump(mode='json')

        with open(cap_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Update profile's capabilities list
        profile = self.get_entity_profile(entity_id)
        if capability.capability_id not in profile.capabilities:
            profile.capabilities.append(capability.capability_id)
            self._save_profile(profile)

        logger.info(f"Added capability {capability.capability_id} to {entity_id}")
        return True

    def get_capability(self, entity_id: str, capability_id: str) -> Optional[Capability]:
        """Get capability by ID."""
        cap_path = self.profiles_dir / entity_id / "capabilities" / f"{capability_id}.yaml"

        if not cap_path.exists():
            return None

        with open(cap_path, 'r') as f:
            data = yaml.safe_load(f)

        return Capability(**data)

    def list_capabilities(self, entity_id: str) -> List[Capability]:
        """List all capabilities for entity."""
        cap_dir = self.profiles_dir / entity_id / "capabilities"

        if not cap_dir.exists():
            return []

        capabilities = []
        for cap_file in cap_dir.glob("*.yaml"):
            with open(cap_file, 'r') as f:
                data = yaml.safe_load(f)
                capabilities.append(Capability(**data))

        return sorted(capabilities, key=lambda c: c.capability_name)

    def remove_capability(self, entity_id: str, capability_id: str) -> bool:
        """Remove capability from entity."""
        cap_path = self.profiles_dir / entity_id / "capabilities" / f"{capability_id}.yaml"

        if not cap_path.exists():
            return False

        cap_path.unlink()

        # Update profile's capabilities list
        profile = self.get_entity_profile(entity_id)
        if capability_id in profile.capabilities:
            profile.capabilities.remove(capability_id)
            self._save_profile(profile)

        logger.info(f"Removed capability {capability_id} from {entity_id}")
        return True

    # ========================================
    # NARRATIVE OPERATIONS
    # ========================================

    def get_narrative(self, entity_id: str) -> str:
        """
        Get narrative markdown content.

        Args:
            entity_id: Entity ID

        Returns:
            Narrative content as markdown string
        """
        narrative_path = self.profiles_dir / entity_id / "narrative.md"

        if not narrative_path.exists():
            return ""

        return narrative_path.read_text()

    def update_narrative(self, entity_id: str, content: str) -> bool:
        """
        Update narrative markdown content.

        Args:
            entity_id: Entity ID
            content: New narrative content

        Returns:
            True if updated successfully
        """
        narrative_path = self.profiles_dir / entity_id / "narrative.md"

        narrative_path.write_text(content)

        logger.info(f"Updated narrative for {entity_id}")
        return True

    def get_narrative_section(self, entity_id: str, section_name: str) -> str:
        """
        Extract specific section from narrative.

        Args:
            entity_id: Entity ID
            section_name: Section heading (e.g., "company_overview")

        Returns:
            Section content as markdown string
        """
        narrative = self.get_narrative(entity_id)

        # Simple section extraction (look for ## Section Name)
        section_heading = f"## {section_name.replace('_', ' ').title()}"
        lines = narrative.split('\n')

        in_section = False
        section_content = []

        for line in lines:
            if line.startswith('## '):
                if in_section:
                    # End of our section
                    break
                if line.lower().startswith(section_heading.lower()):
                    in_section = True
                    continue
            elif in_section:
                section_content.append(line)

        return '\n'.join(section_content).strip()

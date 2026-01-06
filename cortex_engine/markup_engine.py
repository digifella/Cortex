"""
Markup Engine
Version: 2.0.0
Date: 2026-01-06

Purpose: LLM-first intelligent markup of tender documents with @mention suggestions.

Major Enhancements (v2.0.0):
- LLM-first analysis as primary method with pattern-based fallback
- Document structure analysis to identify sections (company, personnel, projects, etc.)
- Context-aware field mapping that distinguishes company vs personnel fields
- Intelligent skip of personnel sections to avoid incorrect suggestions
- Pattern-based detection retained as reliable fallback
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from .mention_parser import MentionParser
from .entity_profile_manager import EntityProfileManager
from .workspace_model import MentionBinding
from .utils import get_logger
from .llm_interface import LLMInterface
from .document_chunker import DocumentChunker, DocumentChunk

logger = get_logger(__name__)


class MarkupEngine:
    """LLM-assisted markup engine for suggesting @mentions."""

    def __init__(
        self,
        entity_manager: EntityProfileManager,
        llm: LLMInterface
    ):
        """
        Initialize markup engine.

        Args:
            entity_manager: Entity profile manager
            llm: LLM interface
        """
        self.entity_manager = entity_manager
        self.llm = llm
        self.parser = MentionParser()

        logger.info("MarkupEngine initialized")

    def analyze_document(
        self,
        document_text: str,
        entity_id: str
    ) -> List[MentionBinding]:
        """
        Analyze document and suggest @mention placements.

        Args:
            document_text: Full document text
            entity_id: Entity profile ID to use

        Returns:
            List of suggested mention bindings

        Example:
            >>> engine = MarkupEngine(entity_manager, llm)
            >>> mentions = engine.analyze_document(tender_text, "longboardfella_consulting")
            >>> print(f"Suggested {len(mentions)} mentions")
        """
        logger.info(f"Analyzing document for entity {entity_id}")

        # Get entity profile to know what data is available
        profile = self.entity_manager.get_entity_profile(entity_id)

        if not profile:
            raise ValueError(f"Entity not found: {entity_id}")

        suggestions = []

        # 0. Detect existing @mentions in document
        existing_mentions = self._detect_existing_mentions(document_text, entity_id)
        suggestions.extend(existing_mentions)

        # 1. PRIMARY: LLM-based intelligent analysis (slower, highly accurate)
        try:
            logger.info("Attempting LLM-based document analysis (primary method)")
            llm_suggestions = self._analyze_with_llm(document_text, entity_id, profile)

            if llm_suggestions:
                logger.info(f"LLM analysis succeeded: {len(llm_suggestions)} suggestions")
                suggestions.extend(llm_suggestions)
            else:
                raise ValueError("LLM returned no suggestions")

        except Exception as e:
            # 2. FALLBACK: Pattern-based detection (fast, deterministic)
            logger.warning(f"LLM analysis failed ({e}), falling back to pattern-based detection")
            pattern_suggestions = self._detect_pattern_based_fields(document_text, entity_id)
            suggestions.extend(pattern_suggestions)
            logger.info(f"Pattern-based fallback: {len(pattern_suggestions)} suggestions")

        logger.info(f"Generated {len(suggestions)} total mention suggestions")

        return suggestions

    def analyze_chunk(
        self,
        chunk: DocumentChunk,
        entity_id: str
    ) -> List[MentionBinding]:
        """
        Analyze a single chunk and suggest @mention placements.

        This is the PREFERRED method for large documents as it:
        - Provides full context to LLM (chunk is small enough)
        - Avoids context limit issues
        - Matches human review workflow

        Args:
            chunk: DocumentChunk object
            entity_id: Entity profile ID to use

        Returns:
            List of suggested mention bindings for this chunk

        Example:
            >>> chunker = DocumentChunker()
            >>> chunks = chunker.create_chunks(document_text)
            >>> for chunk in chunks:
            ...     if chunk.is_completable:
            ...         mentions = engine.analyze_chunk(chunk, "longboardfella_consulting")
            ...         # Review and approve mentions for this chunk
        """
        logger.info(f"Analyzing chunk {chunk.chunk_id}: '{chunk.title}' ({chunk.char_count} chars)")

        # Get entity profile
        profile = self.entity_manager.get_entity_profile(entity_id)
        if not profile:
            raise ValueError(f"Entity not found: {entity_id}")

        suggestions = []

        # 1. Detect existing @mentions in chunk
        existing_mentions = self._detect_existing_mentions(chunk.content, entity_id)
        for mention in existing_mentions:
            mention.chunk_id = chunk.chunk_id
        suggestions.extend(existing_mentions)

        # 2. Use LLM to analyze chunk with FULL context
        try:
            logger.info(f"Running LLM analysis on chunk {chunk.chunk_id}")
            llm_suggestions = self._analyze_chunk_with_llm(chunk, entity_id, profile)

            # Associate with chunk
            for mention in llm_suggestions:
                mention.chunk_id = chunk.chunk_id

            suggestions.extend(llm_suggestions)
            logger.info(f"Chunk {chunk.chunk_id}: Found {len(llm_suggestions)} LLM suggestions")

        except Exception as e:
            # Fallback to pattern-based
            logger.warning(f"LLM analysis failed for chunk {chunk.chunk_id}: {e}")
            pattern_suggestions = self._detect_pattern_based_fields(chunk.content, entity_id)

            for mention in pattern_suggestions:
                mention.chunk_id = chunk.chunk_id

            suggestions.extend(pattern_suggestions)

        logger.info(f"Chunk {chunk.chunk_id}: Generated {len(suggestions)} total suggestions")
        return suggestions

    def _analyze_chunk_with_llm(
        self,
        chunk: DocumentChunk,
        entity_id: str,
        profile
    ) -> List[MentionBinding]:
        """
        Use LLM to analyze a chunk and suggest mentions.

        Since chunks are small (~4000 chars), we can send full context to LLM.
        """
        import json

        # Available entity profile fields
        available_fields = {
            '@companyname': 'Company legal name',
            '@abn': 'Australian Business Number',
            '@acn': 'Australian Company Number',
            '@registered_office': 'Registered office address',
            '@email': 'Contact email',
            '@phone': 'Contact phone number',
            '@website': 'Company website',
            '@narrative[company_overview]': 'Company overview/executive summary'
        }

        prompt = f"""Analyze this section from a tender document and identify fields that need @mention replacements.

Section: {chunk.title}
Content:
```
{chunk.content}
```

Available entity profile mentions:
{json.dumps(available_fields, indent=2)}

CRITICAL RULES:
1. ONLY suggest mentions for fields asking the RESPONDENT to provide their company information
2. DO NOT suggest for:
   - Individual personnel details (surname, first name, personal email/phone)
   - Sections about "Specified Personnel" or "Team Members"
   - Informational text (RFT contact details, instructions)
3. For each field you identify, determine:
   - The best matching @mention
   - The approximate line number in this chunk
   - Whether it's actually asking for respondent data

Respond with JSON array (empty if no fields found):
[
    {{
        "field_label": "Company legal name",
        "mention": "@companyname",
        "line_offset": 15,
        "confidence": "high"
    }}
]"""

        try:
            response = self.llm.generate(prompt)

            # Extract JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            mappings = json.loads(json_str)
            suggestions = []

            for mapping in mappings:
                mention_text = mapping.get('mention', '')
                line_offset = mapping.get('line_offset', 0)

                parsed = self.parser.parse(mention_text)
                if parsed and parsed.is_valid:
                    binding = MentionBinding(
                        mention_text=mention_text,
                        mention_type=parsed.mention_type.value,
                        field_path=parsed.field_path,
                        location=f"{chunk.title} (Line {chunk.start_line + line_offset})",
                        suggested_by_llm=True,
                        chunk_id=chunk.chunk_id
                    )
                    suggestions.append(binding)

            return suggestions

        except Exception as e:
            logger.error(f"Failed to analyze chunk with LLM: {e}")
            return []

    def _detect_existing_mentions(
        self,
        text: str,
        entity_id: str
    ) -> List[MentionBinding]:
        """
        Detect existing @mentions in document.

        Args:
            text: Document text
            entity_id: Entity ID

        Returns:
            List of mention bindings for existing @mentions
        """
        from .field_substitution_engine import FieldSubstitutionEngine

        bindings = []

        # Parse all @mentions in the document
        mentions = self.parser.parse_all(text)

        # Check each mention and create binding
        engine = FieldSubstitutionEngine(self.entity_manager)

        for parsed_mention in mentions:
            if not parsed_mention.is_valid:
                continue

            # Resolve to check if requires LLM
            result = engine.resolve(parsed_mention, entity_id)

            # Determine location (line number)
            lines = text.split('\n')
            location = "Unknown"
            for line_num, line in enumerate(lines):
                if parsed_mention.raw_text in line:
                    location = f"Line {line_num + 1}"
                    break

            binding = MentionBinding(
                mention_text=parsed_mention.raw_text,
                mention_type=parsed_mention.mention_type.value,
                field_path=parsed_mention.field_path,
                location=location,
                suggested_by_llm=False,
                requires_llm=result.requires_llm if result else False
            )

            bindings.append(binding)

        logger.info(f"Detected {len(bindings)} existing @mentions in document")

        return bindings

    def _is_request_field(
        self,
        line: str,
        context_lines: List[str],
        field_type: str
    ) -> bool:
        """
        Use LLM to determine if a detected pattern is actually requesting data from respondent.

        Args:
            line: The line with the detected pattern
            context_lines: Surrounding lines for context (typically 3-5 lines before and after)
            field_type: Type of field detected (e.g., 'email', 'abn', 'phone')

        Returns:
            True if this is requesting data from respondent, False if informational

        Example:
            "Email: contracts@digitalhealth.gov.au" → False (informational)
            "Please provide your contact email:" → True (request)
        """
        try:
            # Build context
            context = '\n'.join(context_lines)

            # Create prompt for LLM
            prompt = f"""Analyze this excerpt from a tender document and determine if it is requesting information FROM the respondent or providing information TO the respondent.

Context from document:
```
{context}
```

Target line: "{line}"
Detected field type: {field_type}

Question: Is this line asking the RESPONDENT to provide their {field_type}, or is this INFORMATIONAL text (e.g., tender contact details, instructions, examples)?

Respond with EXACTLY ONE WORD:
- REQUEST if asking respondent to provide data
- INFORMATIONAL if providing info to respondent

Response:"""

            response = self.llm.generate(prompt)
            response_cleaned = response.strip().upper()

            logger.debug(f"LLM classification for '{line}': {response_cleaned}")

            return "REQUEST" in response_cleaned

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, defaulting to True")
            # Default to suggesting if LLM fails (conservative approach)
            return True

    def _detect_pattern_based_fields(
        self,
        text: str,
        entity_id: str
    ) -> List[MentionBinding]:
        """
        Detect fields using pattern matching.

        Args:
            text: Document text
            entity_id: Entity ID

        Returns:
            List of mention bindings
        """
        suggestions = []

        # Patterns for common fields
        # NOTE: Order matters! More specific patterns MUST come before generic ones
        patterns = {
            # Company details
            r'(?i)legal\s+(?:entity\s+)?name[:\s]*$': '@companyname',
            r'(?i)company\s+name[:\s]*$': '@companyname',
            r'(?i)business\s+name[:\s]*$': '@companyname',
            r'(?i)abn[:\s]*$': '@abn',
            r'(?i)australian\s+business\s+number[:\s]*$': '@abn',
            r'(?i)acn[:\s]*$': '@acn',
            r'(?i)australian\s+company\s+number[:\s]*$': '@acn',

            # Contact details - SPECIFIC patterns first!
            # Email patterns BEFORE generic address patterns
            r'(?i)e-?mail\s+address[:\s]*$': '@email',
            r'(?i)email[:\s]*$': '@email',
            r'(?i)e-mail[:\s]*$': '@email',
            r'(?i)contact\s+email[:\s]*$': '@email',

            # Website before generic address
            r'(?i)website[:\s]*$': '@website',
            r'(?i)web\s+(?:site|address)[:\s]*$': '@website',

            # Phone patterns
            r'(?i)(?:telephone|phone)\s+number[:\s]*$': '@phone',
            r'(?i)mobile\s+(?:phone\s+)?number[:\s]*$': '@phone',
            r'(?i)phone[:\s]*$': '@phone',
            r'(?i)telephone[:\s]*$': '@phone',
            r'(?i)contact\s+(?:phone|number)[:\s]*$': '@phone',

            # Address patterns - MUST be after email/website/phone!
            r'(?i)registered\s+(?:office\s+)?address[:\s]*$': '@registered_office',
            r'(?i)postal\s+address[:\s]*$': '@registered_office',
            r'(?i)business\s+address[:\s]*$': '@registered_office',
            r'(?i)(?:office\s+)?address[:\s]*$': '@registered_office',

            # Executive summary / company overview
            r'(?i)executive\s+summary[:\s]*$': '@narrative[company_overview]',
            r'(?i)company\s+(?:overview|profile)[:\s]*$': '@narrative[company_overview]',
            r'(?i)about\s+(?:the\s+)?company[:\s]*$': '@narrative[company_overview]',

            # Insurance
            r'(?i)insurance\s+coverage[:\s]*$': '@insurance.public_liability.coverage',
            r'(?i)public\s+liability[:\s]*$': '@insurance.public_liability.coverage',
            r'(?i)insurance\s+(?:policy\s+)?number[:\s]*$': '@insurance.public_liability.policy_number',
        }

        lines = text.split('\n')

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()

            # Get context lines (5 before, current, 5 after) for analysis
            context_start = max(0, line_num - 5)
            context_end = min(len(lines), line_num + 6)
            context_lines = lines[context_start:context_end]

            # Skip fields in "personnel" or "team member" sections
            # These need custom fields for each person, not company profile fields
            context_text = ' '.join(context_lines).lower()
            if any(keyword in context_text for keyword in [
                'specified personnel', 'personnel details', 'team member',
                'key personnel', 'nominated personnel', 'staff member',
                'repeat as required for each', 'for each person'
            ]):
                logger.debug(f"Skipping line {line_num + 1} - in personnel/team section")
                continue

            for pattern, mention in patterns.items():
                if re.search(pattern, line_stripped):
                    # Found a potential field - but is it actually requesting respondent data?
                    # Extract field type from mention for better prompting
                    field_type = mention.replace('@', '').split('.')[0].split('[')[0]

                    # Use LLM to classify if this is actually requesting data
                    is_request = self._is_request_field(
                        line=line_stripped,
                        context_lines=context_lines,
                        field_type=field_type
                    )

                    if not is_request:
                        logger.info(f"Skipping informational field at line {line_num + 1}: {line_stripped}")
                        continue  # Skip this suggestion - it's informational text

                    # This is a legitimate request field - suggest it
                    parsed = self.parser.parse(mention)

                    location = f"Line {line_num + 1}"

                    # Check if this section already exists
                    if line_num > 0:
                        # Look at previous lines for section heading
                        for i in range(max(0, line_num - 5), line_num):
                            if self._is_section_heading(lines[i]):
                                location = lines[i].strip()
                                break

                    binding = MentionBinding(
                        mention_text=mention,
                        mention_type=parsed.mention_type.value,
                        field_path=parsed.field_path,
                        location=location,
                        suggested_by_llm=False
                    )

                    suggestions.append(binding)
                    logger.info(f"Suggested {mention} at line {line_num + 1} (verified as request field)")

        return suggestions

    def _analyze_with_llm(
        self,
        text: str,
        entity_id: str,
        profile
    ) -> List[MentionBinding]:
        """
        PRIMARY METHOD: Use LLM to intelligently analyze document and suggest mentions.

        This is a comprehensive analysis that:
        1. Identifies document structure and sections
        2. Classifies each section (company info, personnel, projects, etc.)
        3. For each field requesting data, determines appropriate mention or custom field
        4. Returns high-quality suggestions with context awareness

        Args:
            text: Document text
            entity_id: Entity ID
            profile: Entity profile object

        Returns:
            List of mention bindings with high confidence
        """
        logger.info("Starting LLM-based document analysis")

        try:
            # Step 1: Analyze document structure
            structure = self._llm_analyze_document_structure(text)

            # Step 2: For each section, identify fields and map to mentions
            suggestions = []

            for section in structure.get('sections', []):
                section_type = section.get('type', 'unknown')
                section_suggestions = []

                if section_type == 'company_information':
                    section_suggestions = self._llm_map_company_fields(section, profile, text)

                elif section_type == 'personnel_information':
                    # Skip personnel sections - these need custom fields
                    logger.info(f"Skipping personnel section: {section.get('title', 'Unknown')}")
                    continue

                elif section_type == 'project_examples':
                    section_suggestions = self._llm_map_project_fields(section, profile)

                elif section_type == 'insurance_certification':
                    section_suggestions = self._llm_map_insurance_fields(section, profile)

                elif section_type == 'tenderer_contact':
                    section_suggestions = self._llm_map_contact_fields(section, profile, text)

                suggestions.extend(section_suggestions)

            logger.info(f"LLM analysis complete: {len(suggestions)} suggestions")
            return suggestions

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise

    def _llm_analyze_document_structure(self, text: str) -> Dict:
        """
        Use LLM to analyze document structure and identify sections.

        Returns dict with structure:
        {
            'sections': [
                {
                    'title': 'Company Information',
                    'type': 'company_information',
                    'start_line': 10,
                    'end_line': 25,
                    'fields': ['company name', 'abn', 'address', ...]
                },
                ...
            ]
        }
        """
        import json

        # For long documents, sample strategically to catch all sections
        # Take first 15K chars + middle 10K + last 5K to ensure we see the full structure
        doc_length = len(text)
        if doc_length > 30000:
            sample_text = (
                text[:15000] +
                "\n\n... [middle content omitted] ...\n\n" +
                text[doc_length//2 - 5000:doc_length//2 + 5000] +
                "\n\n... [content omitted] ...\n\n" +
                text[-5000:]
            )
        else:
            sample_text = text

        prompt = f"""Analyze this tender/RFT document and identify its structure.

Document (length: {doc_length} chars, showing key sections):
```
{sample_text}
```

Task: Identify all major sections in the document where the RESPONDENT is being asked to provide information.

For each section, determine:
1. Section title
2. Section type (choose from: company_information, personnel_information, project_examples, insurance_certification, tenderer_contact, other)
3. Approximate line numbers (estimate based on content position)
4. List of fields being requested in that section

CRITICAL RULES - READ CAREFULLY:
1. "Personnel" / "Specified Personnel" / "Key Personnel" / "Team Members" / "Staff" sections:
   - These are asking for INDIVIDUAL PEOPLE'S personal details
   - Type: personnel_information
   - DO NOT suggest company profile fields for these sections
   - Examples: "Personal details of Specified Personnel", "Nominated Personnel", "Key staff"

2. "Tenderer contact" / "Contact person for RFT" sections:
   - These are asking for who to contact about the RFT/tender
   - Type: tenderer_contact
   - Examples: "Tenderer contact details for RFT", "Contact person name/email/phone"

3. Company / Business sections:
   - These ask for the COMPANY/ORGANIZATION details
   - Type: company_information
   - Examples: "Tenderer Details", "Business Information", "Legal Entity Details"

IMPORTANT: If you see phrases like "Repeat as required for each", "where applicable", or fields like "Surname", "First name" - this is ALWAYS personnel_information, NOT company_information.

Respond with valid JSON only:
{{
    "sections": [
        {{
            "title": "Section name from document",
            "type": "company_information|personnel_information|project_examples|insurance_certification|tenderer_contact|other",
            "start_line": 100,
            "end_line": 150,
            "fields": ["field1", "field2", ...]
        }}
    ]
}}"""

        try:
            response = self.llm.generate(prompt)

            # Extract JSON from response (might have markdown code blocks)
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            structure = json.loads(json_str)
            logger.info(f"Document structure identified: {len(structure.get('sections', []))} sections")

            # Log each section for debugging
            for section in structure.get('sections', []):
                logger.info(f"  Section: '{section.get('title')}' -> Type: {section.get('type')} (lines {section.get('start_line')}-{section.get('end_line')})")

            return structure

        except Exception as e:
            logger.error(f"Failed to parse document structure: {e}")
            return {'sections': []}

    def _llm_map_company_fields(self, section: Dict, profile, full_document_text: str = "") -> List[MentionBinding]:
        """Map company information fields to entity profile mentions."""
        import json

        suggestions = []

        # Get available entity profile fields
        available_fields = {
            '@companyname': 'Company legal name',
            '@abn': 'Australian Business Number',
            '@acn': 'Australian Company Number',
            '@registered_office': 'Registered office address',
            '@email': 'Contact email',
            '@phone': 'Contact phone number',
            '@website': 'Company website'
        }

        prompt = f"""Given this section from a tender document, map each field to the appropriate entity profile mention.

Section: {section.get('title', 'Unknown')}
Fields requesting data: {section.get('fields', [])}

Available entity profile mentions:
{json.dumps(available_fields, indent=2)}

CRITICAL: This section should contain COMPANY/ORGANIZATION fields only.

REJECT these field types (DO NOT include in output):
- Individual person's name (Surname, First name, etc.)
- Individual person's email/phone (in context of "Personnel details")
- Fields that appear near "Repeat as required for each person"
- Any field clearly asking for individual staff/team member details

ACCEPT these field types (include in output):
- Company legal name, ABN, ACN
- Company registered office address
- Company contact email/phone (official business contact)
- Company website

For each ACCEPTED field, determine the best matching entity profile mention.

Respond with JSON array (empty array if no fields match):
[
    {{
        "field_label": "Company name",
        "mention": "@companyname",
        "line_number": 105
    }},
    ...
]"""

        try:
            response = self.llm.generate(prompt)

            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            mappings = json.loads(json_str)

            # Get document lines for validation
            doc_lines = full_document_text.split('\n') if full_document_text else []

            for mapping in mappings:
                mention_text = mapping.get('mention', '')
                line_num = mapping.get('line_number', 0)

                # SAFETY CHECK: Verify this line isn't in a personnel section
                # Check 10 lines before and after for personnel keywords
                if line_num > 0 and doc_lines:
                    start = max(0, line_num - 10)
                    end = min(len(doc_lines), line_num + 10)
                    context = ' '.join(doc_lines[start:end]).lower()

                    # Check for personnel section indicators
                    personnel_keywords = [
                        'specified personnel', 'personnel details', 'key personnel',
                        'repeat as required for each', 'surname', 'first name',
                        'personal details of', 'nominated personnel'
                    ]

                    if any(kw in context for kw in personnel_keywords):
                        logger.warning(f"REJECTED: {mention_text} at line {line_num} - in personnel section")
                        continue  # Skip this suggestion

                parsed = self.parser.parse(mention_text)

                if parsed and parsed.is_valid:
                    binding = MentionBinding(
                        mention_text=mention_text,
                        mention_type=parsed.mention_type.value,
                        field_path=parsed.field_path,
                        location=f"Line {line_num}",
                        suggested_by_llm=True
                    )
                    suggestions.append(binding)

            logger.info(f"Mapped {len(suggestions)} company fields")
            return suggestions

        except Exception as e:
            logger.warning(f"Failed to map company fields: {e}")
            return []

    def _llm_map_project_fields(self, section: Dict, profile) -> List[MentionBinding]:
        """Map project example fields to narrative mentions."""
        # Simplified for now - can expand later
        logger.info(f"Project section detected: {section.get('title')} - suggesting @narrative")
        return []

    def _llm_map_insurance_fields(self, section: Dict, profile) -> List[MentionBinding]:
        """Map insurance/certification fields."""
        # Simplified for now - can expand later
        logger.info(f"Insurance section detected: {section.get('title')}")
        return []

    def _llm_map_contact_fields(self, section: Dict, profile, full_document_text: str = "") -> List[MentionBinding]:
        """Map tenderer contact fields to entity profile."""
        # Similar to company fields but for RFT contact section
        return self._llm_map_company_fields(section, profile, full_document_text)

    def _is_section_heading(self, line: str) -> bool:
        """Check if line is a section heading."""
        line = line.strip()

        if not line or len(line) < 5:
            return False

        # Common patterns
        patterns = [
            line.startswith('SECTION'),
            line.startswith('PART'),
            line.startswith('CHAPTER'),
            line.isupper() and len(line.split()) <= 10,
            line.startswith('##'),
        ]

        # Numbered sections
        if line[0].isdigit() and ('.' in line[:10]):
            return True

        return any(patterns)

    def insert_mentions_in_document(
        self,
        document_text: str,
        mention_bindings: List[MentionBinding]
    ) -> str:
        """
        Insert @mentions into document text at suggested locations.

        Args:
            document_text: Original document text
            mention_bindings: List of approved mention bindings

        Returns:
            Document text with @mentions inserted

        Example:
            >>> marked_up = engine.insert_mentions_in_document(
            ...     document_text,
            ...     approved_mentions
            ... )
        """
        lines = document_text.split('\n')

        # Group mentions by location
        mentions_by_location: Dict[str, List[MentionBinding]] = {}
        for binding in mention_bindings:
            if binding.approved and not binding.rejected:
                loc = binding.location
                if loc not in mentions_by_location:
                    mentions_by_location[loc] = []
                mentions_by_location[loc].append(binding)

        # Insert mentions
        for line_num, line in enumerate(lines):
            # Check if this line matches a location
            location_key = f"Line {line_num + 1}"

            if location_key in mentions_by_location:
                # Insert mentions after this line
                for binding in mentions_by_location[location_key]:
                    lines[line_num] += f"\n{binding.mention_text}"

        return '\n'.join(lines)

    def validate_mentions(
        self,
        document_text: str,
        entity_id: str
    ) -> Tuple[List[str], List[str]]:
        """
        Validate all @mentions in document.

        Args:
            document_text: Document text with @mentions
            entity_id: Entity ID

        Returns:
            Tuple of (valid_mentions, invalid_mentions)

        Example:
            >>> valid, invalid = engine.validate_mentions(marked_up_text, "longboardfella_consulting")
            >>> print(f"Valid: {len(valid)}, Invalid: {len(invalid)}")
        """
        # Find all mentions
        mentions = self.parser.parse_all(document_text)

        valid_mentions = []
        invalid_mentions = []

        # Try to resolve each mention
        from .field_substitution_engine import FieldSubstitutionEngine

        engine = FieldSubstitutionEngine(self.entity_manager)

        for mention in mentions:
            result = engine.resolve(mention, entity_id)

            if result.success or result.requires_llm:
                valid_mentions.append(mention.raw_text)
            else:
                invalid_mentions.append(mention.raw_text)

        logger.info(f"Validation: {len(valid_mentions)} valid, {len(invalid_mentions)} invalid")

        return valid_mentions, invalid_mentions

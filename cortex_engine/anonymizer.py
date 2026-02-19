# cortex_engine/anonymizer.py
# Version: 1.0.0 
# Date: 2025-07-30
# Purpose: Document anonymization engine that replaces identifying information
#          with generic placeholders while preserving document structure.

import re
import os
import spacy
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Any
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from cortex_engine.entity_extractor import EntityExtractor, ExtractedEntity
from cortex_engine.utils import get_logger
from cortex_engine.utils.path_utils import normalize_path, ensure_directory

logger = get_logger(__name__)

class AnonymizationMapping:
    """Tracks entity mappings during anonymization."""
    
    def __init__(self):
        self.person_counter = 0
        self.org_counter = 0
        self.project_counter = 0
        self.location_counter = 0
        self.generic_counter = 0
        
        self.mappings = {}  # original -> anonymized
        self.reverse_mappings = {}  # anonymized -> original
        
    def get_anonymous_name(self, entity_name: str, entity_type: str) -> str:
        """Get or create anonymous replacement for an entity."""
        if entity_name in self.mappings:
            return self.mappings[entity_name]
        
        if entity_type == 'person':
            self.person_counter += 1
            anonymous = f"Person {chr(64 + self.person_counter)}"  # Person A, Person B, etc.
        elif entity_type == 'organization':
            self.org_counter += 1
            anonymous = f"Company {self.org_counter}"
        elif entity_type == 'project':
            self.project_counter += 1
            anonymous = f"Project {self.project_counter}"
        elif entity_type == 'location':
            self.location_counter += 1
            anonymous = f"Location {self.location_counter}"
        else:
            self.generic_counter += 1
            anonymous = f"Entity {self.generic_counter}"
        
        self.mappings[entity_name] = anonymous
        self.reverse_mappings[anonymous] = entity_name
        return anonymous
    
    def get_mapping_report(self) -> Dict[str, str]:
        """Get full mapping report for review."""
        return self.mappings.copy()


@dataclass
class AnonymizationOptions:
    """Granular controls for anonymization behavior."""
    redact_people: bool = True
    redact_organizations: bool = True
    redact_projects: bool = True
    redact_locations: bool = True
    redact_emails: bool = True
    redact_phones: bool = True
    redact_urls: bool = True
    redact_headers_footers: bool = True
    redact_personal_pronouns: bool = False
    redact_company_names: bool = False
    custom_company_names: List[str] = field(default_factory=list)
    preserve_source_formatting: bool = True

    @classmethod
    def from_input(cls, value: Optional[Union["AnonymizationOptions", Dict[str, Any]]]) -> "AnonymizationOptions":
        """Build options from dict/user input while keeping safe defaults."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls()

        kwargs: Dict[str, Any] = {}
        for key in (
            "redact_people",
            "redact_organizations",
            "redact_projects",
            "redact_locations",
            "redact_emails",
            "redact_phones",
            "redact_urls",
            "redact_headers_footers",
            "redact_personal_pronouns",
            "redact_company_names",
            "preserve_source_formatting",
        ):
            if key in value:
                kwargs[key] = bool(value.get(key))

        custom_names_raw = value.get("custom_company_names", [])
        if isinstance(custom_names_raw, str):
            custom_names = [name.strip() for name in custom_names_raw.split(",") if name.strip()]
        elif isinstance(custom_names_raw, list):
            custom_names = [str(name).strip() for name in custom_names_raw if str(name).strip()]
        else:
            custom_names = []
        kwargs["custom_company_names"] = custom_names

        return cls(**kwargs)

class DocumentAnonymizer:
    """Main anonymization engine that processes documents."""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        
        # Additional patterns for emails, phone numbers, addresses
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}')
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        self.personal_pronoun_pattern = re.compile(
            r"\b(he|she|him|her|his|hers|himself|herself|they|them|their|theirs|themself|themselves)\b",
            re.IGNORECASE,
        )
        
        # Enhanced name patterns for medical/academic documents
        self.comprehensive_name_patterns = [
            # Dr/Prof/Mr/Ms patterns
            re.compile(r'\b(?:Dr\.?\s+|Prof\.?\s+|Mr\.?\s+|Ms\.?\s+|Mrs\.?\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.IGNORECASE),
            # Principal/Associate investigator patterns
            re.compile(r'(?:Principal|Associate)\s+investigator[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.IGNORECASE),
            # Name in roles table - "Dr FirstName LastName"
            re.compile(r'Dr\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.IGNORECASE),
            # Comma-separated name lists
            re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:,\s*([A-Z][a-z]+\s+[A-Z][a-z]+))*'),
            # Names followed by professional titles
            re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:\s*,\s*(?:MD|PhD|MBBS|FRCS|Surgeon|Registrar|Fellow))', re.IGNORECASE),
            # Hospital/Institution staff patterns
            re.compile(r'(?:Department of|Hospital).*?([A-Z][a-z]+\s+[A-Z][a-z]+)', re.IGNORECASE),
        ]
        
        # Common header/footer patterns
        self.header_footer_patterns = [
            re.compile(r'^\s*Page\s+\d+\s+of\s+\d+\s*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\s*Confidential\s*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\s*©.*?(\d{4}).*$', re.MULTILINE),
            re.compile(r'^\s*All rights reserved.*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\s*\w+@\w+\.\w+\s*$', re.MULTILINE),  # Email in header/footer
        ]
        
        # Load spaCy for additional entity detection
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except (OSError, ImportError) as e:
            logger.warning(f"spaCy model not available for anonymization: {e}")
            self.nlp = None

    @staticmethod
    def _is_structural_line(line: str) -> bool:
        stripped = (line or "").strip()
        if not stripped:
            return True
        if stripped.startswith(("#", ">", "|", "```", "---")):
            return True
        if re.match(r"^[-*+]\s+", stripped):
            return True
        if re.match(r"^\d+\.\s+", stripped):
            return True
        if re.match(r"^[A-Z][A-Z0-9 _-]{3,}:?$", stripped):
            return True
        return False

    def _normalize_extracted_text(self, text: str) -> str:
        """
        Normalize extracted text similarly to Textifier:
        - remove literal CR/CRLF markers
        - normalize line endings
        - reflow hard-wrapped plain lines into paragraphs
        - preserve structural lines
        """
        normalized = (text or "")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"\s*<CRLF>\s*", "\n", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s*<CR>\s*", "\n", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        lines = [ln.rstrip() for ln in normalized.split("\n")]
        out_lines: List[str] = []
        para_buffer: List[str] = []
        in_code_fence = False

        def flush_paragraph() -> None:
            if not para_buffer:
                return
            paragraph = " ".join(part.strip() for part in para_buffer if part.strip())
            if paragraph:
                out_lines.append(paragraph)
            para_buffer.clear()

        for raw in lines:
            line = raw.strip()
            if line.startswith("```"):
                flush_paragraph()
                in_code_fence = not in_code_fence
                out_lines.append(raw)
                continue

            if in_code_fence:
                out_lines.append(raw)
                continue

            if not line:
                flush_paragraph()
                if out_lines and out_lines[-1] != "":
                    out_lines.append("")
                continue

            if self._is_structural_line(line):
                flush_paragraph()
                out_lines.append(raw)
                continue

            para_buffer.append(line)

        flush_paragraph()
        result = "\n".join(out_lines).strip()
        return result + "\n" if result else ""
    
    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file formats."""
        try:
            if file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return self._normalize_extracted_text(f.read())
            elif file_path.suffix.lower() in ['.pdf']:
                # Use PyMuPDF for PDF text extraction
                try:
                    import fitz
                    doc = fitz.open(file_path)
                    text_parts: List[str] = []
                    for page_index, page in enumerate(doc, start=1):
                        page_text = page.get_text("text") or ""
                        text_parts.append(page_text.strip())
                        if page_index < len(doc):
                            text_parts.append("")
                            text_parts.append("---")
                            text_parts.append("")
                    doc.close()
                    return self._normalize_extracted_text("\n".join(text_parts))
                except ImportError:
                    logger.error("PyMuPDF not available for PDF processing")
                    return ""
            elif file_path.suffix.lower() in ['.docx']:
                # Use python-docx for Word documents
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text_lines: List[str] = []
                    for paragraph in doc.paragraphs:
                        text_lines.append(paragraph.text)
                    return self._normalize_extracted_text("\n".join(text_lines))
                except ImportError:
                    logger.error("python-docx not available for DOCX processing")
                    return ""
            else:
                logger.warning(f"Unsupported file format: {file_path.suffix}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def extract_names_with_comprehensive_patterns(self, text: str) -> List[ExtractedEntity]:
        """Extract names using comprehensive pattern matching for medical/academic documents."""
        entities = []
        found_names = set()
        
        # Apply all comprehensive name patterns
        for pattern in self.comprehensive_name_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                # Extract name from groups
                if match.groups():
                    for group in match.groups():
                        if group and group not in found_names:
                            # Clean and validate the name
                            name = group.strip()
                            if self._is_valid_person_name(name):
                                entities.append(ExtractedEntity(
                                    name=name,
                                    entity_type='person',
                                    confidence=0.9,  # High confidence for pattern matches
                                    extraction_method='comprehensive_pattern',
                                    context_mentions=[match.group(0)]
                                ))
                                found_names.add(name)
        
        return entities
    
    def _is_valid_person_name(self, name: str) -> bool:
        """Validate if a string looks like a person's name."""
        if not name or len(name) < 3:
            return False
        
        # Must have at least 2 words
        words = name.split()
        if len(words) < 2:
            return False
        
        # Each word should start with capital letter
        for word in words:
            if not word[0].isupper():
                return False
        
        # Exclude common non-name patterns  
        exclude_patterns = {
            'Department of', 'Hospital Location', 'Private Hospital', 
            'Study Protocol', 'Version', 'April', 'General Surgery',
            'Colorectal Surgery', 'Research Fellow', 'Quality of',
            'Life and', 'Functional Location', 'Early Closure',
            'Associate investigators', 'Principal investigator', 'Position Department',
            'Name Position', 'Department Role', 'Surgery Associate', 'Surgery Principal',
            'Research Department', 'Hospital This', 'General Associate', 'General Surgical',
            'Colorectal Associate', 'Colorectal Principal', 'Surgical Department'
        }
        
        if name in exclude_patterns:
            return False
        
        # Additional validation: reject if contains common non-name words
        non_name_words = {'department', 'surgery', 'hospital', 'protocol', 'version', 
                         'study', 'research', 'associate', 'principal', 'position',
                         'role', 'general', 'colorectal', 'surgical'}
        name_words = {word.lower() for word in words}
        if len(name_words.intersection(non_name_words)) > 0:
            return False
        
        # Must contain alphabetic characters
        if not any(c.isalpha() for c in name):
            return False
        
        return True
    
    def llm_powered_entity_detection(self, text: str) -> List[ExtractedEntity]:
        """Use LLM to identify entities that pattern matching might miss."""
        # This would integrate with available LLM APIs
        # For now, implementing a rule-based approach that mimics LLM analysis
        entities = []
        
        # Look for potential names in structured content
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for lines that might contain names in lists or tables
            if any(keyword in line.lower() for keyword in ['investigator', 'researcher', 'author', 'dr ', 'prof ']):
                # Extract potential names from the line
                words = line.split()
                i = 0
                while i < len(words) - 1:
                    # Look for capitalized word pairs
                    if (words[i][0].isupper() and 
                        i + 1 < len(words) and 
                        words[i + 1][0].isupper() and
                        words[i].isalpha() and 
                        words[i + 1].replace(',', '').isalpha()):
                        
                        potential_name = f"{words[i]} {words[i + 1].rstrip(',')}"
                        if self._is_valid_person_name(potential_name):
                            entities.append(ExtractedEntity(
                                name=potential_name,
                                entity_type='person',
                                confidence=0.85,
                                extraction_method='llm_analysis',
                                context_mentions=[line]
                            ))
                    i += 1
        
        return entities
    
    def post_process_missed_entities(self, text: str, existing_entities: List[str]) -> List[ExtractedEntity]:
        """Post-processing to catch entities that were missed by other methods."""
        entities = []
        existing_names = {entity.lower() for entity in existing_entities}
        
        # Look for capitalized word sequences that might be names
        # This catches names like "Sukhwant Khanijaun" that might be missed
        pattern = re.compile(r'\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b')
        matches = pattern.finditer(text)
        
        for match in matches:
            potential_name = match.group(1)
            if (potential_name.lower() not in existing_names and 
                self._is_valid_person_name(potential_name) and
                not self._is_likely_non_name(potential_name)):
                
                entities.append(ExtractedEntity(
                    name=potential_name,
                    entity_type='person',
                    confidence=0.7,
                    extraction_method='post_processing',
                    context_mentions=[match.group(0)]
                ))
                existing_names.add(potential_name.lower())
        
        return entities
    
    def _is_likely_non_name(self, text: str) -> bool:
        """Check if text is likely not a person name."""
        non_name_indicators = {
            'Hospital Location', 'Private Hospital', 'Department of', 'Study Protocol',
            'Version', 'Quality of', 'Life and', 'Functional Location', 'Early Closure',
            'Post Rectal', 'Public and', 'General Surgery', 'Colorectal Surgery',
            'Research Fellow', 'Principal Investigator', 'Associate Investigator',
            'Royal Melbourne', 'Royal Hospital', 'Melbourne Hospital', 'Protocol Version',
            'Study Site', 'Position Department', 'Name Position', 'Department Role'
        }
        
        if text in non_name_indicators:
            return True
        
        # Check if text contains institutional keywords
        institutional_keywords = ['hospital', 'department', 'surgery', 'protocol', 
                                'study', 'research', 'version', 'institute', 'clinic']
        text_lower = text.lower()
        
        return any(keyword in text_lower for keyword in institutional_keywords)
    
    def identify_entities_for_anonymization(
        self,
        text: str,
        filename: str = "",
        confidence_threshold: float = 0.3,
        options: Optional[Union[AnonymizationOptions, Dict[str, Any]]] = None,
    ) -> List[ExtractedEntity]:
        """Identify all entities that should be anonymized using multiple methods."""
        options = AnonymizationOptions.from_input(options)
        all_entities = []
        
        # Method 1: Existing entity extraction system
        metadata = {
            'file_name': filename,
            'document_type': 'Document',
            'summary': text[:500]
        }
        entities, _ = self.entity_extractor.extract_entities_and_relationships(text, metadata)
        all_entities.extend(entities)
        
        # Method 2: Comprehensive pattern matching
        pattern_entities = self.extract_names_with_comprehensive_patterns(text)
        all_entities.extend(pattern_entities)
        
        # Method 3: LLM-powered detection
        llm_entities = self.llm_powered_entity_detection(text)
        all_entities.extend(llm_entities)
        
        # Method 4: Enhanced spaCy processing
        if self.nlp:
            try:
                doc = self.nlp(text[:15000])  # Process more text
                
                for ent in doc.ents:
                    if ent.label_ == 'PERSON':
                        # More aggressive person name extraction
                        name = ent.text.strip()
                        if self._is_valid_person_name(name):
                            all_entities.append(ExtractedEntity(
                                name=name,
                                entity_type='person',
                                confidence=0.8,
                                extraction_method='enhanced_spacy'
                            ))
                    elif ent.label_ in ['GPE', 'LOC']:
                        all_entities.append(ExtractedEntity(
                            name=ent.text.strip(),
                            entity_type='location',
                            confidence=0.8,
                            extraction_method='spacy'
                        ))
            except Exception as e:
                logger.warning(f"Enhanced spaCy processing failed: {e}")
        
        # Method 5: Post-processing for missed entities
        existing_entity_names = [entity.name for entity in all_entities]
        missed_entities = self.post_process_missed_entities(text, existing_entity_names)
        all_entities.extend(missed_entities)
        
        # Filter and deduplicate entities
        filtered_entities = []
        seen_names = set()
        common_words = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'will', 'can', 'has',
            'have', 'had', 'was', 'were', 'are', 'been', 'being', 'team', 'project',
            'report', 'date', 'page', 'website', 'footer', 'included', 'solutions',
            'services', 'system', 'systems', 'analysis', 'distribution', 'benefit',
            'cloud', 'migration', 'prepared', 'confidential', 'document', 'study',
            'protocol', 'version', 'hospital', 'department', 'surgery', 'research'
        }
        
        for entity in all_entities:
            # Deduplicate by normalized name
            normalized_name = entity.name.lower().strip()
            if normalized_name in seen_names:
                continue
            
            # Skip very short entities or common words
            if len(entity.name) < 3:
                continue
            if normalized_name in common_words:
                continue
            
            # Skip entities that are mostly numbers or symbols
            if entity.name.replace('-', '').replace(' ', '').isdigit():
                continue
            
            # Use dynamic confidence threshold
            if entity.confidence >= confidence_threshold:
                entity_type = str(getattr(entity, "entity_type", "")).lower()
                if entity_type == "person" and not options.redact_people:
                    continue
                if entity_type == "organization" and not options.redact_organizations:
                    continue
                if entity_type == "project" and not options.redact_projects:
                    continue
                if entity_type == "location" and not options.redact_locations:
                    continue
                filtered_entities.append(entity)
                seen_names.add(normalized_name)
        
        logger.info(f"Identified {len(filtered_entities)} entities for anonymization using {len(all_entities)} total detections")
        return filtered_entities
    
    def anonymize_text(
        self,
        text: str,
        entities: List[ExtractedEntity],
        mapping: AnonymizationMapping,
        options: Optional[Union[AnonymizationOptions, Dict[str, Any]]] = None,
    ) -> str:
        """Replace identified entities with anonymous placeholders."""
        options = AnonymizationOptions.from_input(options)
        anonymized_text = text
        
        # Sort entities by length (longest first) to avoid partial replacements
        sorted_entities = sorted(entities, key=lambda e: len(e.name), reverse=True)
        
        replacements_made = 0
        
        for entity in sorted_entities:
            original_name = entity.name
            anonymous_name = mapping.get_anonymous_name(original_name, entity.entity_type)
            
            try:
                # Validate entity name before processing
                if not original_name or len(original_name.strip()) == 0:
                    continue
                
                # Clean the entity name of problematic characters
                cleaned_name = original_name.strip()
                
                # Create case-insensitive replacement pattern with word boundaries
                # Use word boundaries to avoid partial matches
                escaped_name = re.escape(cleaned_name)
                
                # Validate the escaped pattern before compiling
                try:
                    pattern = re.compile(r'\b' + escaped_name + r'\b', re.IGNORECASE)
                except re.error as regex_err:
                    # Fallback to simple string replacement without word boundaries
                    logger.warning(f"Word boundary regex failed for '{cleaned_name}': {regex_err}. Using simple replacement.")
                    try:
                        pattern = re.compile(re.escape(cleaned_name), re.IGNORECASE)
                    except re.error as simple_regex_err:
                        logger.warning(f"Simple regex also failed for '{cleaned_name}': {simple_regex_err}. Using string replacement.")
                        # Last resort: simple string replacement
                        if cleaned_name in anonymized_text:
                            anonymized_text = anonymized_text.replace(cleaned_name, anonymous_name)
                            replacements_made += 1
                            logger.debug(f"String replaced '{cleaned_name}' -> '{anonymous_name}'")
                        continue
                
                # Count occurrences before replacement
                matches = pattern.findall(anonymized_text)
                if matches:
                    anonymized_text = pattern.sub(anonymous_name, anonymized_text)
                    replacements_made += len(matches)
                    logger.debug(f"Replaced '{cleaned_name}' -> '{anonymous_name}' ({len(matches)} times)")
                    
            except Exception as e:
                logger.warning(f"Unexpected error replacing '{original_name}': {e}. Skipping replacement.")
                continue
        
        # Deterministic custom company name masking for known organizations
        custom_company_replacements = 0
        if options.redact_company_names and options.custom_company_names:
            for company_name in sorted(set(options.custom_company_names), key=len, reverse=True):
                anonymous_name = mapping.get_anonymous_name(company_name, "organization")
                pattern = re.compile(r"\b" + re.escape(company_name) + r"\b", re.IGNORECASE)
                matches = pattern.findall(anonymized_text)
                if matches:
                    anonymized_text = pattern.sub(anonymous_name, anonymized_text)
                    custom_company_replacements += len(matches)

        pronoun_count = 0
        if options.redact_personal_pronouns:
            pronoun_count = len(self.personal_pronoun_pattern.findall(anonymized_text))
            anonymized_text = self.personal_pronoun_pattern.sub('[PRONOUN]', anonymized_text)

        # Handle emails, phones, URLs
        email_count = 0
        if options.redact_emails:
            email_count = len(self.email_pattern.findall(anonymized_text))
            anonymized_text = self.email_pattern.sub('[EMAIL]', anonymized_text)

        phone_count = 0
        if options.redact_phones:
            phone_count = len(self.phone_pattern.findall(anonymized_text))
            anonymized_text = self.phone_pattern.sub('[PHONE]', anonymized_text)

        url_count = 0
        if options.redact_urls:
            url_count = len(self.url_pattern.findall(anonymized_text))
            anonymized_text = self.url_pattern.sub('[URL]', anonymized_text)

        # Handle headers/footers
        footer_replacements = 0
        if options.redact_headers_footers:
            for pattern in self.header_footer_patterns:
                matches = pattern.findall(anonymized_text)
                if matches:
                    anonymized_text = pattern.sub('[HEADER/FOOTER REMOVED]', anonymized_text)
                    footer_replacements += len(matches)
        
        logger.info(
            f"Anonymization complete: {replacements_made} entity replacements, "
            f"{custom_company_replacements} custom company replacements, {pronoun_count} pronouns, "
            f"{email_count} emails, {phone_count} phones, {url_count} URLs, "
            f"{footer_replacements} headers/footers"
        )
        
        return anonymized_text
    
    def _detect_docker_environment(self) -> bool:
        """Detect if running in a Docker container."""
        try:
            return (Path("/.dockerenv").exists() or 
                   "DOCKER_CONTAINER" in os.environ or
                   (Path("/proc/self/cgroup").exists() and 
                    "docker" in Path("/proc/self/cgroup").read_text()))
        except Exception:
            return False

    def _get_safe_output_path(self, intended_path: Union[str, Path]) -> Path:
        """Get safe output path, avoiding Docker mount issues."""
        intended_path = normalize_path(intended_path)
        
        # If in Docker and trying to write to a mounted volume, use temp directory
        if self._detect_docker_environment():
            # Check if path looks like a Windows mount (common Docker issue)
            path_str = str(intended_path)
            if any(pattern in path_str.lower() for pattern in ['/mnt/c/', '/mnt/d/', '/host/']):
                logger.warning(f"Docker detected: Using temp directory instead of mounted volume {intended_path}")
                import tempfile
                temp_dir = Path(tempfile.gettempdir()) / "cortex_anonymizer"
                ensure_directory(temp_dir)
                return temp_dir / intended_path.name
        
        return intended_path

    def anonymize_single_file(self, input_path: Union[str, Path], 
                            output_path: Optional[Union[str, Path]] = None,
                            mapping: Optional[AnonymizationMapping] = None,
                            confidence_threshold: float = 0.3,
                            options: Optional[Union[AnonymizationOptions, Dict[str, Any]]] = None) -> Tuple[str, AnonymizationMapping]:
        """Anonymize a single file."""
        
        input_path = normalize_path(input_path)
        if not input_path or not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create mapping if not provided
        if mapping is None:
            mapping = AnonymizationMapping()
        
        # Determine output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_anonymized.txt"
        else:
            output_path = normalize_path(output_path)
        
        resolved_options = AnonymizationOptions.from_input(options)

        logger.info(f"Anonymizing {input_path} -> {output_path}")
        
        # Extract text
        text = self.extract_text_from_file(input_path)
        if not text:
            raise ValueError(f"Could not extract text from {input_path}")

        if resolved_options.preserve_source_formatting:
            text = self._normalize_extracted_text(text)
        
        # Identify entities
        entities = self.identify_entities_for_anonymization(
            text,
            input_path.name,
            confidence_threshold,
            resolved_options,
        )
        
        # Anonymize
        anonymized_text = self.anonymize_text(text, entities, mapping, resolved_options)
        
        # Add anonymization header
        header = f"""[ANONYMIZED DOCUMENT]
Original file: {input_path.name}
Anonymized on: {logger.handlers[0].formatter.formatTime(logger.makeRecord('', 0, '', 0, '', (), None)) if logger.handlers else 'Unknown'}
Entities replaced: {len(mapping.mappings)}

--- DOCUMENT CONTENT ---

"""
        
        anonymized_text = header + anonymized_text
        
        # Write output (use safe path to avoid Docker volume issues)
        safe_output_path = self._get_safe_output_path(output_path)
        ensure_directory(safe_output_path.parent)
        with open(safe_output_path, 'w', encoding='utf-8') as f:
            f.write(anonymized_text)
        
        if safe_output_path != output_path:
            logger.info(f"Anonymized document saved to safe location: {safe_output_path} (intended: {output_path})")
        else:
            logger.info(f"Anonymized document saved to {output_path}")
        return str(safe_output_path), mapping
    
    def anonymize_batch(self, input_paths: List[Union[str, Path]], 
                       output_directory: Optional[Union[str, Path]] = None,
                       shared_mapping: bool = True,
                       confidence_threshold: float = 0.3,
                       options: Optional[Union[AnonymizationOptions, Dict[str, Any]]] = None) -> Dict[str, str]:
        """Anonymize multiple files with optional shared entity mapping."""
        
        results = {}
        mapping = AnonymizationMapping() if shared_mapping else None
        
        for input_path in input_paths:
            try:
                input_path = normalize_path(input_path)
                if not input_path or not input_path.exists():
                    logger.warning(f"Skipping non-existent file: {input_path}")
                    continue
                
                # Determine output path
                if output_directory:
                    output_dir = normalize_path(output_directory)
                    ensure_directory(output_dir)
                    output_path = output_dir / f"{input_path.stem}_anonymized.txt"
                else:
                    output_path = None
                
                # Use individual mapping if not sharing
                file_mapping = mapping if shared_mapping else AnonymizationMapping()
                
                output_file, final_mapping = self.anonymize_single_file(
                    input_path, output_path, file_mapping, confidence_threshold, options
                )
                
                results[str(input_path)] = output_file
                
                # Update shared mapping
                if shared_mapping:
                    mapping = final_mapping
                    
            except Exception as e:
                logger.error(f"Failed to anonymize {input_path}: {e}", exc_info=True)
                error_msg = str(e)
                if "bad escape" in error_msg:
                    error_msg = "Regex pattern error in entity names. Some special characters may not be supported."
                results[str(input_path)] = f"ERROR: {error_msg}"
        
        # Save mapping report if shared mapping was used
        if shared_mapping and mapping and output_directory:
            mapping_file = normalize_path(output_directory) / "anonymization_mapping.txt"
            self.save_mapping_report(mapping, mapping_file)
        
        return results
    
    def save_mapping_report(self, mapping: AnonymizationMapping, 
                          output_path: Union[str, Path]) -> None:
        """Save the anonymization mapping to a file for reference."""
        
        output_path = normalize_path(output_path)
        safe_output_path = self._get_safe_output_path(output_path)
        ensure_directory(safe_output_path.parent)
        
        with open(safe_output_path, 'w', encoding='utf-8') as f:
            f.write("ANONYMIZATION MAPPING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write("This file contains the mapping between original entities and their anonymous replacements.\n")
            f.write("Keep this file secure and separate from anonymized documents.\n\n")
            
            f.write("MAPPINGS:\n")
            f.write("-" * 20 + "\n")
            
            # Group by entity type
            by_type = defaultdict(list)
            for original, anonymous in mapping.mappings.items():
                # Determine type from anonymous name
                if anonymous.startswith("Person"):
                    by_type["People"].append((original, anonymous))
                elif anonymous.startswith("Company"):
                    by_type["Organizations"].append((original, anonymous))
                elif anonymous.startswith("Project"):
                    by_type["Projects"].append((original, anonymous))
                elif anonymous.startswith("Location"):
                    by_type["Locations"].append((original, anonymous))
                else:
                    by_type["Other"].append((original, anonymous))
            
            for entity_type, mappings in by_type.items():
                f.write(f"\n{entity_type}:\n")
                for original, anonymous in sorted(mappings):
                    f.write(f"  {original} -> {anonymous}\n")
        
        logger.info(f"Anonymization mapping saved to {output_path}")

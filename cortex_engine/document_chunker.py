"""
Document Chunker
Version: 1.0.0
Date: 2026-01-06

Purpose: Break large tender documents into manageable chunks for LLM analysis.
Strategy: Section-aware chunking that respects document structure.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentSection:
    """Represents a section in the document."""
    title: str
    start_line: int
    end_line: int
    section_type: str  # 'company', 'personnel', 'project', 'other'
    content: str


@dataclass
class DocumentChunk:
    """Represents a reviewable chunk of the document."""
    chunk_id: int
    title: str
    start_line: int
    end_line: int
    content: str
    char_count: int
    section_types: List[str]
    is_completable: bool  # True if contains fields user needs to complete


class DocumentChunker:
    """Intelligent document chunking for tender response workflow."""

    def __init__(self, target_chunk_size: int = 4000, max_chunk_size: int = 6000):
        """
        Initialize chunker.

        Args:
            target_chunk_size: Target characters per chunk (default: 4000)
            max_chunk_size: Maximum characters per chunk (default: 6000)
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size

        # Section heading patterns
        self.section_patterns = [
            r'^\s*#{1,3}\s+(.+)$',  # Markdown headers
            r'^\s*SECTION\s+\d+[:.]\s*(.+)$',  # "SECTION 4: Title"
            r'^\s*PART\s+[A-Z0-9]+[:.]\s*(.+)$',  # "PART A: Title"
            r'^\s*\d+\.\s+([A-Z][^.]+)$',  # "4. TITLE" (all caps)
            r'^\s*\d+\.\d+\s+([A-Z][^.]+)$',  # "4.1 TITLE"
        ]

        # Personnel section keywords
        self.personnel_keywords = [
            'personnel', 'team member', 'staff', 'key personnel',
            'specified personnel', 'nominated personnel', 'cv', 'resume',
            'qualifications', 'experience', 'individual', 'person'
        ]

        # Company section keywords
        self.company_keywords = [
            'company', 'business', 'organization', 'tenderer details',
            'legal entity', 'abn', 'acn', 'registered office'
        ]

        logger.info(f"DocumentChunker initialized (target: {target_chunk_size}, max: {max_chunk_size})")

    def identify_sections(self, document_text: str) -> List[DocumentSection]:
        """
        Identify all sections in the document.

        Args:
            document_text: Full document text

        Returns:
            List of DocumentSection objects
        """
        lines = document_text.split('\n')
        sections = []
        current_section_start = 0
        current_section_title = "Introduction"

        for line_num, line in enumerate(lines):
            # Check if this line is a section heading
            is_heading = False
            heading_text = None

            for pattern in self.section_patterns:
                match = re.match(pattern, line.strip(), re.IGNORECASE)
                if match:
                    is_heading = True
                    heading_text = match.group(1).strip()
                    break

            # Check for all-caps lines (common section headers)
            if not is_heading and line.strip() and len(line.strip()) > 10:
                if line.strip().isupper() and not line.strip().endswith(':'):
                    is_heading = True
                    heading_text = line.strip()

            if is_heading and heading_text:
                # Save previous section
                if line_num > current_section_start:
                    section_content = '\n'.join(lines[current_section_start:line_num])
                    section_type = self._classify_section(current_section_title, section_content)

                    sections.append(DocumentSection(
                        title=current_section_title,
                        start_line=current_section_start,
                        end_line=line_num - 1,
                        section_type=section_type,
                        content=section_content
                    ))

                # Start new section
                current_section_start = line_num
                current_section_title = heading_text

        # Add final section
        if current_section_start < len(lines):
            section_content = '\n'.join(lines[current_section_start:])
            section_type = self._classify_section(current_section_title, section_content)

            sections.append(DocumentSection(
                title=current_section_title,
                start_line=current_section_start,
                end_line=len(lines) - 1,
                section_type=section_type,
                content=section_content
            ))

        logger.info(f"Identified {len(sections)} sections in document")
        return sections

    def _classify_section(self, title: str, content: str) -> str:
        """
        Classify section type based on title and content.

        Args:
            title: Section title
            content: Section content

        Returns:
            Section type: 'company', 'personnel', 'project', 'other'
        """
        title_lower = title.lower()
        content_lower = content.lower()

        # Check for personnel section
        if any(kw in title_lower for kw in self.personnel_keywords):
            return 'personnel'

        if any(kw in content_lower for kw in ['surname', 'first name', 'repeat as required for each']):
            return 'personnel'

        # Check for company section
        if any(kw in title_lower for kw in self.company_keywords):
            return 'company'

        # Check for project examples
        if any(kw in title_lower for kw in ['project', 'case study', 'example', 'experience']):
            return 'project'

        return 'other'

    def create_chunks(
        self,
        document_text: str,
        start_section: Optional[str] = None,
        end_section: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Create chunks from document, optionally filtering by section range.

        Args:
            document_text: Full document text
            start_section: Optional section name to start from
            end_section: Optional section name to end at

        Returns:
            List of DocumentChunk objects
        """
        sections = self.identify_sections(document_text)

        # Filter sections if range specified
        if start_section or end_section:
            filtered_sections = []
            include = not start_section  # Start including if no start_section

            for section in sections:
                if start_section and start_section.lower() in section.title.lower():
                    include = True

                if include:
                    filtered_sections.append(section)

                if end_section and end_section.lower() in section.title.lower():
                    break

            sections = filtered_sections

        # Create chunks
        chunks = []
        chunk_id = 1
        current_chunk_sections = []
        current_chunk_size = 0

        for section in sections:
            section_size = len(section.content)

            # If single section exceeds max size, split it
            if section_size > self.max_chunk_size:
                # First, save any accumulated sections
                if current_chunk_sections:
                    chunks.append(self._build_chunk(chunk_id, current_chunk_sections))
                    chunk_id += 1
                    current_chunk_sections = []
                    current_chunk_size = 0

                # Split large section
                sub_chunks = self._split_large_section(section, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)

            # If adding this section would exceed target, create chunk
            elif current_chunk_size + section_size > self.target_chunk_size and current_chunk_sections:
                chunks.append(self._build_chunk(chunk_id, current_chunk_sections))
                chunk_id += 1
                current_chunk_sections = [section]
                current_chunk_size = section_size

            # Otherwise, accumulate
            else:
                current_chunk_sections.append(section)
                current_chunk_size += section_size

        # Add remaining sections
        if current_chunk_sections:
            chunks.append(self._build_chunk(chunk_id, current_chunk_sections))

        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks

    def _build_chunk(self, chunk_id: int, sections: List[DocumentSection]) -> DocumentChunk:
        """Build a chunk from multiple sections."""
        if not sections:
            raise ValueError("Cannot build chunk from empty sections list")

        title = sections[0].title if len(sections) == 1 else f"{sections[0].title} + {len(sections) - 1} more"
        content = '\n\n'.join(s.content for s in sections)
        section_types = list(set(s.section_type for s in sections))

        # Chunk is completable if it contains company/project sections (not personnel)
        is_completable = any(t in ['company', 'project'] for t in section_types)

        return DocumentChunk(
            chunk_id=chunk_id,
            title=title,
            start_line=sections[0].start_line,
            end_line=sections[-1].end_line,
            content=content,
            char_count=len(content),
            section_types=section_types,
            is_completable=is_completable
        )

    def _split_large_section(self, section: DocumentSection, start_chunk_id: int) -> List[DocumentChunk]:
        """Split a section that's too large into multiple chunks."""
        chunks = []
        lines = section.content.split('\n')
        chunk_lines = []
        chunk_size = 0
        chunk_id = start_chunk_id

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            if chunk_size + line_size > self.target_chunk_size and chunk_lines:
                # Create chunk
                content = '\n'.join(chunk_lines)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    title=f"{section.title} (Part {chunk_id - start_chunk_id + 1})",
                    start_line=section.start_line,  # Approximate
                    end_line=section.end_line,  # Approximate
                    content=content,
                    char_count=len(content),
                    section_types=[section.section_type],
                    is_completable=section.section_type in ['company', 'project']
                ))

                chunk_id += 1
                chunk_lines = [line]
                chunk_size = line_size
            else:
                chunk_lines.append(line)
                chunk_size += line_size

        # Add remaining lines
        if chunk_lines:
            content = '\n'.join(chunk_lines)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                title=f"{section.title} (Part {chunk_id - start_chunk_id + 1})",
                start_line=section.start_line,
                end_line=section.end_line,
                content=content,
                char_count=len(content),
                section_types=[section.section_type],
                is_completable=section.section_type in ['company', 'project']
            ))

        return chunks

    def filter_completable_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Filter chunks to only those that need user completion.
        Renumbers chunks sequentially starting from 1.

        Args:
            chunks: All chunks

        Returns:
            Chunks that contain completable fields (excludes personnel sections), renumbered 1, 2, 3...
        """
        completable = [c for c in chunks if c.is_completable]

        # Renumber chunks sequentially (1, 2, 3...) for easier navigation
        renumbered = []
        for new_id, chunk in enumerate(completable, start=1):
            renumbered_chunk = DocumentChunk(
                chunk_id=new_id,
                title=chunk.title,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
                char_count=chunk.char_count,
                section_types=chunk.section_types,
                is_completable=chunk.is_completable
            )
            renumbered.append(renumbered_chunk)

        logger.info(f"Filtered to {len(renumbered)} completable chunks (from {len(chunks)} total), renumbered 1-{len(renumbered)}")
        return renumbered

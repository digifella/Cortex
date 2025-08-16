# ## File: cortex_engine/instruction_parser.py
# Version: 4.0.0 (Unified Parser)
# Date: 2025-07-15
# Purpose: A unified module to parse .docx files and define instruction structures.
#          - CRITICAL FIX (v4.0.0): Restored the 'parse_template_for_instructions'
#            function required by the Proposal Co-pilot, which was erroneously
#            removed during previous refactoring. This file now contains all
#            necessary parsing functions for the entire suite.

import re
import docx
import io
from typing import List, Optional, NamedTuple, Dict

# This is the instruction object used by the original, powerful co-pilot.
class CortexInstruction(NamedTuple):
    section_heading: str
    instruction_raw: str
    task_type: str
    parameter: str
    placeholder_paragraph: docx.text.paragraph.Paragraph
    sub_instruction: Optional[str] = None

INSTRUCTION_REGEX = re.compile(r"\[([A-Z_]+)(?:::(.*?))?\]")

def parse_template_for_instructions(doc: docx.document.Document) -> List[CortexInstruction]:
    """
    (RESTORED) Parses a docx Document object to find Cortex instructions
    and sub-instructions for the Proposal Co-pilot.
    """
    instructions: List[CortexInstruction] = []
    paragraphs = list(doc.paragraphs)

    for i, p in enumerate(paragraphs):
        p_text = p.text.strip()
        match = INSTRUCTION_REGEX.search(p_text)

        if match:
            task_type, parameter = match.groups()
            parameter = parameter or ""
            heading_text = "Unknown Section"
            sub_instruction_text = None

            if (i + 1) < len(paragraphs):
                next_p_text = paragraphs[i + 1].text.strip()
                if next_p_text.startswith("##"):
                    sub_instruction_text = next_p_text.lstrip('#- ').strip()

            for j in range(i - 1, -1, -1):
                prev_p = paragraphs[j]
                prev_p_text = prev_p.text.strip()
                is_heading_style = prev_p.style.name.startswith('Heading')
                is_heading_text_marker = prev_p_text.lower().startswith('section:')

                if is_heading_style or is_heading_text_marker:
                    heading_text_raw = prev_p_text.split(':', 1)[1] if is_heading_text_marker else prev_p_text
                    heading_text = heading_text_raw.strip().strip(' "\'■\t*-')
                    break

            instruction = CortexInstruction(
                section_heading=heading_text,
                instruction_raw=p_text,
                task_type=task_type.strip(),
                parameter=parameter.strip(),
                placeholder_paragraph=p,
                sub_instruction=sub_instruction_text
            )
            instructions.append(instruction)

    return instructions

# This function is used by the new Template Editor.
def iter_block_items(parent):
    """
    Yield each paragraph and table child within *parent*, in document order.
    """
    if isinstance(parent, docx.document.Document):
        parent_elm = parent.element.body
    elif isinstance(parent, docx.table._Cell):
        parent_elm = parent._tc
    else:
        try:
            parent_elm = parent._element
        except AttributeError:
            raise ValueError("Unsupported parent type for iter_block_items")

    for child in parent_elm.iterchildren():
        if isinstance(child, docx.oxml.text.paragraph.CT_P):
            yield docx.text.paragraph.Paragraph(child, parent)
        elif isinstance(child, docx.oxml.table.CT_Tbl):
            yield docx.table.Table(child, parent)
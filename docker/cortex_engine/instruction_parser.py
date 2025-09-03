from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from dataclasses import dataclass
from typing import List, Optional, Union
import io
import docx


def iter_block_items(parent):
    """Yield each paragraph and table child within parent, in document order.

    Works for a Document or _Cell parent. Based on python-docx cookbook pattern.
    """
    if hasattr(parent, "element") and hasattr(parent.element, "body"):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


@dataclass
class CortexInstruction:
    section_heading: str
    instruction_raw: str
    task_type: str
    parameter: Optional[str] = ""
    placeholder_paragraph: Optional[str] = ""
    sub_instruction: Optional[str] = ""


def parse_template_for_instructions(doc: Union[bytes, 'docx.document.Document']) -> List[CortexInstruction]:
    """Minimal parser that scans a .docx for bracketed Cortex placeholders.

    Looks for tokens like [GENERATE_FROM_KB_ONLY], [PROMPT_HUMAN], etc. and returns a list.
    """
    instructions: List[CortexInstruction] = []
    try:
        if isinstance(doc, bytes):
            d = docx.Document(io.BytesIO(doc))
        else:
            d = doc
        current_heading = "Document"
        for block in iter_block_items(d):
            text = ""
            if isinstance(block, Paragraph):
                text = block.text or ""
                # Simple heading detection
                try:
                    if block.style and block.style.name and block.style.name.startswith('Heading'):
                        current_heading = text or current_heading
                        continue
                except Exception:
                    pass
            elif isinstance(block, Table):
                for r in block.rows:
                    for c in r.cells:
                        if c.text:
                            text += c.text + "\n"
            if "[" in text and "]" in text:
                # Extract simple [TOKEN] markers
                parts = text.split("[")
                for part in parts[1:]:
                    token = part.split("]", 1)[0].strip()
                    if token:
                        # Map token to task_type; default pass-through
                        task = token
                        instructions.append(CortexInstruction(
                            section_heading=current_heading,
                            instruction_raw=token,
                            task_type=task,
                            parameter="",
                            placeholder_paragraph=text.strip(),
                            sub_instruction="",
                        ))
    except Exception:
        pass
    return instructions

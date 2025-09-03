from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    confidence: float = 0.5
    aliases: List[str] = field(default_factory=list)
    extraction_method: str = "stub"
    context_mentions: List[str] = field(default_factory=list)


@dataclass
class ExtractedRelationship:
    source: str
    target: str
    relationship_type: str = "related_to"
    confidence: float = 0.5
    context: str = ""
    source_entity_type: str = "entity"
    target_entity_type: str = "entity"


class EntityExtractor:
    def extract_entities_and_relationships(self, text: str, metadata: Dict) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        # Minimal stub: return empty results
        return [], []


# cortex_engine/entity_extractor.py
import re
from typing import List, Dict, Tuple, Set, Optional
from pydantic import BaseModel, Field
import spacy
from collections import defaultdict, Counter
import logging
from cortex_engine.utils import get_logger

logger = get_logger(__name__)

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str  # 'person', 'organization', 'project', 'report'
    aliases: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    context_mentions: List[str] = Field(default_factory=list)
    extraction_method: str = "pattern"  # 'spacy', 'pattern', 'context'
    
class ExtractedRelationship(BaseModel):
    source: str
    target: str
    relationship_type: str  # 'worked_on', 'authored', 'client_of', 'collaborated_with'
    context: str = ""
    confidence: float = 1.0
    evidence: List[str] = Field(default_factory=list)
    strength_indicators: Dict[str, float] = Field(default_factory=dict)

class EntityExtractor:
    def __init__(self):
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logging.info("Loaded spaCy model successfully")
        except OSError as e:
            logging.warning(f"spaCy model not found: {e}. Attempting to download...")
            try:
                import subprocess
                result = subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to download spaCy model: {result.stderr}")
                self.nlp = spacy.load("en_core_web_sm")
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, RuntimeError) as download_error:
                logging.error(f"Failed to download or load spaCy model: {download_error}")
                raise RuntimeError("Cannot initialize EntityExtractor without spaCy model") from download_error
        
        # Pattern matching for common document patterns
        self.consultant_patterns = [
            r"(?:consultant|author|prepared by|written by|compiled by|lead|manager)[:;\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\((?:consultant|author|lead|manager)\)",
            r"(?:Project Manager|Technical Lead|Consultant)[:;\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ]
        
        self.client_patterns = [
            r"(?:client|customer|for|prepared for|submitted to)[:;\s]+([A-Z][a-z]+(?:\s+(?:&\s+)?[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\(client\)",
        ]
        
        self.project_patterns = [
            r"(?:project|initiative|program)[:;\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Project|Initiative|Program)",
        ]
        
        # Enhanced contextual patterns for relationship inference
        self.collaboration_indicators = [
            r"(?:collaborated with|worked with|partnered with|co-authored with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?:team included|team members|contributors)[:;\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+and\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:worked|collaborated|contributed)"
        ]
        
        self.expertise_indicators = [
            r"(?:expert in|specialist in|experienced in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?:consultant for|advisor to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ]
    
    def normalize_entity_name(self, name: str) -> str:
        """Normalize entity names to handle variations."""
        # Remove extra whitespace
        name = ' '.join(name.split())
        # Remove common suffixes
        name = re.sub(r'\s+(Inc\.|LLC|Ltd\.|Limited|Corporation|Corp\.)$', '', name, flags=re.IGNORECASE)
        # Remove common prefixes
        name = re.sub(r'^(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+', '', name, flags=re.IGNORECASE)
        return name.strip()
    
    def calculate_entity_confidence(self, entity_name: str, extraction_method: str, 
                                  context_evidence: List[str]) -> float:
        """Calculate confidence score for entity extraction."""
        base_confidence = {
            'spacy': 0.8,
            'pattern': 0.7,
            'context': 0.6
        }.get(extraction_method, 0.5)
        
        # Boost confidence based on evidence
        evidence_boost = min(0.2, len(context_evidence) * 0.05)
        
        # Penalize very short or very long names
        name_penalty = 0.0
        if len(entity_name.split()) == 1 and len(entity_name) < 4:
            name_penalty = 0.2  # Very short names are likely false positives
        elif len(entity_name.split()) > 5:
            name_penalty = 0.1  # Very long names might be extraction errors
        
        final_confidence = min(1.0, base_confidence + evidence_boost - name_penalty)
        return final_confidence
    
    def extract_contextual_relationships(self, text: str, entities: Dict[str, ExtractedEntity]) -> List[ExtractedRelationship]:
        """Extract relationships using contextual analysis."""
        relationships = []
        text_sentences = re.split(r'[.!?]', text)
        
        entity_names = {entity.name.lower(): entity for entity in entities.values()}
        
        for sentence in text_sentences[:50]:  # Limit for performance
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Find collaboration patterns
            for pattern in self.collaboration_indicators:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    # Extract collaborators from the match
                    collaborator_text = match.group(1)
                    collaborators = [name.strip() for name in re.split(r',|and', collaborator_text)]
                    
                    # Create collaboration relationships
                    for i, collab1 in enumerate(collaborators):
                        collab1_normalized = self.normalize_entity_name(collab1).lower()
                        if collab1_normalized in entity_names:
                            for collab2 in collaborators[i+1:]:
                                collab2_normalized = self.normalize_entity_name(collab2).lower()
                                if collab2_normalized in entity_names:
                                    relationships.append(ExtractedRelationship(
                                        source=entity_names[collab1_normalized].name,
                                        target=entity_names[collab2_normalized].name,
                                        relationship_type='collaborated_with',
                                        context=sentence,
                                        confidence=0.8,
                                        evidence=[sentence],
                                        strength_indicators={'contextual_mention': 1.0}
                                    ))
        
        return relationships
    
    def resolve_entity_disambiguation(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Resolve entity disambiguation by merging similar entities."""
        resolved_entities = []
        entity_clusters = defaultdict(list)
        
        # Group similar entities
        for entity in entities:
            normalized_name = entity.name.lower().replace(' ', '')
            # Simple fuzzy matching based on normalized names
            matched_cluster = None
            
            for cluster_key in entity_clusters:
                if self._names_are_similar(normalized_name, cluster_key):
                    matched_cluster = cluster_key
                    break
            
            if matched_cluster:
                entity_clusters[matched_cluster].append(entity)
            else:
                entity_clusters[normalized_name].append(entity)
        
        # Merge entities in each cluster
        for cluster_entities in entity_clusters.values():
            if len(cluster_entities) == 1:
                resolved_entities.append(cluster_entities[0])
            else:
                # Merge multiple entities
                merged_entity = self._merge_entities(cluster_entities)
                resolved_entities.append(merged_entity)
        
        return resolved_entities
    
    def _names_are_similar(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """Check if two normalized names are similar enough to be the same entity."""
        # Simple similarity check - can be enhanced with more sophisticated algorithms
        if name1 == name2:
            return True
        
        # Check if one is a substring of the other (for abbreviations)
        if len(name1) > 3 and len(name2) > 3:
            if name1 in name2 or name2 in name1:
                return True
        
        # Check character overlap (simple Jaccard similarity)
        set1, set2 = set(name1), set(name2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union > 0:
            similarity = intersection / union
            return similarity >= threshold
        
        return False
    
    def _merge_entities(self, entities: List[ExtractedEntity]) -> ExtractedEntity:
        """Merge multiple entities into one, keeping the best attributes."""
        # Choose the entity with highest confidence as base
        base_entity = max(entities, key=lambda e: e.confidence)
        
        # Collect all aliases and context mentions
        all_aliases = set(base_entity.aliases)
        all_contexts = list(base_entity.context_mentions)
        
        for entity in entities:
            if entity != base_entity:
                all_aliases.update(entity.aliases)
                all_aliases.add(entity.name)  # Add the name as an alias
                all_contexts.extend(entity.context_mentions)
        
        # Calculate merged confidence (average weighted by evidence)
        total_evidence = sum(len(e.context_mentions) + 1 for e in entities)
        weighted_confidence = sum(e.confidence * (len(e.context_mentions) + 1) for e in entities) / total_evidence
        
        return ExtractedEntity(
            name=base_entity.name,
            entity_type=base_entity.entity_type,
            aliases=list(all_aliases),
            confidence=weighted_confidence,
            context_mentions=all_contexts[:10],  # Limit to prevent bloat
            extraction_method=base_entity.extraction_method
        )
    
    def extract_entities_and_relationships(self, 
                                         text: str, 
                                         metadata: Dict) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Extract entities and relationships from document text and metadata."""
        entities = []
        relationships = []
        entity_map = {}  # name -> ExtractedEntity
        
        # Use filename and metadata for initial context
        doc_type = metadata.get('document_type', '')
        filename = metadata.get('file_name', '')
        summary = metadata.get('summary', '')
        
        # Extract from structured metadata
        if 'thematic_tags' in metadata:
            for tag in metadata['thematic_tags']:
                if any(keyword in tag.lower() for keyword in ['project', 'initiative', 'program']):
                    entity = ExtractedEntity(
                        name=tag,
                        entity_type='project'
                    )
                    entities.append(entity)
                    entity_map[tag] = entity
        
        # Combine text sources for extraction
        combined_text = f"{filename}\n{summary}\n{text[:10000]}"
        
        # Extract using NER
        try:
            doc = self.nlp(combined_text)
            
            persons = set()
            organizations = set()
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = self.normalize_entity_name(ent.text)
                    if len(name.split()) <= 4:  # Avoid false positives with very long names
                        persons.add(name)
                elif ent.label_ == "ORG":
                    name = self.normalize_entity_name(ent.text)
                    organizations.add(name)
        except Exception as e:
            logging.error(f"spaCy NER failed: {e}")
        
        # Extract using patterns
        text_for_patterns = combined_text[:5000]  # Focus on document header
        
        for pattern in self.consultant_patterns:
            matches = re.findall(pattern, text_for_patterns, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                name = self.normalize_entity_name(match)
                if len(name.split()) <= 4:
                    persons.add(name)
        
        for pattern in self.client_patterns:
            matches = re.findall(pattern, text_for_patterns, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                name = self.normalize_entity_name(match)
                organizations.add(name)
        
        for pattern in self.project_patterns:
            matches = re.findall(pattern, text_for_patterns, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                name = self.normalize_entity_name(match)
                if name not in entity_map:
                    entity = ExtractedEntity(
                        name=name,
                        entity_type='project'
                    )
                    entities.append(entity)
                    entity_map[name] = entity
        
        # Create entities for persons with confidence scoring
        for person in persons:
            if person not in entity_map:
                confidence = self.calculate_entity_confidence(person, 'spacy', [summary])
                entity = ExtractedEntity(
                    name=person,
                    entity_type='person',
                    confidence=confidence,
                    context_mentions=[summary] if summary else [],
                    extraction_method='spacy'
                )
                entities.append(entity)
                entity_map[person] = entity
        
        # Create entities for organizations with confidence scoring
        for org in organizations:
            if org not in entity_map:
                confidence = self.calculate_entity_confidence(org, 'spacy', [summary])
                entity = ExtractedEntity(
                    name=org,
                    entity_type='organization',
                    confidence=confidence,
                    context_mentions=[summary] if summary else [],
                    extraction_method='spacy'
                )
                entities.append(entity)
                entity_map[org] = entity
        
        # Extract relationships based on document type
        if doc_type in ['Project Plan', 'Final Report', 'Draft Report', 'Proposal/Quote', 'Technical Documentation']:
            # Link consultants to documents
            for person in persons:
                relationships.append(ExtractedRelationship(
                    source=person,
                    target=filename,
                    relationship_type='authored'
                ))
            
            # Link organizations as clients
            for org in organizations:
                # Try to determine if it's a client based on context
                org_pattern = re.escape(org)
                if re.search(rf"(?:for|client|customer)[\s:]*{org_pattern}", text_for_patterns, re.IGNORECASE):
                    relationships.append(ExtractedRelationship(
                        source=org,
                        target=filename,
                        relationship_type='client_of'
                    ))
        
        # Extract collaboration relationships
        if len(persons) > 1:
            person_list = list(persons)
            for i in range(len(person_list)):
                for j in range(i+1, len(person_list)):
                    relationships.append(ExtractedRelationship(
                        source=person_list[i],
                        target=person_list[j],
                        relationship_type='collaborated_with',
                        context=filename
                    ))
        
        # Link projects to documents
        for entity in entity_map.values():
            if entity.entity_type == 'project':
                relationships.append(ExtractedRelationship(
                    source=entity.name,
                    target=filename,
                    relationship_type='documented_in'
                ))
        
        # Extract contextual relationships
        contextual_relationships = self.extract_contextual_relationships(combined_text, entity_map)
        relationships.extend(contextual_relationships)
        
        # Resolve entity disambiguation
        resolved_entities = self.resolve_entity_disambiguation(entities)
        
        # Calculate relationship confidence and strength
        enhanced_relationships = []
        for rel in relationships:
            # Calculate relationship strength based on evidence
            strength_score = 1.0
            if rel.evidence:
                strength_score += len(rel.evidence) * 0.1
            if rel.context:
                strength_score += 0.2
            
            rel.strength_indicators['calculated_strength'] = min(2.0, strength_score)
            enhanced_relationships.append(rel)
        
        logger.info(f"Extracted {len(resolved_entities)} entities and {len(enhanced_relationships)} relationships")
        return resolved_entities, enhanced_relationships
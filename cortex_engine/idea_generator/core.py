# ## File: cortex_engine/idea_generator/core.py  
# Version: 1.0.0
# Date: 2025-08-08
# Purpose: Core IdeaGenerator class - extracted from monolithic module

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..collection_manager import WorkingCollectionManager
from ..graph_manager import EnhancedGraphManager
from ..utils.logging_utils import get_logger
from ..theme_visualizer import ThemeNetworkVisualizer
from .double_diamond import DoubleDiamondProcessor
from .agents import IdeationAgents
from .export import IdeaExporter

logger = get_logger(__name__)

class IdeaGenerator:
    """
    Core engine for generating innovative ideas from knowledge collections.
    
    Implements the Double Diamond methodology:
    - Discover: Analyze collection for themes and opportunities
    - Define: Formulate specific problem statements  
    - Develop: Generate diverse solutions using multi-agent ideation
    - Deliver: Create structured, actionable reports
    """
    
    def __init__(self, vector_index=None, graph_manager=None):
        """Initialize the Idea Generator with required components."""
        self.collection_mgr = WorkingCollectionManager()
        self.vector_index = vector_index
        self.graph_manager = graph_manager
        self.theme_visualizer = ThemeNetworkVisualizer()
        
        # Initialize specialized processors
        self.diamond_processor = DoubleDiamondProcessor(vector_index, graph_manager)
        self.ideation_agents = IdeationAgents()
        self.exporter = IdeaExporter()
        
    def run_discovery(self, collection_name: str, seed_ideas: str = "", 
                     constraints: str = "", goals: str = "", 
                     research: bool = False, llm_provider: str = "Local (Ollama)",
                     filters: Optional[Dict[str, Any]] = None,
                     selected_themes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute the Discovery phase of the Double Diamond process.
        
        Args:
            collection_name: Name of the working collection to analyze
            seed_ideas: Initial ideas or themes to guide analysis
            constraints: Limitations or boundaries to consider
            goals: Innovation objectives and desired outcomes
            research: Whether to supplement with web research
            llm_provider: AI model to use for analysis
            filters: Optional filtering criteria
            selected_themes: Pre-selected themes to focus on
            
        Returns:
            Dict containing discovery results with themes, opportunities, and analysis
        """
        return self.diamond_processor.run_discovery(
            collection_name, seed_ideas, constraints, goals, 
            research, llm_provider, filters, selected_themes
        )
    
    def run_define(self, discovery_results: Dict[str, Any], 
                   focus_themes: List[str], problem_scope: str = "",
                   llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """Execute the Define phase of the Double Diamond process."""
        return self.diamond_processor.run_define(
            discovery_results, focus_themes, problem_scope, llm_provider
        )
    
    def run_develop(self, define_results: Dict[str, Any],
                    innovation_approach: str = "incremental",
                    ideation_modes: List[str] = None,
                    llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """Execute the Develop phase of the Double Diamond process."""
        return self.diamond_processor.run_develop(
            define_results, innovation_approach, ideation_modes, llm_provider
        )
    
    def run_deliver(self, develop_results: Dict[str, Any],
                    implementation_focus: str = "balanced",
                    llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """Execute the Deliver phase of the Double Diamond process."""
        return self.diamond_processor.run_deliver(
            develop_results, implementation_focus, llm_provider
        )
        
    def export_results(self, phase_results: Dict[str, Any], 
                      output_dir: str, filename_prefix: str = "idea_session") -> Dict[str, str]:
        """Export idea generation results to files."""
        return self.exporter.export_results(phase_results, output_dir, filename_prefix)
    
    def _validate_collection(self, collection_name: str) -> bool:
        """Validate that collection exists and has content."""
        collections = self.collection_mgr.get_collection_names()
        if collection_name not in collections:
            logger.warning(f"Collection '{collection_name}' not found")
            return False
            
        doc_ids = self.collection_mgr.get_doc_ids_by_name(collection_name)
        if not doc_ids:
            logger.warning(f"Collection '{collection_name}' is empty")
            return False
            
        return True
    
    def _get_collection_content(self, collection_name: str, 
                              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve and optionally filter collection document content."""
        try:
            # Get document IDs for the collection
            doc_ids = self.collection_mgr.get_doc_ids_by_name(collection_name)
            if not doc_ids:
                return []
            
            # If no vector index, create basic document entries for analysis
            if not self.vector_index:
                logger.info("No vector index available - creating basic document entries for analysis")
                collection_docs = []
                for doc_id in doc_ids:
                    # Create basic document structure for analysis
                    doc_content = {
                        "doc_id": doc_id,
                        "title": doc_id,  # Use doc_id as title when we can't get metadata
                        "content": f"Document {doc_id}",  # Basic content placeholder
                        "document_type": "Unknown",  # Default values for analysis
                        "proposal_outcome": "N/A",
                        "thematic_tags": [],
                        "entities": [],
                        "metadata": {"doc_id": doc_id}
                    }
                    collection_docs.append(doc_content)
                return collection_docs
            
            collection_docs = []
            
            # Try to get real documents from vector store with actual metadata
            logger.info("Attempting to retrieve real documents with metadata from vector store")
            retriever = self.vector_index.as_retriever(similarity_top_k=100)  # Get many results
            
            try:
                # Do a broad search to get documents from the vector store
                all_results = retriever.retrieve("document")  # Generic search to get documents
                
                # Filter results to only include documents in our collection
                collection_doc_set = set(doc_ids)
                
                for node in all_results:
                    if node.metadata.get("doc_id") in collection_doc_set:
                        doc_content = {
                            "doc_id": node.metadata.get("doc_id"),
                            "title": node.metadata.get("file_name", "Unknown"),
                            "content": node.text[:2000],  # Limit content size
                            "document_type": node.metadata.get("document_type", "Unknown"),
                            "proposal_outcome": node.metadata.get("proposal_outcome", "N/A"),
                            "thematic_tags": node.metadata.get("thematic_tags", "").split(", ") if node.metadata.get("thematic_tags") else [],
                            "entities": node.metadata.get("extracted_entities", []),
                            "metadata": node.metadata
                        }
                        
                        # Apply filters if provided
                        if filters and not self._passes_filters(doc_content, filters):
                            continue
                            
                        collection_docs.append(doc_content)
                        
                        # Remove from set so we know we found it
                        collection_doc_set.discard(node.metadata.get("doc_id"))
                
                # For any documents we couldn't find in vector store, create basic entries
                for remaining_doc_id in collection_doc_set:
                    logger.warning(f"Could not find document {remaining_doc_id} in vector store, creating basic entry")
                    doc_content = {
                        "doc_id": remaining_doc_id,
                        "title": f"Document {remaining_doc_id[:8]}...",
                        "content": f"Document content for {remaining_doc_id}",
                        "document_type": "Document",
                        "proposal_outcome": "N/A",
                        "thematic_tags": [],
                        "entities": [],
                        "metadata": {"doc_id": remaining_doc_id}
                    }
                    collection_docs.append(doc_content)
                
                logger.info(f"Retrieved {len(collection_docs)} documents with metadata from vector store")
                
            except Exception as e:
                logger.warning(f"Failed to retrieve from vector store: {e}, falling back to basic documents")
                # Fallback to basic document creation
                for doc_id in doc_ids:
                    doc_content = {
                        "doc_id": doc_id,
                        "title": f"Document {doc_id[:8]}...",
                        "content": f"Document content for {doc_id}",
                        "document_type": "Document",
                        "proposal_outcome": "N/A",
                        "thematic_tags": [],
                        "entities": [],
                        "metadata": {"doc_id": doc_id}
                    }
                    collection_docs.append(doc_content)
            
            return collection_docs
            
        except Exception as e:
            logger.error(f"Error retrieving collection content: {e}")
            return []
    
    def analyze_collection_for_filters(self, collection_name: str) -> Dict[str, Any]:
        """
        Analyze a collection to extract filter options and statistics.
        
        Args:
            collection_name: Name of the working collection to analyze
            
        Returns:
            Dict containing collection analysis with filter options
        """
        try:
            logger.info(f"Starting analysis for collection '{collection_name}'")
            
            # Validate collection exists
            if not self._validate_collection(collection_name):
                return {"error": f"Collection '{collection_name}' is not valid or empty"}
            
            # Get all documents in collection
            collection_docs = self._get_collection_content(collection_name)
            
            if not collection_docs:
                return {"error": f"No documents found in collection '{collection_name}'"}
            
            # Extract filter options and statistics
            doc_types = set()
            proposal_outcomes = set()
            thematic_tags = set()
            clients = set()
            consultants = set()
            
            for doc in collection_docs:
                # Document types
                if doc.get("document_type"):
                    doc_types.add(doc["document_type"])
                
                # Proposal outcomes
                if doc.get("proposal_outcome") and doc["proposal_outcome"] != "N/A":
                    proposal_outcomes.add(doc["proposal_outcome"])
                
                # Thematic tags
                tags = doc.get("thematic_tags", [])
                if isinstance(tags, list):
                    thematic_tags.update(tags)
                
                # Extract entities for client/consultant filters
                entities = doc.get("entities", [])
                if isinstance(entities, list):
                    for entity in entities:
                        if isinstance(entity, dict):
                            entity_type = entity.get("entity_type")
                            entity_name = entity.get("name", "")
                            
                            if entity_type == "organization" and entity_name:
                                clients.add(entity_name)
                            elif entity_type == "person" and entity_name:
                                consultants.add(entity_name)
            
            # Return analysis results
            analysis = {
                "total_documents": len(collection_docs),
                "document_types": sorted(list(doc_types)),
                "proposal_outcomes": sorted(list(proposal_outcomes)),
                "thematic_tags": sorted(list(thematic_tags)),
                "clients": sorted(list(clients)),
                "consultants": sorted(list(consultants)),
                "success": True
            }
            
            logger.info(f"Analysis complete: {len(collection_docs)} documents, "
                       f"{len(doc_types)} document types, {len(thematic_tags)} unique tags")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Exception in analyze_collection_for_filters: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def generate_intelligent_themes(self, collection_docs: List[Dict[str, Any]], llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """
        Use LLM to analyze document content and generate intelligent, high-level themes
        for innovation and ideation purposes.
        
        Args:
            collection_docs: List of documents with content and metadata
            llm_provider: LLM provider to use for analysis
            
        Returns:
            Dict containing generated themes and analysis
        """
        try:
            logger.info(f"Starting intelligent theme generation using {llm_provider}")
            
            # Use new unified LLM service
            from ..llm_service import create_llm_service
            from ..exceptions import ModelError
            
            service_manager = create_llm_service("ideation", llm_provider)
            
            try:
                llm = service_manager.get_llm()
            except ModelError as e:
                logger.error(f"LLM initialization failed: {e}")
                return {"error": str(e)}
            
            # Prepare document summaries for LLM analysis
            doc_summaries = []
            for doc in collection_docs[:10]:  # Limit to first 10 docs to avoid token limits
                summary = {
                    "title": doc.get("title", "Unknown"),
                    "document_type": doc.get("document_type", "Unknown"),
                    "content_preview": doc.get("content", "")[:1000],  # First 1000 chars
                    "existing_tags": doc.get("thematic_tags", []),
                    "summary": doc.get("metadata", {}).get("summary", "")[:500] if doc.get("metadata", {}).get("summary") else ""
                }
                doc_summaries.append(summary)
            
            # Create the LLM prompt
            prompt = self._create_theme_analysis_prompt(doc_summaries)
            
            logger.info("Sending theme analysis request to LLM...")
            response = llm.complete(prompt)
            
            # Parse the LLM response
            themes = self._parse_theme_response(str(response))
            
            logger.info(f"Generated {len(themes)} intelligent themes")
            return {
                "themes": themes,
                "total_themes": len(themes),
                "llm_provider": llm_provider,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent theme generation: {e}")
            return {"error": f"Theme generation failed: {str(e)}"}
    
    def _create_theme_analysis_prompt(self, doc_summaries: List[Dict[str, Any]]) -> str:
        """Create a comprehensive prompt for LLM-based theme analysis."""
        
        # Build document context
        doc_context = ""
        for i, doc in enumerate(doc_summaries, 1):
            doc_context += f"""
Document {i}: {doc['title']}
Type: {doc['document_type']}
Content Preview: {doc['content_preview']}
Summary: {doc['summary']}
Existing Tags: {', '.join(doc['existing_tags']) if doc['existing_tags'] else 'None'}
---
"""
        
        prompt = f"""You are an expert innovation strategist analyzing a collection of documents to identify high-level themes for ideation and innovation opportunities.

TASK: Analyze the following documents and generate 8-12 intelligent, actionable themes that capture the deeper meaning and innovation potential of this content.

DOCUMENT COLLECTION:
{doc_context}

INSTRUCTIONS:
1. Look beyond surface-level tags and metadata
2. Identify underlying concepts, methodologies, and innovation opportunities  
3. Generate themes that are:
   - Actionable for innovation projects
   - High-level but specific enough to be meaningful
   - Focused on outcomes, approaches, and opportunities
   - Suitable for ideation and development work

EXAMPLE GOOD THEMES (for context, not to copy):
- "Adult education pedagogical approaches"
- "Use of technology to increase learner engagement" 
- "How AI can increase adult education outcomes"
- "Personalized learning pathway optimization"
- "Evidence-based instructional design methodologies"

RESPONSE FORMAT:
Return ONLY a JSON list of themes, each as a string. No other text.
Example: ["Theme 1", "Theme 2", "Theme 3"]

THEMES:"""

        return prompt
    
    def _parse_theme_response(self, response_text: str) -> List[str]:
        """Parse LLM response to extract themes list."""
        try:
            import json
            import re
            
            # Try to find JSON array in the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                themes_json = json_match.group(0)
                themes = json.loads(themes_json)
                
                # Validate that we got a list of strings
                if isinstance(themes, list) and all(isinstance(theme, str) for theme in themes):
                    return [theme.strip() for theme in themes if theme.strip()]
                    
            # Fallback: try to extract themes from text format
            lines = response_text.split('\n')
            themes = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('[') and not line.startswith(']'):
                    # Remove quotes, numbers, dashes etc.
                    theme = re.sub(r'^[\d\-\*\"\'\[\]]+\s*', '', line)
                    theme = theme.strip('\'"')
                    if theme and len(theme) > 10:  # Reasonable theme length
                        themes.append(theme)
            
            return themes[:12]  # Limit to 12 themes max
            
        except Exception as e:
            logger.warning(f"Error parsing theme response: {e}")
            return ["Theme extraction failed - please try again"]

    def generate_problem_statements(self, themes: List[str], innovation_goals: str, 
                                  constraints: str = "", research_options: Dict[str, Any] = None,
                                  llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """
        Generate problem statements from selected themes and innovation parameters.
        
        Args:
            themes: List of selected themes for ideation
            innovation_goals: User's innovation goals and objectives
            constraints: Any constraints or limitations
            research_options: Research configuration options
            llm_provider: LLM provider to use
            
        Returns:
            Dict containing generated problem statements
        """
        try:
            logger.info(f"Generating problem statements from {len(themes)} themes using {llm_provider}")
            
            # Create the LLM prompt for problem statement generation
            prompt = self._create_problem_statement_prompt(themes, innovation_goals, constraints)
            
            # Use new unified LLM service
            from ..llm_service import create_llm_service
            from ..exceptions import ModelError
            
            service_manager = create_llm_service("ideation", llm_provider)
            
            try:
                llm = service_manager.get_llm()
            except ModelError as e:
                logger.error(f"LLM initialization failed: {e}")
                return {"error": str(e)}
            
            logger.info("Sending problem statement generation request to LLM...")
            response = llm.complete(prompt)
            
            # Parse the LLM response
            problem_statements = self._parse_problem_statements_response(str(response))
            
            logger.info(f"Generated {len(problem_statements)} problem statements")
            
            # Format problem statements by theme for the UI
            formatted_statements = []
            statements_per_theme = len(problem_statements) // len(themes) if themes else 1
            
            for i, theme in enumerate(themes):
                start_idx = i * statements_per_theme
                end_idx = start_idx + statements_per_theme if i < len(themes) - 1 else len(problem_statements)
                theme_statements = problem_statements[start_idx:end_idx]
                
                formatted_statements.append({
                    "theme": theme,
                    "problems": theme_statements
                })
            
            return {
                "status": "success",
                "problem_statements": formatted_statements,
                "total_statements": len(problem_statements),
                "selected_themes": themes,
                "analysis_model": llm_provider,
                "document_count": 0,  # Will be set by UI if needed
                "llm_provider": llm_provider
            }
            
        except Exception as e:
            logger.error(f"Error in problem statement generation: {e}")
            return {"error": f"Problem statement generation failed: {str(e)}"}
    
    def _create_problem_statement_prompt(self, themes: List[str], innovation_goals: str, constraints: str = "") -> str:
        """Create a prompt for generating problem statements from themes."""
        
        themes_list = "\n".join([f"- {theme}" for theme in themes])
        
        prompt = f"""You are an innovation strategist tasked with transforming themes into actionable problem statements that drive innovation.

SELECTED THEMES:
{themes_list}

INNOVATION GOALS:
{innovation_goals}

CONSTRAINTS:
{constraints if constraints else "None specified"}

TASK: Generate 5-8 specific, actionable problem statements that:
1. Combine insights from the selected themes
2. Align with the innovation goals
3. Respect any constraints mentioned
4. Are specific enough to drive concrete innovation projects
5. Use "How might we..." format for actionability

EXAMPLE GOOD PROBLEM STATEMENTS:
- "How might we leverage AI-powered personalization to improve adult learning outcomes in healthcare education?"
- "How might we design inclusive digital health platforms that accommodate diverse learning styles and accessibility needs?"
- "How might we integrate real-time patient data into medical training simulations to enhance clinical decision-making skills?"

RESPONSE FORMAT:
Return ONLY a JSON list of problem statements, each as a string starting with "How might we...". No other text.
Example: ["How might we...", "How might we...", "How might we..."]

PROBLEM STATEMENTS:"""

        return prompt
    
    def _parse_problem_statements_response(self, response_text: str) -> List[str]:
        """Parse LLM response to extract problem statements list."""
        try:
            import json
            import re
            
            # Try to find JSON array in the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                statements_json = json_match.group(0)
                statements = json.loads(statements_json)
                
                # Validate that we got a list of strings
                if isinstance(statements, list) and all(isinstance(stmt, str) for stmt in statements):
                    return [stmt.strip() for stmt in statements if stmt.strip()]
                    
            # Fallback: try to extract statements from text format
            lines = response_text.split('\n')
            statements = []
            for line in lines:
                line = line.strip()
                if line and ("how might we" in line.lower() or line.startswith('"')):
                    # Remove quotes, numbers, dashes etc.
                    statement = re.sub(r'^[\d\-\*\"\'\[\]]+\s*', '', line)
                    statement = statement.strip('\'"')
                    if statement and len(statement) > 20:  # Reasonable statement length
                        statements.append(statement)
            
            return statements[:8]  # Limit to 8 statements max
            
        except Exception as e:
            logger.warning(f"Error parsing problem statements response: {e}")
            return ["How might we transform these themes into actionable innovation opportunities?"]

    def _passes_filters(self, doc_content: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document passes the applied filters."""
        try:
            # Document type filter
            if "document_type" in filters and filters["document_type"]:
                if doc_content.get("document_type") != filters["document_type"]:
                    return False
            
            # Proposal outcome filter
            if "proposal_outcome" in filters and filters["proposal_outcome"]:
                if doc_content.get("proposal_outcome") != filters["proposal_outcome"]:
                    return False
            
            # Thematic tags filter
            if "thematic_tags" in filters and filters["thematic_tags"]:
                doc_tags = doc_content.get("thematic_tags", [])
                if not any(tag in doc_tags for tag in filters["thematic_tags"]):
                    return False
            
            # Client filter (check entities for organizations)
            if "client_filter" in filters and filters["client_filter"]:
                entities = doc_content.get("entities", [])
                client_found = any(
                    entity.get("entity_type") == "organization" and 
                    filters["client_filter"].lower() in entity.get("name", "").lower()
                    for entity in entities
                )
                if not client_found:
                    return False
            
            # Consultant filter (check entities for people)
            if "consultant_filter" in filters and filters["consultant_filter"]:
                entities = doc_content.get("entities", [])
                consultant_found = any(
                    entity.get("entity_type") == "person" and 
                    filters["consultant_filter"].lower() in entity.get("name", "").lower()
                    for entity in entities
                )
                if not consultant_found:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error applying filters: {e}")
            return True  # Include document if filter evaluation fails
    
    def get_entity_filter_options(self, collection_name: str) -> Dict[str, Any]:
        """
        Get entity-based filter options for a collection.
        
        Args:
            collection_name: Name of the working collection
            
        Returns:
            Dict containing consultant and client entity options
        """
        try:
            # Get collection content
            collection_docs = self._get_collection_content(collection_name)
            
            consultants = set()
            clients = set()
            
            for doc in collection_docs:
                entities = doc.get("entities", [])
                if isinstance(entities, list):
                    for entity in entities:
                        if isinstance(entity, dict):
                            entity_type = entity.get("entity_type")
                            entity_name = entity.get("name", "")
                            
                            if entity_type == "person" and entity_name:
                                consultants.add(entity_name)
                            elif entity_type == "organization" and entity_name:
                                clients.add(entity_name)
            
            return {
                "consultants": sorted(list(consultants)),
                "clients": sorted(list(clients))
            }
            
        except Exception as e:
            logger.error(f"Error getting entity filter options: {e}")
            return {"consultants": [], "clients": []}

    def generate_ideas_from_problems(self, problem_statements: List[str], 
                                   collection_name: str, themes: List[str] = None,
                                   num_ideas_per_problem: int = 3, 
                                   creativity_level: str = "Balanced",
                                   focus_areas: List[str] = None,
                                   include_implementation: bool = False,
                                   llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """
        Generate creative ideas from problem statements.
        
        Args:
            problem_statements: List of problem statements to generate ideas for
            collection_name: Name of the collection for context
            themes: List of themes for additional context
            num_ideas_per_problem: Number of ideas to generate per problem
            creativity_level: Level of creativity (Conservative, Balanced, Highly Creative)
            focus_areas: Optional focus areas to guide ideation
            include_implementation: Whether to include implementation notes
            llm_provider: LLM provider to use
            
        Returns:
            Dict containing generated ideas and metadata
        """
        try:
            logger.info(f"Generating ideas from {len(problem_statements)} problem statements using {llm_provider}")
            
            # Create the LLM prompt for idea generation
            prompt = self._create_idea_generation_prompt(
                problem_statements, themes, num_ideas_per_problem,
                creativity_level, focus_areas, include_implementation
            )
            
            # Use new unified LLM service
            from ..llm_service import create_llm_service
            from ..exceptions import ModelError
            
            service_manager = create_llm_service("ideation", llm_provider)
            
            try:
                llm = service_manager.get_llm()
            except ModelError as e:
                logger.error(f"LLM initialization failed: {e}")
                return {"error": str(e)}
            
            logger.info("Sending idea generation request to LLM...")
            response = llm.complete(prompt)
            
            # Parse the LLM response
            ideas = self._parse_ideas_response(str(response), problem_statements)
            
            # Convert format to match UI expectations (problem -> problem_statement)
            idea_groups = []
            for idea_item in ideas:
                idea_groups.append({
                    "problem_statement": idea_item["problem"],
                    "ideas": idea_item["ideas"]
                })
            
            logger.info(f"Generated {sum(len(p['ideas']) for p in idea_groups)} total ideas")
            
            return {
                "status": "success",
                "idea_groups": idea_groups,
                "total_problems": len(problem_statements),
                "total_ideas": sum(len(p['ideas']) for p in idea_groups),
                "creativity_level": creativity_level,
                "focus_areas": focus_areas,
                "include_implementation": include_implementation,
                "llm_provider": llm_provider,
                "collection_name": collection_name,
                "themes": themes
            }
            
        except Exception as e:
            logger.error(f"Error in idea generation: {e}")
            return {"error": f"Idea generation failed: {str(e)}"}
    
    def _create_idea_generation_prompt(self, problem_statements: List[str], 
                                     themes: List[str] = None,
                                     num_ideas_per_problem: int = 3,
                                     creativity_level: str = "Balanced",
                                     focus_areas: List[str] = None,
                                     include_implementation: bool = False) -> str:
        """Create a prompt for LLM-based idea generation."""
        
        problems_text = "\n".join([f"{i+1}. {problem}" for i, problem in enumerate(problem_statements)])
        
        creativity_guidance = {
            "Conservative": "Focus on proven, practical solutions with minimal risk",
            "Balanced": "Combine proven approaches with some innovative elements", 
            "Highly Creative": "Embrace bold, innovative, and disruptive thinking"
        }
        
        # Build the implementation field conditionally
        implementation_field = '"implementation": "Brief implementation approach",\n        ' if include_implementation else ''
        implementation_requirement = "5. Include brief implementation notes for each idea" if include_implementation else ""
        
        prompt = f"""You are an expert innovation consultant generating creative solutions to specific problems.

PROBLEM STATEMENTS:
{problems_text}

CONTEXT:
- Themes: {', '.join(themes) if themes else 'Not specified'}
- Focus Areas: {', '.join(focus_areas) if focus_areas else 'General innovation'}
- Creativity Level: {creativity_level} - {creativity_guidance.get(creativity_level, '')}

TASK: Generate {num_ideas_per_problem} innovative, actionable ideas for EACH problem statement.

REQUIREMENTS:
1. Ideas should be specific and actionable
2. Match the {creativity_level.lower()} creativity level
3. Consider the themes and focus areas provided
4. Each idea should be distinct and valuable
{implementation_requirement}

RESPONSE FORMAT:
Return ONLY a JSON structure like this:
{{
  "problem_1": {{
    "problem": "The first problem statement",
    "ideas": [
      {{
        "title": "Idea title",
        "description": "Detailed description of the idea",
        {implementation_field}"impact": "Expected impact and benefits"
      }}
    ]
  }},
  "problem_2": {{
    "problem": "The second problem statement", 
    "ideas": [...]
  }}
}}

GENERATE IDEAS:"""

        return prompt
    
    def _parse_ideas_response(self, response_text: str, problem_statements: List[str]) -> List[Dict[str, Any]]:
        """Parse LLM response to extract ideas by problem."""
        try:
            import json
            import re
            
            # Try to find JSON object in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                ideas_json = json_match.group(0)
                ideas_data = json.loads(ideas_json)
                
                # Convert to expected format
                parsed_ideas = []
                for i, problem in enumerate(problem_statements):
                    problem_key = f"problem_{i+1}"
                    if problem_key in ideas_data:
                        problem_ideas = ideas_data[problem_key]
                        parsed_ideas.append({
                            "problem": problem,
                            "ideas": problem_ideas.get("ideas", [])
                        })
                    else:
                        # Fallback for missing problem
                        parsed_ideas.append({
                            "problem": problem,
                            "ideas": [{"title": "Idea generation incomplete", 
                                     "description": "Please try again with a simpler request",
                                     "impact": "N/A"}]
                        })
                
                return parsed_ideas
                    
            # Fallback: create basic structure if parsing fails
            fallback_ideas = []
            for problem in problem_statements:
                fallback_ideas.append({
                    "problem": problem,
                    "ideas": [{
                        "title": "Parsing Error",
                        "description": "Ideas were generated but could not be parsed properly. Please try again.",
                        "impact": "Please regenerate ideas"
                    }]
                })
            
            return fallback_ideas
            
        except Exception as e:
            logger.warning(f"Error parsing ideas response: {e}")
            # Return error structure
            error_ideas = []
            for problem in problem_statements:
                error_ideas.append({
                    "problem": problem,
                    "ideas": [{
                        "title": "Generation Error",
                        "description": f"Idea generation failed: {str(e)}",
                        "impact": "Please try again"
                    }]
                })
            return error_ideas
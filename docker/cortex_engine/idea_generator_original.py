"""
Idea Generator - Core Logic for Novel Concept Synthesis

This module implements the backend logic for the Idea Generator feature,
following the Double Diamond methodology for structured innovation.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.graph_manager import EnhancedGraphManager
from cortex_engine.utils.logging_utils import get_logger
from cortex_engine.theme_visualizer import ThemeNetworkVisualizer

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
        
    def run_discovery(self, collection_name: str, seed_ideas: str = "", 
                     constraints: str = "", goals: str = "", 
                     research: bool = False, llm_provider: str = "Local (Ollama)",
                     filters: Optional[Dict[str, Any]] = None,
                     selected_themes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute the Discovery phase of the Double Diamond process.
        
        Analyzes the selected knowledge collection to identify:
        - Key themes and patterns
        - Knowledge gaps and opportunities  
        - Potential innovation areas
        - Supporting evidence from documents
        
        Args:
            collection_name: Name of the working collection to analyze
            seed_ideas: Initial ideas or themes to guide analysis
            constraints: Limitations or boundaries to consider
            goals: Innovation objectives and desired outcomes
            research: Whether to supplement with web research
            llm_provider: AI model to use for analysis
            filters: Optional dict with filtering criteria:
                - document_type: Filter by document type (e.g., "Final Report")
                - proposal_outcome: Filter by outcome (e.g., "Won")
                - client_filter: Filter by client/organization name
                - consultant_filter: Filter by consultant name
                - date_range: Filter by date range
                - thematic_tags: Filter by specific tags
            
        Returns:
            Dict containing discovery results with themes, opportunities, and analysis
        """
        try:
            logger.info(f"Starting Discovery phase for collection: {collection_name}")
            
            # Validate collection exists and has content
            if not self._validate_collection(collection_name):
                return {
                    "status": "error",
                    "error": f"Collection '{collection_name}' not found or empty"
                }
            
            # Retrieve collection documents for analysis (with optional filtering)
            collection_docs = self._get_collection_content(collection_name, filters)
            
            if not collection_docs:
                # Try to get just basic info from collection IDs
                doc_ids = self.collection_mgr.get_doc_ids_by_name(collection_name)
                if doc_ids:
                    logger.warning(f"Could not retrieve document content, but found {len(doc_ids)} document IDs. Using fallback approach.")
                    # Create minimal collection info for analysis
                    collection_docs = [{
                        "title": f"Document {i+1}",
                        "content": f"Document from {collection_name} collection",
                        "document_type": "Unknown",
                        "entities": [],
                        "themes": []
                    } for i in range(min(len(doc_ids), 5))]  # Limit to 5 for fallback
                else:
                    return {
                        "status": "error", 
                        "error": "No documents found in selected collection"
                    }
            
            # Build discovery analysis prompt with selected themes
            discovery_prompt = self._build_discovery_prompt(
                collection_docs, seed_ideas, constraints, goals, filters, selected_themes
            )
            
            # Execute analysis using task engine
            analysis_result = self._execute_discovery_analysis(
                discovery_prompt, llm_provider
            )
            
            if not analysis_result or "error" in analysis_result:
                return {
                    "status": "error",
                    "error": analysis_result.get("error", "Analysis failed")
                }
            
            # Parse structured results
            discovery_results = self._parse_discovery_results(analysis_result)
            
            # Add web research if requested
            if research:
                web_insights = self._add_web_research(discovery_results, seed_ideas, goals)
                discovery_results["web_insights"] = web_insights
            
            # Generate theme visualization data
            try:
                theme_analysis = self.theme_visualizer.extract_themes_from_discovery(
                    discovery_results, collection_docs
                )
                discovery_results["theme_network"] = theme_analysis
                logger.info("Theme network analysis completed successfully")
            except Exception as e:
                logger.error(f"Theme visualization failed: {e}")
                discovery_results["theme_network"] = {"error": str(e)}
            
            # Add metadata
            discovery_results.update({
                "status": "success",
                "collection_name": collection_name,
                "document_count": len(collection_docs),
                "analysis_model": llm_provider,
                "web_research_enabled": research,
                "filters_applied": filters if filters else {},
                "filtered_analysis": bool(filters and any(filters.values()))
            })
            
            logger.info(f"Discovery phase completed successfully for {collection_name}")
            return discovery_results
            
        except Exception as e:
            logger.error(f"Discovery phase failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_problem_statements(self, collection_name: str, selected_themes: List[str],
                                  seed_ideas: str = "", constraints: str = "", 
                                  goals: str = "", llm_provider: str = "Local (Ollama)",
                                  filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate focused problem statements from selected themes.
        Simplified approach: Themes → Problem Statements → Ideas
        """
        try:
            logger.info(f"Generating problem statements for {len(selected_themes)} themes")
            
            # Store selected themes for fallback text extraction
            self._last_selected_themes = selected_themes
            
            # Get collection content for context
            collection_docs = self._get_collection_content(collection_name, filters)
            
            if not collection_docs:
                return {
                    "status": "error",
                    "error": "No documents found in selected collection"
                }
            
            # Build problem statement generation prompt
            problem_prompt = self._build_problem_statement_prompt(
                collection_docs, selected_themes, seed_ideas, constraints, goals, filters
            )
            
            # Execute problem statement generation
            problem_result = self._execute_discovery_analysis(problem_prompt, llm_provider)
            
            if not problem_result or "error" in problem_result:
                return {
                    "status": "error", 
                    "error": problem_result.get("error", "Problem statement generation failed")
                }
            
            # Parse results
            problem_statements = self._parse_problem_statements(problem_result)
            
            # Add metadata
            problem_statements.update({
                "status": "success",
                "collection_name": collection_name,
                "selected_themes": selected_themes,
                "document_count": len(collection_docs),
                "analysis_model": llm_provider,
                "filtered_analysis": bool(filters),
                "filters_applied": ", ".join([f"{k}: {v}" for k, v in filters.items()]) if filters else None
            })
            
            return problem_statements
            
        except Exception as e:
            logger.error(f"Problem statement generation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_ideas_from_problems(self, problem_statements: List[str], 
                                   collection_name: str = "", themes: List[str] = None,
                                   num_ideas_per_problem: int = 5, creativity_level: str = "Balanced",
                                   focus_areas: List[str] = None, include_implementation: bool = True,
                                   llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """
        Generate innovative ideas from selected problem statements.
        Final step: Problem Statements → Ideas
        """
        try:
            logger.info(f"Generating ideas for {len(problem_statements)} problem statements")
            
            # Store problem statements for fallback text extraction
            self._last_problems = problem_statements
            
            # Build idea generation prompt
            ideas_prompt = self._build_ideas_prompt(
                problem_statements, themes or [], num_ideas_per_problem, 
                creativity_level, focus_areas or [], include_implementation
            )
            
            # Execute idea generation
            ideas_result = self._execute_discovery_analysis(ideas_prompt, llm_provider)
            
            if not ideas_result or "error" in ideas_result:
                return {
                    "status": "error",
                    "error": ideas_result.get("error", "Idea generation failed")
                }
            
            # Parse results
            parsed_ideas = self._parse_ideas_results(ideas_result)
            
            # Add metadata
            parsed_ideas.update({
                "status": "success",
                "collection_name": collection_name,
                "themes": themes or [],
                "problem_count": len(problem_statements),
                "creativity_level": creativity_level,
                "focus_areas": focus_areas or [],
                "analysis_model": llm_provider
            })
            
            return parsed_ideas
            
        except Exception as e:
            logger.error(f"Idea generation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _build_problem_statement_prompt(self, collection_docs: List[Dict], 
                                       selected_themes: List[str], seed_ideas: str, 
                                       constraints: str, goals: str, 
                                       filters: Optional[Dict[str, Any]] = None) -> str:
        """Build focused problem statement generation prompt."""
        
        # Prepare document summaries (abbreviated for focus)
        doc_summaries = []
        for i, doc in enumerate(collection_docs[:10], 1):  # Limit to 10 for focus
            summary = f"Document {i}: {doc['title']}\n"
            summary += f"Type: {doc['document_type']}\n"
            summary += f"Key insight: {doc['content'][:200]}...\n"
            doc_summaries.append(summary)
        
        docs_text = "\n---\n".join(doc_summaries)
        
        # Build themes context
        themes_text = "\n".join([f"• {theme}" for theme in selected_themes])
        
        prompt = f"""
# Problem Statement Generation

You are an innovation strategist tasked with generating actionable problem statements from identified themes.

## Selected Themes to Focus On:
{themes_text}

## Knowledge Context:
{docs_text}

## User Goals:
**Seed Ideas:** {seed_ideas if seed_ideas else 'None provided'}
**Constraints:** {constraints if constraints else 'None specified'}
**Innovation Goals:** {goals if goals else 'Generate innovative solutions'}

## Task:
For each of the selected themes above, generate 1-2 specific, actionable problem statements that could drive innovation. Each problem statement should:
- Be specific and actionable (not generic)
- Connect to the knowledge base context
- Be solvable through innovation/technology
- Focus on substantial business impact

Provide your response in this JSON format:

```json
{{
    "problem_statements": [
        {{
            "theme": "Theme name",
            "problems": [
                "How might we [specific problem statement related to this theme]?",
                "What if we could [alternative problem approach for this theme]?"
            ]
        }}
    ],
    "summary": "Brief overview of the problem space and innovation potential"
}}
```

Focus on generating problems that are specific, actionable, and connected to the actual content in the knowledge base.

CRITICAL: You MUST respond with ONLY the JSON object shown above. Do not include any explanatory text before or after the JSON. Start your response directly with the opening curly brace {{ and end with the closing curly brace }}.
"""
        return prompt
    
    def _build_ideas_prompt(self, problem_statements: List[str], themes: List[str],
                           num_ideas_per_problem: int, creativity_level: str,
                           focus_areas: List[str], include_implementation: bool) -> str:
        """Build idea generation prompt."""
        
        # Build problems list
        problems_text = "\n".join([f"• {problem}" for problem in problem_statements])
        
        # Build themes context
        themes_text = ", ".join(themes) if themes else "General innovation"
        
        # Build focus areas
        focus_text = ""
        if focus_areas:
            focus_text = f"\n## Focus Areas:\nPrioritize solutions in these areas: {', '.join(focus_areas)}\n"
        
        # Creativity instructions
        creativity_instructions = {
            "Practical": "Focus on realistic, implementable solutions with clear business value.",
            "Balanced": "Balance creative innovation with practical feasibility.",
            "Highly Creative": "Prioritize breakthrough thinking and innovative approaches, even if ambitious."
        }
        
        creativity_guide = creativity_instructions.get(creativity_level, creativity_instructions["Balanced"])
        
        # Implementation requirement
        implementation_req = ""
        if include_implementation:
            implementation_req = ', "implementation": "Brief implementation guidance"'
        
        prompt = f"""
# Innovative Idea Generation

You are an innovation expert tasked with generating creative, actionable solutions for specific problem statements.

## Problem Statements to Solve:
{problems_text}

## Context:
**Themes:** {themes_text}
**Creativity Level:** {creativity_level} - {creativity_guide}
{focus_text}
## Task:
For each problem statement above, generate {num_ideas_per_problem} innovative, actionable ideas. Each idea should be:
- **Specific and actionable** (not generic)
- **Innovative** (creative approach or novel application)
- **Practical** (could realistically be implemented)
- **Valuable** (clear benefit or improvement)

Provide your response in this JSON format:

```json
{{
    "idea_groups": [
        {{
            "problem_statement": "Copy the exact problem statement here",
            "ideas": [
                {{
                    "title": "Concise idea title",
                    "description": "2-3 sentence description of the solution"{implementation_req}
                }}
            ]
        }}
    ],
    "summary": "Brief overview of the generated ideas and their innovation potential"
}}
```

CRITICAL: You MUST respond with ONLY the JSON object shown above. Do not include any explanatory text before or after the JSON. Start your response directly with the opening curly brace {{{{ and end with the closing curly brace }}}}.
"""
        return prompt
    
    def _parse_ideas_results(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse idea generation results."""
        try:
            logger.info(f"Parsing ideas result: {type(analysis_result)}")
            
            # Handle different response formats (similar to problem statement parsing)
            content = None
            if isinstance(analysis_result, dict):
                if "content" in analysis_result:
                    content = analysis_result["content"]
                elif "response" in analysis_result:
                    content = analysis_result["response"]
                elif "message" in analysis_result:
                    content = analysis_result["message"]
            
            if content:
                logger.info(f"Ideas content preview: {content[:300]}...")
                
                import json
                import re
                
                # Try to extract JSON from the content
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    logger.info(f"Found JSON block for ideas")
                    parsed = json.loads(json_str)
                    return parsed
                
                # Look for bare JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logger.info(f"Found bare JSON for ideas")
                    try:
                        parsed = json.loads(json_str)
                        return parsed
                    except:
                        logger.warning("Failed to parse bare JSON for ideas")
                
                # Fallback: try to parse whole content as JSON
                try:
                    parsed = json.loads(content)
                    return parsed
                except:
                    logger.warning("Failed to parse content as JSON for ideas")
                
                # Final fallback: extract ideas from unstructured text
                logger.info("Attempting to extract ideas from unstructured text")
                return self._extract_ideas_from_text(content)
            
            # Last resort fallback
            logger.warning("No content found for ideas, using generic fallback")
            return {
                "idea_groups": [
                    {
                        "problem_statement": "General Innovation",
                        "ideas": [
                            {
                                "title": "Leverage Existing Knowledge",
                                "description": "Use insights from the knowledge collection to drive innovation."
                            },
                            {
                                "title": "Cross-Domain Solutions",
                                "description": "Apply learnings from one domain to solve problems in another."
                            }
                        ]
                    }
                ],
                "summary": "Generated generic ideas due to parsing issues",
                "raw_analysis": analysis_result
            }
            
        except Exception as e:
            logger.error(f"Failed to parse ideas: {e}")
            return {
                "idea_groups": [],
                "summary": f"Ideas parsing failed: {e}",
                "raw_analysis": analysis_result
            }
    
    def _extract_ideas_from_text(self, content: str) -> Dict[str, Any]:
        """Extract ideas from unstructured text when JSON parsing fails."""
        try:
            import re
            
            # Look for numbered/bulleted ideas
            idea_patterns = [
                r"[1-9]\.\s*(.+?)(?=\n|$)",
                r"[-•*]\s*(.+?)(?=\n|$)",
                r"Idea:\s*(.+?)(?=\n|$)",
                r"Solution:\s*(.+?)(?=\n|$)"
            ]
            
            ideas = []
            for pattern in idea_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    clean_match = match.strip()
                    if len(clean_match) > 15 and clean_match not in ideas:  # Skip very short matches
                        ideas.append(clean_match)
            
            # If we found ideas, structure them
            if ideas:
                # Limit to reasonable number
                ideas = ideas[:20]
                
                # Try to group by problem statements (if available)
                problems = getattr(self, '_last_problems', ["Innovation Challenge"])
                
                # Simple distribution across problems
                ideas_per_problem = max(1, len(ideas) // len(problems))
                
                idea_groups = []
                for i, problem in enumerate(problems):
                    start_idx = i * ideas_per_problem
                    end_idx = start_idx + ideas_per_problem if i < len(problems) - 1 else len(ideas)
                    problem_ideas = ideas[start_idx:end_idx]
                    
                    if problem_ideas:
                        # Convert to structured format
                        structured_ideas = []
                        for idea in problem_ideas:
                            # Try to split title and description
                            if ':' in idea:
                                parts = idea.split(':', 1)
                                title = parts[0].strip()
                                description = parts[1].strip()
                            else:
                                title = idea[:50] + "..." if len(idea) > 50 else idea
                                description = idea
                            
                            structured_ideas.append({
                                "title": title,
                                "description": description
                            })
                        
                        idea_groups.append({
                            "problem_statement": problem,
                            "ideas": structured_ideas
                        })
                
                return {
                    "idea_groups": idea_groups,
                    "summary": f"Extracted {len(ideas)} ideas from unstructured response"
                }
            
            # If no ideas found, return generic fallback
            return {
                "idea_groups": [
                    {
                        "problem_statement": "General Innovation",
                        "ideas": [
                            {
                                "title": "Knowledge-Based Innovation",
                                "description": "Leverage insights from the knowledge collection to identify new opportunities."
                            },
                            {
                                "title": "Process Optimization",
                                "description": "Apply systematic approaches to improve existing processes and workflows."
                            }
                        ]
                    }
                ],
                "summary": "Could not parse specific ideas, generated generic solutions",
                "raw_content": content[:1000] + "..." if len(content) > 1000 else content
            }
            
        except Exception as e:
            logger.error(f"Ideas text extraction failed: {e}")
            return {
                "idea_groups": [
                    {
                        "problem_statement": "Extraction Error",
                        "ideas": [
                            {
                                "title": "Review and Refine",
                                "description": "Analyze the approach and improve the idea generation process."
                            }
                        ]
                    }
                ],
                "summary": f"Ideas extraction failed: {e}",
                "raw_content": content if isinstance(content, str) else str(content)
            }
    
    def _parse_problem_statements(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse problem statement generation results."""
        try:
            logger.info(f"Parsing problem statement result: {type(analysis_result)}")
            logger.info(f"Analysis result keys: {analysis_result.keys() if isinstance(analysis_result, dict) else 'Not a dict'}")
            
            # Handle different response formats
            content = None
            if isinstance(analysis_result, dict):
                if "content" in analysis_result:
                    content = analysis_result["content"]
                elif "response" in analysis_result:
                    content = analysis_result["response"]
                elif "message" in analysis_result:
                    content = analysis_result["message"]
            
            if content:
                logger.info(f"Content preview: {content[:500]}...")
                
                # Try to extract JSON from the content
                import json
                import re
                
                # Look for JSON block
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    logger.info(f"Found JSON block: {json_str[:200]}...")
                    parsed = json.loads(json_str)
                    return parsed
                
                # Look for bare JSON (starts with { and ends with })
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logger.info(f"Found bare JSON: {json_str[:200]}...")
                    try:
                        parsed = json.loads(json_str)
                        return parsed
                    except:
                        logger.warning("Failed to parse bare JSON")
                
                # Fallback: try to parse the whole content as JSON
                try:
                    parsed = json.loads(content)
                    return parsed
                except:
                    logger.warning("Failed to parse content as JSON")
                
                # Final fallback: try to extract problem statements from unstructured text
                logger.info("Attempting to extract problem statements from unstructured text")
                return self._extract_problems_from_text(content)
            
            # Last resort fallback: create structured response from raw content
            logger.warning("No content found, using generic fallback")
            content_fallback = str(analysis_result.get("content") or analysis_result.get("response") or "No content available")
            return {
                "problem_statements": [
                    {
                        "theme": "Analysis Results",
                        "problems": [
                            "Generated analysis content available",
                            "See raw analysis for details"
                        ]
                    }
                ],
                "summary": content_fallback[:500] + "..." if len(content_fallback) > 500 else content_fallback,
                "raw_analysis": analysis_result
            }
            
        except Exception as e:
            logger.error(f"Failed to parse problem statements: {e}")
            return {
                "problem_statements": [],
                "summary": f"Parsing failed: {e}",
                "raw_analysis": analysis_result
            }
    
    def _extract_problems_from_text(self, content: str) -> Dict[str, Any]:
        """Extract problem statements from unstructured text when JSON parsing fails."""
        try:
            import re
            
            # Look for "How might we" or similar problem statement patterns
            problem_patterns = [
                r"how might we ([^?]+\?)",
                r"what if we could ([^?]+\?)",
                r"how could we ([^?]+\?)",
                r"what ways might we ([^?]+\?)",
                r"how can we ([^?]+\?)"
            ]
            
            problems = []
            for pattern in problem_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    problem = f"How might we {match}" if not match.lower().startswith('how') else match
                    if problem not in problems:
                        problems.append(problem.strip())
            
            # Look for numbered/bulleted problem statements
            bullet_patterns = [
                r"[1-9]\.\s*(.+?)(?=\n|$)",
                r"[-•*]\s*(.+?)(?=\n|$)",
                r"Problem:\s*(.+?)(?=\n|$)"
            ]
            
            for pattern in bullet_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    clean_match = match.strip()
                    if len(clean_match) > 10 and clean_match not in problems:  # Skip very short matches
                        # Convert to question format if not already
                        if not clean_match.endswith('?'):
                            clean_match = f"How might we {clean_match.lower()}?"
                        problems.append(clean_match)
            
            # If we found problems, structure them
            if problems:
                # Limit to reasonable number
                problems = problems[:10]
                
                # Try to group by themes mentioned in the content
                theme_groups = {}
                selected_themes = getattr(self, '_last_selected_themes', ['General Innovation'])
                
                # Simple distribution across themes
                problems_per_theme = max(1, len(problems) // len(selected_themes))
                
                for i, theme in enumerate(selected_themes):
                    start_idx = i * problems_per_theme
                    end_idx = start_idx + problems_per_theme if i < len(selected_themes) - 1 else len(problems)
                    theme_problems = problems[start_idx:end_idx]
                    
                    if theme_problems:
                        theme_groups[theme] = theme_problems
                
                # Convert to expected format
                problem_statements = []
                for theme, theme_problems in theme_groups.items():
                    problem_statements.append({
                        "theme": theme,
                        "problems": theme_problems
                    })
                
                return {
                    "problem_statements": problem_statements,
                    "summary": f"Extracted {len(problems)} problem statements from unstructured response"
                }
            
            # If no problems found, return generic fallback
            return {
                "problem_statements": [
                    {
                        "theme": "General Innovation",
                        "problems": [
                            "How might we leverage the insights from this knowledge collection?",
                            "What innovative solutions could we develop based on these themes?"
                        ]
                    }
                ],
                "summary": "Could not parse specific problems, generated generic problem statements",
                "raw_content": content[:1000] + "..." if len(content) > 1000 else content
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {
                "problem_statements": [
                    {
                        "theme": "Analysis Error",
                        "problems": [
                            "How might we improve the problem statement generation process?",
                            "What alternative approaches could we use for extracting insights?"
                        ]
                    }
                ],
                "summary": f"Text extraction failed: {e}",
                "raw_content": content if isinstance(content, str) else str(content)
            }
    
    def _validate_collection(self, collection_name: str) -> bool:
        """Validate that the collection exists and has content."""
        try:
            collections = self.collection_mgr.get_collection_names()
            if collection_name not in collections:
                return False
                
            doc_ids = self.collection_mgr.get_doc_ids_by_name(collection_name)
            return len(doc_ids) > 0
            
        except Exception as e:
            logger.error(f"Collection validation failed: {e}")
            return False
    
    def _get_collection_content(self, collection_name: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Retrieve documents from the specified collection with optional filtering."""
        try:
            if self.vector_index is None:
                logger.error("Vector index not provided to IdeaGenerator")
                return []
            
            # Get document IDs from collection
            doc_ids = self.collection_mgr.get_doc_ids_by_name(collection_name)
            if not doc_ids:
                logger.warning(f"No document IDs found for collection: {collection_name}")
                return []
            
            logger.info(f"Found {len(doc_ids)} documents in collection '{collection_name}'")
            
            # Get the vector store to query metadata
            try:
                vector_store = self.vector_index.vector_store
                if vector_store is None:
                    logger.error("Vector store is None")
                    return []
            except Exception as e:
                logger.error(f"Failed to access vector store: {e}")
                return []
            
            # Apply entity-based filtering first if specified
            if filters and (filters.get("consultant_filter") or filters.get("client_filter")):
                logger.info("Applying entity-based filtering...")
                doc_ids = self.apply_entity_filters(doc_ids, filters)
                logger.info(f"After entity filtering: {len(doc_ids)} documents remain")
            
            # Query for documents with these IDs (use more documents for better theme analysis)
            limited_doc_ids = doc_ids[:500]  # Increase from 20 to 500 for better theme diversity
            logger.info(f"Querying vector store for {len(limited_doc_ids)} documents from collection {collection_name}")
            
            # Build filtering query for vector store
            where_conditions = []
            
            # Always filter by collection document IDs
            where_conditions.append({"doc_id": {"$in": limited_doc_ids}})
            
            # Apply additional filters if provided
            if filters:
                logger.info(f"Applying filters: {filters}")
                
                # Document type filter
                if filters.get("document_type") and filters["document_type"] != "Any":
                    where_conditions.append({"document_type": filters["document_type"]})
                
                # Proposal outcome filter
                if filters.get("proposal_outcome") and filters["proposal_outcome"] != "Any":
                    where_conditions.append({"proposal_outcome": filters["proposal_outcome"]})
                
                # Thematic tags filter (contains any of the specified tags)
                if filters.get("thematic_tags"):
                    tag_conditions = []
                    for tag in filters["thematic_tags"]:
                        tag_conditions.append({"thematic_tags": {"$regex": f".*{tag}.*"}})
                    if tag_conditions:
                        where_conditions.append({"$or": tag_conditions})
            
            # Combine all conditions with AND logic
            if len(where_conditions) > 1:
                where_clause = {"$and": where_conditions}
            elif len(where_conditions) == 1:
                where_clause = where_conditions[0]
            else:
                where_clause = {}
            
            logger.info(f"Vector store query with where clause: {where_clause}")
            
            # FIXED: Use metadata doc_id query to filter by collection documents
            logger.info(f"Querying vector store for specific doc_ids: {limited_doc_ids[:5]}... (showing first 5)")
            
            try:
                # Query by metadata doc_id field (not ChromaDB primary key)
                results = vector_store._collection.get(
                    where={"doc_id": {"$in": limited_doc_ids}},
                    include=["metadatas", "documents"]
                )
                retrieved_count = len(results.get('metadatas', []))
                logger.info(f"Metadata doc_id query retrieved {retrieved_count} documents (expected: {len(limited_doc_ids)})")
                
                # Handle duplicates: ChromaDB may have duplicate doc_ids, so deduplicate
                if retrieved_count > len(limited_doc_ids):
                    logger.warning(f"WARNING: Got {retrieved_count} docs but only requested {len(limited_doc_ids)} - deduplicating by doc_id")
                
            except Exception as e:
                logger.error(f"Metadata doc_id query failed: {e}")
                return []
            
            collection_content = []
            if results and 'metadatas' in results and 'documents' in results:
                logger.info(f"Retrieved {len(results['metadatas'])} documents from vector store")
                
                # Deduplicate by doc_id - keep only the first occurrence of each unique doc_id
                seen_doc_ids = set()
                unique_results = {"metadatas": [], "documents": []}
                
                for metadata, document in zip(results['metadatas'], results['documents']):
                    doc_id = metadata.get("doc_id")
                    if doc_id and doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        unique_results["metadatas"].append(metadata)
                        unique_results["documents"].append(document)
                
                deduplicated_count = len(unique_results["metadatas"])
                if deduplicated_count < len(results['metadatas']):
                    logger.info(f"Deduplicated: {len(results['metadatas'])} -> {deduplicated_count} unique documents")
                
                # Process deduplicated results
                for metadata, document in zip(unique_results['metadatas'], unique_results['documents']):
                    content_summary = {
                        "title": metadata.get("file_name", "Untitled"),
                        "content": document[:1000] if document else "",  # First 1000 chars
                        "document_type": metadata.get("document_type", "Unknown"),
                        "entities": [],  # Could extract from metadata if available
                        "themes": metadata.get("thematic_tags", "").split(", ") if metadata.get("thematic_tags") else [],
                        "metadata": metadata  # INCLUDE FULL METADATA for theme extraction
                    }
                    collection_content.append(content_summary)
            else:
                logger.warning("No results returned from vector store query")
            
            logger.info(f"Processed {len(collection_content)} documents for analysis")
            return collection_content
            
        except Exception as e:
            logger.error(f"Failed to retrieve collection content: {e}")
            return []
    
    def _build_discovery_prompt(self, collection_docs: List[Dict], 
                               seed_ideas: str, constraints: str, goals: str, 
                               filters: Optional[Dict[str, Any]] = None,
                               selected_themes: Optional[List[str]] = None) -> str:
        """Build the discovery analysis prompt for the LLM."""
        
        # Prepare document summaries
        doc_summaries = []
        for i, doc in enumerate(collection_docs, 1):
            summary = f"Document {i}: {doc['title']}\n"
            summary += f"Type: {doc['document_type']}\n"
            if doc['entities']:
                summary += f"Key Entities: {', '.join(doc['entities'][:5])}\n"
            summary += f"Content Preview: {doc['content'][:300]}...\n"
            doc_summaries.append(summary)
        
        docs_text = "\n---\n".join(doc_summaries)
        
        # Build filter context
        filter_context = ""
        if filters:
            filter_context = "\n## Applied Filters:\n"
            filter_context += "This analysis is focused on a filtered subset of the collection:\n"
            
            if filters.get("document_type"):
                filter_context += f"- **Document Type:** {filters['document_type']} (focusing on specific document categories)\n"
            if filters.get("proposal_outcome"):
                filter_context += f"- **Proposal Outcome:** {filters['proposal_outcome']} (analyzing {filters['proposal_outcome'].lower()} projects)\n"
            if filters.get("thematic_tags"):
                tags = ', '.join(filters['thematic_tags'])
                filter_context += f"- **Thematic Focus:** {tags} (examining specific themes/technologies)\n"
            if filters.get("consultant"):
                filter_context += f"- **Consultant Focus:** {filters['consultant']} (analyzing this consultant's expertise area)\n"
            if filters.get("client"):
                filter_context += f"- **Client Focus:** {filters['client']} (examining work for this specific client)\n"
            
            filter_context += "\n**Analysis Direction:** Please tailor your discovery analysis to leverage these specific focus areas. Consider how the filtering creates opportunities for deeper, more targeted insights.\n"
        
        # Build selected themes context
        themes_context = ""
        if selected_themes:
            themes_context = f"\n## Selected Focus Themes:\n"
            themes_context += f"The user has identified these {len(selected_themes)} key themes from the collection to focus the ideation on:\n"
            for i, theme in enumerate(selected_themes, 1):
                themes_context += f"  {i}. **{theme}**\n"
            themes_context += f"\n**Analysis Direction:** Please center your discovery analysis around these selected themes. Look for patterns, opportunities, and insights specifically related to these {len(selected_themes)} focus areas. Avoid generic themes and focus on actionable innovation potential within these domains.\n"

        prompt = f"""
# Discovery Phase Analysis

You are an innovation analyst conducting the DISCOVERY phase of the Double Diamond design process. 
Analyze the provided knowledge collection to identify themes, patterns, opportunities, and insights 
that could spark innovative ideas.

**Important:** Focus on meaningful, multi-word themes that represent substantial business concepts rather than single words or organizational terms.
{filter_context}{themes_context}
## Collection Content:
{docs_text}

## User Context:
**Seed Ideas:** {seed_ideas if seed_ideas else 'None provided'}
**Constraints:** {constraints if constraints else 'None specified'}  
**Innovation Goals:** {goals if goals else 'General innovation exploration'}

## Analysis Requirements:

Provide a structured analysis in the following JSON format:

```json
{{
    "themes": [
        "List 3-5 meaningful business themes (prefer multi-word concepts like 'digital transformation', 'customer experience optimization')",
        "Avoid generic single words (meeting, presentation, collaboration) and organizational names",
        "Focus on substantial patterns that offer innovation potential"
    ],
    "opportunities": [
        "List 3-5 specific opportunity areas for innovation", 
        "Focus on gaps, unmet needs, or potential improvements",
        "Each opportunity should be actionable"
    ],
    "key_insights": [
        "List 3-5 notable insights or unexpected connections",
        "Highlight surprising patterns or relationships"
    ],
    "knowledge_gaps": [
        "List 2-3 areas where additional knowledge could be valuable",
        "Identify missing perspectives or information"
    ],
    "innovation_potential": [
        "List 3-4 high-potential areas for innovation based on the analysis",
        "Consider emerging trends and cross-domain opportunities"
    ],
    "analysis_summary": "A 2-3 paragraph summary of the overall discovery findings and their implications for innovation"
}}
```

Focus on:
- Identifying latent connections between different documents
- Spotting emerging patterns and trends
- Finding gaps where innovation could add value
- Connecting user goals with knowledge insights
- Maintaining an optimistic, opportunity-focused perspective

Ensure all suggestions are grounded in the actual content of the provided documents.
"""
        
        return prompt
    
    def _execute_discovery_analysis(self, prompt: str, llm_provider: str) -> Dict[str, Any]:
        """Execute the discovery analysis using a simplified LLM approach."""
        try:
            # For Sprint 1, implement a simplified version that will be enhanced in later sprints
            # This provides immediate functionality while we build out the full task engine integration
            
            if "Local" in llm_provider:
                # Use local Ollama for analysis
                result = self._execute_ollama_analysis(prompt)
            else:
                # Placeholder for cloud providers (to be implemented in Sprint 1 completion)
                result = {
                    "response": f"""
# Discovery Analysis Results (Simplified)

Based on the provided knowledge collection, here are the key findings:

## Key Themes
1. **Content Analysis**: The collection contains diverse knowledge requiring synthesis
2. **Innovation Potential**: Multiple areas show potential for creative development  
3. **Knowledge Integration**: Cross-domain connections are possible

## Opportunities
1. **Systematic Analysis**: Apply structured methodologies to extract insights
2. **Pattern Recognition**: Identify recurring themes and relationships
3. **Gap Identification**: Find areas where innovation could add value

## Innovation Potential
1. **Knowledge Synthesis**: Combine insights from different sources
2. **Creative Problem Solving**: Apply existing knowledge to new challenges
3. **Structured Ideation**: Use systematic approaches for idea generation

## Summary
The selected collection provides a solid foundation for innovation. The Discovery phase has identified several promising areas for further exploration. Proceeding to the Define phase will help formulate specific problem statements and innovation targets.

*Note: This is a simplified analysis for Sprint 1. Enhanced AI-powered analysis will be available in future updates.*
                    """.strip()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Discovery analysis execution failed: {e}")
            return {"error": str(e)}
    
    def _execute_ollama_analysis(self, prompt: str) -> Dict[str, Any]:
        """Execute analysis using local Ollama model."""
        try:
            import ollama
            
            logger.info("Executing Ollama analysis...")
            
            # Use the default model for analysis
            response = ollama.chat(
                model='mistral:7b-instruct-v0.3-q4_K_M',  # Default Cortex model
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={'temperature': 0.7}
            )
            
            ollama_response = response['message']['content']
            logger.info(f"Ollama response preview: {ollama_response[:300]}...")
            
            return {"response": ollama_response}
            
        except ImportError:
            logger.warning("Ollama not available, using fallback analysis")
            return self._generate_fallback_analysis()
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            return self._generate_fallback_analysis()
    
    def _generate_fallback_analysis(self) -> Dict[str, Any]:
        """Generate a fallback analysis when LLM services are unavailable."""
        return {
            "response": """
# Discovery Analysis (Offline Mode)

The Discovery phase has been completed using offline analysis methods.

## Key Findings

**Themes Identified:**
1. Document collection contains valuable knowledge assets
2. Multiple content types provide diverse perspectives
3. Systematic analysis reveals patterns and connections

**Opportunities for Innovation:**
1. Knowledge synthesis across document boundaries
2. Pattern-based insight generation  
3. Structured ideation methodologies

**Next Steps:**
1. Proceed to Define phase for problem statement formulation
2. Apply specific innovation frameworks
3. Generate targeted solution concepts

*Note: Enhanced AI analysis will be available when LLM services are configured.*
            """.strip()
        }
    
    def _parse_discovery_results(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure the discovery analysis results."""
        try:
            # Extract the response text
            response_text = analysis_result.get("response", "")
            
            # Try to extract JSON from the response
            start_marker = "```json"
            end_marker = "```"
            
            if start_marker in response_text and end_marker in response_text:
                json_start = response_text.find(start_marker) + len(start_marker)
                json_end = response_text.find(end_marker, json_start)
                json_text = response_text[json_start:json_end].strip()
                
                try:
                    parsed_results = json.loads(json_text)
                    return parsed_results
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response, using fallback parsing")
            
            # Fallback: Create structured response from text
            return {
                "themes": ["Analysis completed - see full analysis below"],
                "opportunities": ["Review detailed analysis for specific opportunities"],
                "analysis": response_text,
                "parsing_note": "Full text analysis provided (JSON parsing failed)"
            }
            
        except Exception as e:
            logger.error(f"Failed to parse discovery results: {e}")
            return {
                "themes": ["Analysis parsing failed"],
                "opportunities": ["Please review raw analysis"],
                "analysis": str(analysis_result),
                "error": str(e)
            }
    
    def _add_web_research(self, discovery_results: Dict, seed_ideas: str, goals: str) -> List[str]:
        """Add supplementary web research insights (placeholder for now)."""
        # TODO: Implement web research integration in Sprint 1 completion
        return [
            "Web research integration coming soon",
            "Will supplement collection analysis with external insights",
            "Focus areas: industry trends, competitive landscape, emerging technologies"
        ]
    
    def run_define(self, discovery_results: Dict[str, Any], selected_opportunities: List[str] = None) -> Dict[str, Any]:
        """
        Execute the Define phase of the Double Diamond process.
        
        Takes Discovery results and formulates specific "How Might We..." problem statements.
        
        Args:
            discovery_results: Results from the Discovery phase
            selected_opportunities: User-selected opportunities to focus on (optional)
            
        Returns:
            Dict containing problem statements and refined focus areas
        """
        try:
            logger.info("Starting Define phase - generating 'How Might We' problem statements")
            
            # Extract relevant information from discovery results
            themes = discovery_results.get("themes", [])
            opportunities = discovery_results.get("opportunities", [])
            key_insights = discovery_results.get("key_insights", [])
            collection_name = discovery_results.get("collection_name", "Unknown")
            
            # Use selected opportunities or all if none selected
            focus_opportunities = selected_opportunities if selected_opportunities else opportunities
            
            if not focus_opportunities:
                return {
                    "status": "error",
                    "error": "No opportunities available for problem statement generation"
                }
            
            # Build define phase prompt
            define_prompt = self._build_define_prompt(themes, focus_opportunities, key_insights, collection_name)
            
            # Execute analysis
            analysis_result = self._execute_define_analysis(define_prompt)
            
            if not analysis_result or "error" in analysis_result:
                return {
                    "status": "error",
                    "error": analysis_result.get("error", "Define analysis failed")
                }
            
            # Parse results
            define_results = self._parse_define_results(analysis_result)
            
            # Add metadata
            define_results.update({
                "status": "success",
                "source_opportunities": focus_opportunities,
                "discovery_context": {
                    "collection_name": collection_name,
                    "themes_count": len(themes),
                    "insights_count": len(key_insights)
                }
            })
            
            logger.info(f"Define phase completed - generated {len(define_results.get('problem_statements', []))} problem statements")
            return define_results
            
        except Exception as e:
            error_msg = f"Define phase failed: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    def run_develop(self, problem_statements: List[Dict], selected_statements: List[str] = None, 
                   llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """
        Execute the Develop phase of the Double Diamond process.
        
        Generates diverse solutions using multi-agent ideation approach.
        
        Args:
            problem_statements: List of problem statements from Define phase
            selected_statements: User-selected statements to focus on (optional)
            llm_provider: AI model to use for ideation
            
        Returns:
            Dict containing generated ideas organized by agent type
        """
        try:
            logger.info("Starting Develop phase - multi-agent ideation")
            
            # Filter to selected statements or use all
            if selected_statements:
                focus_statements = [stmt for stmt in problem_statements if stmt.get('statement') in selected_statements]
            else:
                focus_statements = problem_statements
            
            if not focus_statements:
                return {
                    "status": "error",
                    "error": "No problem statements available for ideation"
                }
            
            develop_results = {
                "status": "success",
                "problem_statements_used": focus_statements,
                "ideation_results": {}
            }
            
            # Multi-agent ideation for each problem statement
            for i, statement in enumerate(focus_statements):
                statement_text = statement.get('statement', '')
                context = statement.get('context', '')
                
                logger.info(f"Generating ideas for problem statement {i+1}/{len(focus_statements)}")
                
                # Agent 1: Solution Brainstormer
                brainstormed_ideas = self._agent_solution_brainstormer(statement_text, context, llm_provider)
                
                # Agent 2: Analogy Finder  
                analogy_ideas = self._agent_analogy_finder(statement_text, context, llm_provider)
                
                # Agent 3: Feasibility Analyzer (analyzes combined ideas)
                all_ideas = brainstormed_ideas + analogy_ideas
                feasibility_analysis = self._agent_feasibility_analyzer(all_ideas, statement_text, llm_provider)
                
                develop_results["ideation_results"][statement_text] = {
                    "brainstormed_solutions": brainstormed_ideas,
                    "analogy_solutions": analogy_ideas,
                    "feasibility_analysis": feasibility_analysis,
                    "total_ideas": len(all_ideas)
                }
            
            logger.info(f"Develop phase completed for {len(focus_statements)} problem statements")
            return develop_results
            
        except Exception as e:
            error_msg = f"Develop phase failed: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    def _build_define_prompt(self, themes: List[str], opportunities: List[str], 
                           insights: List[str], collection_name: str) -> str:
        """Build the Define phase prompt for generating 'How Might We' statements."""
        
        themes_text = "\n".join(f"- {theme}" for theme in themes)
        opportunities_text = "\n".join(f"- {opp}" for opp in opportunities)
        insights_text = "\n".join(f"- {insight}" for insight in insights)
        
        prompt = f"""
# Define Phase: Problem Statement Generation

You are an innovation strategist conducting the DEFINE phase of the Double Diamond design process.
Transform broad opportunities into specific, actionable "How Might We..." problem statements.

## Discovery Context:
**Collection:** {collection_name}

**Key Themes:**
{themes_text}

**Opportunity Areas:**
{opportunities_text}

**Key Insights:**
{insights_text}

## Task:
Generate 5-7 specific "How Might We..." problem statements that:
1. Build on the identified opportunities
2. Are specific and actionable
3. Invite creative solutions
4. Connect to the knowledge base themes

## Output Format:
Provide your response as a JSON object:

```json
{{
    "problem_statements": [
        {{
            "statement": "How might we [specific problem statement]?",
            "context": "Brief explanation of why this is important",
            "opportunity_connection": "Which opportunity area this addresses",
            "potential_impact": "Expected impact if solved"
        }}
    ],
    "focus_areas": [
        "List 3-4 refined focus areas for ideation"
    ],
    "constraints_to_consider": [
        "List 2-3 key constraints or considerations"
    ]
}}
```

Focus on creating problem statements that are:
- Specific enough to guide solution generation
- Broad enough to allow creative exploration
- Grounded in the actual knowledge from the collection
- Actionable and solution-oriented
"""
        
        return prompt
    
    def _execute_define_analysis(self, prompt: str) -> Dict[str, Any]:
        """Execute the Define phase analysis using LLM."""
        try:
            # Use the same execution pattern as Discovery
            import ollama
            
            response = ollama.chat(
                model='mistral:7b-instruct-v0.3-q4_K_M',
                messages=[{
                    'role': 'user', 
                    'content': prompt
                }],
                options={'temperature': 0.8}  # More creative for problem framing
            )
            
            return {"response": response['message']['content']}
            
        except ImportError:
            logger.warning("Ollama not available, using fallback analysis")
            return self._generate_fallback_define()
        except Exception as e:
            logger.error(f"Define analysis failed: {e}")
            return self._generate_fallback_define()
    
    def _generate_fallback_define(self) -> Dict[str, Any]:
        """Generate fallback Define phase results when LLM is unavailable."""
        return {
            "response": """
```json
{
    "problem_statements": [
        {
            "statement": "How might we better organize and connect the knowledge within our collection?",
            "context": "The collection contains valuable information that could be more effectively structured",
            "opportunity_connection": "Knowledge organization and accessibility",
            "potential_impact": "Improved knowledge discovery and utilization"
        },
        {
            "statement": "How might we identify and fill the gaps in our current knowledge base?", 
            "context": "Analysis reveals areas where additional information would be valuable",
            "opportunity_connection": "Knowledge gap identification",
            "potential_impact": "More comprehensive understanding and decision-making"
        },
        {
            "statement": "How might we leverage cross-domain insights for innovative solutions?",
            "context": "The collection spans multiple domains with potential for creative connections",
            "opportunity_connection": "Cross-domain innovation",
            "potential_impact": "Novel solutions through interdisciplinary thinking"
        }
    ],
    "focus_areas": [
        "Knowledge organization and structure",
        "Cross-domain connection identification", 
        "Gap analysis and strategic information needs"
    ],
    "constraints_to_consider": [
        "Available resources and time",
        "Technical feasibility",
        "Organizational capacity for change"
    ]
}
```
            """.strip()
        }
    
    def _parse_define_results(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure the Define phase results."""
        try:
            response_text = analysis_result.get("response", "")
            
            # Extract JSON from response
            start_marker = "```json"
            end_marker = "```"
            
            if start_marker in response_text and end_marker in response_text:
                json_start = response_text.find(start_marker) + len(start_marker)
                json_end = response_text.find(end_marker, json_start)
                json_text = response_text[json_start:json_end].strip()
                
                try:
                    parsed_results = json.loads(json_text)
                    return parsed_results
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Define JSON response")
            
            # Fallback parsing
            return {
                "problem_statements": [
                    {
                        "statement": "Review the full analysis for problem statements",
                        "context": "JSON parsing failed", 
                        "opportunity_connection": "See detailed analysis",
                        "potential_impact": "Analysis available in raw format"
                    }
                ],
                "focus_areas": ["Review detailed analysis"],
                "full_analysis": response_text
            }
            
        except Exception as e:
            logger.error(f"Failed to parse Define results: {e}")
            return {
                "problem_statements": [],
                "focus_areas": [],
                "error": str(e),
                "raw_response": str(analysis_result)
            }
    
    def _agent_solution_brainstormer(self, problem_statement: str, context: str, llm_provider: str) -> List[str]:
        """Agent 1: Generate diverse solution ideas through brainstorming."""
        prompt = f"""
You are a creative solution brainstormer. Generate 5-7 diverse, innovative solutions for this problem:

**Problem:** {problem_statement}
**Context:** {context}

Requirements:
- Think outside the box
- Include both practical and ambitious ideas
- Consider different approaches and perspectives
- Be specific and actionable

Return only a numbered list of solution ideas, one per line.
"""
        
        try:
            import ollama
            response = ollama.chat(
                model='mistral:7b-instruct-v0.3-q4_K_M',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.9}  # High creativity
            )
            
            # Extract numbered list
            content = response['message']['content']
            ideas = []
            for line in content.split('\n'):
                line = line.strip()
                if line and any(line.startswith(str(i)) for i in range(1, 10)):
                    # Remove number prefix and clean
                    idea = line.split('.', 1)[-1].strip()
                    if idea:
                        ideas.append(idea)
            
            return ideas[:7]  # Limit to 7 ideas
            
        except Exception as e:
            logger.error(f"Solution brainstormer failed: {e}")
            return [
                "Implement systematic knowledge organization framework",
                "Create cross-reference mapping system", 
                "Develop automated insight generation tools",
                "Establish collaborative review processes",
                "Design user-friendly knowledge interfaces"
            ]
    
    def _agent_analogy_finder(self, problem_statement: str, context: str, llm_provider: str) -> List[str]:
        """Agent 2: Find cross-domain analogies and inspiration."""
        prompt = f"""
You are an analogy expert. Find inspiration from other domains to solve this problem:

**Problem:** {problem_statement}
**Context:** {context}

Think about how similar challenges are solved in:
- Nature and biology
- Other industries 
- Historical examples
- Technology domains
- Social systems

Generate 4-5 solution ideas inspired by analogies from different domains.
Format each as: "Like [analogy source]: [solution description]"
"""
        
        try:
            import ollama
            response = ollama.chat(
                model='mistral:7b-instruct-v0.3-q4_K_M',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.8}
            )
            
            content = response['message']['content']
            ideas = []
            for line in content.split('\n'):
                line = line.strip()
                if line and ('like' in line.lower() or ':' in line):
                    ideas.append(line)
            
            return ideas[:5]  # Limit to 5 analogy ideas
            
        except Exception as e:
            logger.error(f"Analogy finder failed: {e}")
            return [
                "Like library cataloging: Create hierarchical knowledge taxonomy",
                "Like neural networks: Build interconnected concept maps", 
                "Like ecosystem food webs: Map knowledge flow and dependencies",
                "Like jazz improvisation: Enable flexible knowledge recombination",
                "Like archaeological layers: Preserve knowledge evolution history"
            ]
    
    def _agent_feasibility_analyzer(self, ideas: List[str], problem_statement: str, llm_provider: str) -> Dict[str, Any]:
        """Agent 3: Analyze feasibility and provide recommendations."""
        ideas_text = "\n".join(f"- {idea}" for idea in ideas)
        
        prompt = f"""
You are a feasibility analyst. Evaluate these solution ideas for the problem:

**Problem:** {problem_statement}

**Solution Ideas:**
{ideas_text}

Analyze each idea considering:
- Implementation complexity (Low/Medium/High)
- Resource requirements
- Time to value
- Innovation potential
- Risk level

Provide a structured analysis and recommend the top 3 most promising ideas.

Format as JSON:
{{"high_potential": ["idea1", "idea2", "idea3"], "quick_wins": ["easy idea1", "easy idea2"], "analysis_summary": "brief overall assessment"}}
"""
        
        try:
            import ollama
            response = ollama.chat(
                model='mistral:7b-instruct-v0.3-q4_K_M',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.6}  # Balanced creativity/analysis
            )
            
            content = response['message']['content']
            
            # Try to extract JSON
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                try:
                    return json.loads(content[start:end])
                except:
                    pass
            
            return {
                "high_potential": ideas[:3] if len(ideas) >= 3 else ideas,
                "quick_wins": ideas[-2:] if len(ideas) >= 2 else ideas[:1],
                "analysis_summary": "Feasibility analysis completed - see individual idea assessments"
            }
            
        except Exception as e:
            logger.error(f"Feasibility analyzer failed: {e}")
            return {
                "high_potential": ideas[:3] if ideas else [],
                "quick_wins": ideas[-2:] if len(ideas) >= 2 else [],
                "analysis_summary": "Fallback analysis - all ideas have moderate feasibility"
            }
    
    def run_deliver(self, develop_results: Dict[str, Any], selected_ideas: List[str] = None, 
                   llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """
        Execute the Deliver phase of the Double Diamond process.
        
        Creates detailed, structured reports for selected ideas from the Develop phase.
        
        Args:
            develop_results: Results from the Develop phase
            selected_ideas: User-selected ideas to create reports for (optional)
            llm_provider: AI model to use for report generation
            
        Returns:
            Dict containing detailed reports for each selected idea
        """
        try:
            logger.info("Starting Deliver phase - creating structured idea reports")
            
            # Extract ideation results
            ideation_results = develop_results.get("ideation_results", {})
            if not ideation_results:
                return {
                    "status": "error",
                    "error": "No ideation results available for report generation"
                }
            
            # Collect all ideas if none specifically selected
            all_ideas = []
            idea_contexts = {}  # Map idea to its problem statement context
            
            for problem_statement, results in ideation_results.items():
                brainstormed = results.get("brainstormed_solutions", [])
                analogies = results.get("analogy_solutions", [])
                high_potential = results.get("feasibility_analysis", {}).get("high_potential", [])
                
                # Combine all idea types, prioritizing high potential
                problem_ideas = high_potential + brainstormed + analogies
                
                for idea in problem_ideas:
                    if idea not in all_ideas:
                        all_ideas.append(idea)
                        idea_contexts[idea] = {
                            "problem_statement": problem_statement,
                            "is_high_potential": idea in high_potential
                        }
            
            # Filter to selected ideas or use top ideas
            if selected_ideas:
                focus_ideas = [idea for idea in selected_ideas if idea in all_ideas]
            else:
                # Default to top 5 most promising ideas
                focus_ideas = all_ideas[:5]
            
            if not focus_ideas:
                return {
                    "status": "error",
                    "error": "No valid ideas selected for report generation"
                }
            
            deliver_results = {
                "status": "success",
                "source_develop_results": develop_results,
                "selected_ideas_count": len(focus_ideas),
                "detailed_reports": {}
            }
            
            # Generate detailed report for each selected idea
            for i, idea in enumerate(focus_ideas):
                logger.info(f"Generating detailed report for idea {i+1}/{len(focus_ideas)}")
                
                context = idea_contexts.get(idea, {})
                problem_statement = context.get("problem_statement", "Unknown")
                is_high_potential = context.get("is_high_potential", False)
                
                # Generate structured report
                report = self._generate_idea_report(
                    idea, problem_statement, is_high_potential, llm_provider
                )
                
                deliver_results["detailed_reports"][idea] = report
            
            logger.info(f"Deliver phase completed - generated {len(focus_ideas)} detailed reports")
            return deliver_results
            
        except Exception as e:
            error_msg = f"Deliver phase failed: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    def _generate_idea_report(self, idea: str, problem_statement: str, 
                            is_high_potential: bool, llm_provider: str) -> Dict[str, Any]:
        """Generate a detailed, structured report for a specific idea."""
        
        prompt = f"""
You are an innovation strategist creating a detailed, actionable report for a promising idea. 
Provide a comprehensive analysis in structured JSON format.

**Idea:** {idea}
**Problem Context:** {problem_statement}
**High Potential:** {"Yes" if is_high_potential else "No"}

Create a detailed report with the following structure:

```json
{{
    "idea_title": "Concise, compelling title for this idea (3-7 words)",
    "core_concept": "Clear, detailed explanation of the core concept and how it works (2-3 paragraphs)",
    "supporting_evidence": [
        "Evidence point 1 - why this idea has merit",
        "Evidence point 2 - supporting research or precedents",
        "Evidence point 3 - market or user validation"
    ],
    "potential_applications": [
        "Application area 1 with specific use cases",
        "Application area 2 with specific use cases", 
        "Application area 3 with specific use cases"
    ],
    "implementation_roadmap": {{
        "phase_1": "Immediate next steps (0-3 months)",
        "phase_2": "Short-term development (3-12 months)",
        "phase_3": "Long-term scaling (1-3 years)"
    }},
    "resource_requirements": {{
        "team_size": "Estimated team size and key roles needed",
        "budget_estimate": "Rough budget range and key cost factors",
        "technology_needs": "Technical infrastructure and tools required",
        "timeline": "Realistic timeline for meaningful results"
    }},
    "risk_analysis": {{
        "technical_risks": "Primary technical challenges and mitigation strategies",
        "market_risks": "Market adoption challenges and how to address them",
        "resource_risks": "Resource availability and capability gaps",
        "overall_risk_level": "Low/Medium/High with brief justification"
    }},
    "success_metrics": [
        "Key metric 1 - how to measure progress",
        "Key metric 2 - how to measure impact",
        "Key metric 3 - how to measure success"
    ],
    "next_steps": [
        "Immediate action 1 - what to do first",
        "Immediate action 2 - what to do second", 
        "Immediate action 3 - what to do third"
    ]
}}
```

Requirements:
- Be specific and actionable, not generic
- Base recommendations on the actual idea content
- Provide realistic timelines and resource estimates
- Focus on practical implementation details
- Include both opportunities and challenges

IMPORTANT: Return ONLY the JSON object, no additional text or explanations.
"""
        
        try:
            import ollama
            
            response = ollama.chat(
                model='mistral:7b-instruct-v0.3-q4_K_M',
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={'temperature': 0.7}  # Balanced creativity and structure
            )
            
            content = response['message']['content']
            
            # Extract JSON from response
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                json_text = content[start:end].strip()
            elif '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_text = content[start:end]
            else:
                raise ValueError("No JSON found in response")
            
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse report JSON for idea: {idea}")
                return self._generate_fallback_report(idea, problem_statement)
        
        except Exception as e:
            logger.error(f"Report generation failed for idea '{idea}': {e}")
            return self._generate_fallback_report(idea, problem_statement)
    
    def _generate_fallback_report(self, idea: str, problem_statement: str) -> Dict[str, Any]:
        """Generate a fallback report when LLM processing fails."""
        return {
            "idea_title": f"Implementation Plan: {idea[:50]}",
            "core_concept": f"This idea addresses the problem: {problem_statement}. The proposed solution is: {idea}. Further analysis and development planning is needed to fully elaborate on the implementation details.",
            "supporting_evidence": [
                "Idea generated through systematic ideation process",
                "Addresses identified opportunity from knowledge analysis",
                "Selected for development based on feasibility assessment"
            ],
            "potential_applications": [
                "Primary application area requires further investigation",
                "Secondary applications to be explored during development",
                "Cross-domain applications possible with adaptation"
            ],
            "implementation_roadmap": {
                "phase_1": "Conduct detailed feasibility study and requirements analysis",
                "phase_2": "Develop prototype and test core functionality", 
                "phase_3": "Scale implementation and refine based on feedback"
            },
            "resource_requirements": {
                "team_size": "2-5 people depending on scope and timeline",
                "budget_estimate": "To be determined based on detailed analysis",
                "technology_needs": "Standard development tools and infrastructure",
                "timeline": "6-18 months for meaningful results"
            },
            "risk_analysis": {
                "technical_risks": "Implementation complexity and technical feasibility to be assessed",
                "market_risks": "User adoption and market fit require validation",
                "resource_risks": "Budget and team availability need confirmation",
                "overall_risk_level": "Medium - requires further analysis"
            },
            "success_metrics": [
                "Completion of development milestones on schedule",
                "User acceptance and satisfaction ratings",
                "Achievement of functional requirements and objectives"
            ],
            "next_steps": [
                "Conduct detailed feasibility and requirements analysis",
                "Identify and assemble development team",
                "Create detailed project plan and timeline"
            ]
        }
    
    def save_deliver_results(self, deliver_results: Dict[str, Any], 
                           collection_name: str = "Unknown") -> Dict[str, Any]:
        """
        Save the Deliver phase results to structured files.
        
        Args:
            deliver_results: Results from the Deliver phase
            collection_name: Name of the source collection
            
        Returns:
            Dict containing save status and file paths
        """
        try:
            logger.info("Saving Deliver phase results to structured files")
            
            if deliver_results.get("status") != "success":
                return {
                    "status": "error",
                    "error": "Cannot save - Deliver results not successful"
                }
            
            # Create save directory with timestamp
            from datetime import datetime
            import os
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_collection = collection_name.replace(" ", "_").replace("/", "_")
            save_dir = Path(f"idea_generator_reports/{safe_collection}_{timestamp}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            detailed_reports = deliver_results.get("detailed_reports", {})
            saved_files = []
            
            # Save individual reports
            for i, (idea, report) in enumerate(detailed_reports.items(), 1):
                # Create safe filename
                safe_title = report.get("idea_title", f"Idea_{i}").replace(" ", "_").replace("/", "_")
                safe_title = "".join(c for c in safe_title if c.isalnum() or c in "_-")[:50]
                
                # Save as Markdown
                md_filename = f"{i:02d}_{safe_title}.md"
                md_path = save_dir / md_filename
                
                markdown_content = self._generate_markdown_report(idea, report, i)
                
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                saved_files.append(str(md_path))
                
                # Save as JSON for programmatic access
                json_filename = f"{i:02d}_{safe_title}.json"
                json_path = save_dir / json_filename
                
                json_data = {
                    "original_idea": idea,
                    "report": report,
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "collection_source": collection_name,
                        "report_number": i
                    }
                }
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                saved_files.append(str(json_path))
            
            # Save summary index
            summary_path = save_dir / "00_SUMMARY.md"
            summary_content = self._generate_summary_report(deliver_results, collection_name)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            saved_files.append(str(summary_path))
            
            # Save complete results as JSON
            complete_data = {
                "deliver_results": deliver_results,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "collection_source": collection_name,
                    "total_reports": len(detailed_reports)
                }
            }
            
            complete_path = save_dir / "complete_results.json"
            with open(complete_path, 'w', encoding='utf-8') as f:
                json.dump(complete_data, f, indent=2, ensure_ascii=False)
            
            saved_files.append(str(complete_path))
            
            logger.info(f"Saved {len(detailed_reports)} reports to {save_dir}")
            
            return {
                "status": "success",
                "save_directory": str(save_dir),
                "saved_files": saved_files,
                "report_count": len(detailed_reports)
            }
            
        except Exception as e:
            error_msg = f"Save operation failed: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    def _generate_markdown_report(self, idea: str, report: Dict[str, Any], report_number: int) -> str:
        """Generate a formatted Markdown report for an individual idea."""
        
        title = report.get("idea_title", f"Idea Report {report_number}")
        
        md = f"""# {title}

## Original Idea
{idea}

## Core Concept
{report.get("core_concept", "No core concept provided")}

## Supporting Evidence
"""
        
        evidence = report.get("supporting_evidence", [])
        for i, point in enumerate(evidence, 1):
            md += f"{i}. {point}\n"
        
        md += f"""
## Potential Applications
"""
        
        applications = report.get("potential_applications", [])
        for app in applications:
            md += f"- {app}\n"
        
        md += f"""
## Implementation Roadmap

### Phase 1 (0-3 months)
{report.get("implementation_roadmap", {}).get("phase_1", "Not specified")}

### Phase 2 (3-12 months)
{report.get("implementation_roadmap", {}).get("phase_2", "Not specified")}

### Phase 3 (1-3 years)
{report.get("implementation_roadmap", {}).get("phase_3", "Not specified")}

## Resource Requirements

"""
        
        resources = report.get("resource_requirements", {})
        md += f"- **Team Size:** {resources.get('team_size', 'Not specified')}\n"
        md += f"- **Budget Estimate:** {resources.get('budget_estimate', 'Not specified')}\n"
        md += f"- **Technology Needs:** {resources.get('technology_needs', 'Not specified')}\n"
        md += f"- **Timeline:** {resources.get('timeline', 'Not specified')}\n"
        
        md += f"""
## Risk Analysis

### Overall Risk Level
{report.get("risk_analysis", {}).get("overall_risk_level", "Unknown")}

### Technical Risks
{report.get("risk_analysis", {}).get("technical_risks", "Not assessed")}

### Market Risks
{report.get("risk_analysis", {}).get("market_risks", "Not assessed")}

### Resource Risks
{report.get("risk_analysis", {}).get("resource_risks", "Not assessed")}

## Success Metrics
"""
        
        metrics = report.get("success_metrics", [])
        for i, metric in enumerate(metrics, 1):
            md += f"{i}. {metric}\n"
        
        md += f"""
## Next Steps
"""
        
        next_steps = report.get("next_steps", [])
        for i, step in enumerate(next_steps, 1):
            md += f"{i}. {step}\n"
        
        md += f"""
---
*Generated by Cortex Suite Idea Generator*
"""
        
        return md
    
    def _generate_summary_report(self, deliver_results: Dict[str, Any], collection_name: str) -> str:
        """Generate a summary Markdown report for all ideas."""
        
        from datetime import datetime
        
        detailed_reports = deliver_results.get("detailed_reports", {})
        
        md = f"""# Idea Generator Report Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Source Collection:** {collection_name}  
**Total Ideas:** {len(detailed_reports)}

## Overview

This report contains detailed analysis and implementation plans for {len(detailed_reports)} innovative ideas generated through the Double Diamond methodology applied to the "{collection_name}" knowledge collection.

## Generated Reports

"""
        
        for i, (idea, report) in enumerate(detailed_reports.items(), 1):
            title = report.get("idea_title", f"Idea {i}")
            risk_level = report.get("risk_analysis", {}).get("overall_risk_level", "Unknown")
            
            md += f"""### {i}. {title}
- **Original Idea:** {idea[:100]}{'...' if len(idea) > 100 else ''}
- **Risk Level:** {risk_level}
- **File:** `{i:02d}_{title.replace(' ', '_')}.md`

"""
        
        md += f"""
## Files in This Report

- `00_SUMMARY.md` - This overview document
- `XX_[IdeaTitle].md` - Individual idea reports in Markdown format
- `XX_[IdeaTitle].json` - Individual idea reports in JSON format
- `complete_results.json` - Complete results data for programmatic access

## How to Use These Reports

1. **Review the Summary** - Start with this document for an overview
2. **Read Individual Reports** - Each idea has a detailed Markdown report
3. **Access Structured Data** - Use JSON files for integration with other tools
4. **Follow Implementation Plans** - Each report includes phased roadmaps and next steps

---
*Generated by Cortex Suite Idea Generator using the Double Diamond methodology*
"""
        
        return md
    
    def analyze_collection_for_filters(self, collection_name: str) -> Dict[str, Any]:
        """
        Analyze a collection to provide filter suggestions and statistics.
        
        Args:
            collection_name: Name of the collection to analyze
            
        Returns:
            Dict containing filter options and collection statistics
        """
        try:
            logger.info(f"Analyzing collection '{collection_name}' for filter options")
            
            if self.vector_index is None:
                return {"error": "Vector index not available"}
            
            # Get all documents in the collection
            doc_ids = self.collection_mgr.get_doc_ids_by_name(collection_name)
            if not doc_ids:
                return {"error": "Collection is empty"}
            
            # Query vector store for all collection documents
            vector_store = self.vector_index.vector_store
            results = vector_store._collection.get(
                where={"doc_id": {"$in": doc_ids}},
                include=["metadatas"]
            )
            
            if not results or 'metadatas' not in results:
                return {"error": "No metadata found"}
            
            # Analyze metadata to extract filter options
            doc_types = set()
            outcomes = set()
            thematic_tags = set()
            date_range = {"earliest": None, "latest": None}
            
            for metadata in results['metadatas']:
                # Document types
                if metadata.get('document_type'):
                    doc_types.add(metadata['document_type'])
                
                # Proposal outcomes
                if metadata.get('proposal_outcome'):
                    outcomes.add(metadata['proposal_outcome'])
                
                # Thematic tags
                if metadata.get('thematic_tags'):
                    tags = metadata['thematic_tags'].split(', ')
                    thematic_tags.update(tag.strip() for tag in tags if tag.strip())
                
                # Date range
                if metadata.get('last_modified_date'):
                    try:
                        from datetime import datetime
                        date = datetime.fromisoformat(metadata['last_modified_date'])
                        if date_range["earliest"] is None or date < date_range["earliest"]:
                            date_range["earliest"] = date
                        if date_range["latest"] is None or date > date_range["latest"]:
                            date_range["latest"] = date
                    except:
                        pass
            
            # Convert to lists and sort
            filter_options = {
                "document_types": sorted(list(doc_types)),
                "proposal_outcomes": sorted(list(outcomes)),
                "thematic_tags": sorted(list(thematic_tags)),
                "total_documents": len(results['metadatas']),
                "date_range": {
                    "earliest": date_range["earliest"].isoformat() if date_range["earliest"] else None,
                    "latest": date_range["latest"].isoformat() if date_range["latest"] else None
                }
            }
            
            # Generate statistics
            doc_type_counts = {}
            outcome_counts = {}
            for metadata in results['metadatas']:
                doc_type = metadata.get('document_type', 'Unknown')
                outcome = metadata.get('proposal_outcome', 'Unknown')
                doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            filter_options["statistics"] = {
                "document_type_distribution": doc_type_counts,
                "outcome_distribution": outcome_counts
            }
            
            logger.info(f"Collection analysis complete: {len(doc_types)} document types, {len(thematic_tags)} unique tags")
            return filter_options
            
        except Exception as e:
            logger.error(f"Collection analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_filtered_collection(self, collection_name: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform enhanced analysis of a collection with applied filters.
        
        Args:
            collection_name: Name of the collection to analyze
            filters: Optional filters to apply
            
        Returns:
            Dict containing detailed filtered analysis
        """
        try:
            logger.info(f"Performing enhanced analysis of collection '{collection_name}' with filters: {filters}")
            
            if self.vector_index is None:
                return {"error": "Vector index not available"}
            
            # Get filtered collection content
            collection_content = self._get_collection_content(collection_name, filters)
            if not collection_content:
                return {"error": "No documents found after filtering"}
            
            # Analyze the filtered content
            analysis = {
                "total_documents": len(collection_content),
                "filters_applied": filters or {},
                "content_themes": [],
                "entity_insights": {},
                "temporal_patterns": {},
                "quality_metrics": {}
            }
            
            # Extract metadata for analysis
            metadatas = []
            contents = []
            for item in collection_content:
                metadatas.append(item.get('metadata', {}))
                contents.append(item.get('content', ''))
            
            # Theme analysis from filtered content
            if contents:
                # Combine all content for theme extraction
                combined_content = ' '.join(contents[:50])  # Limit for performance
                
                # Simple keyword extraction (could be enhanced with NLP)
                from collections import Counter
                words = combined_content.lower().split()
                # Filter out common words and extract meaningful terms
                stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
                meaningful_words = [word for word in words if len(word) > 3 and word not in stopwords]
                common_themes = Counter(meaningful_words).most_common(10)
                analysis["content_themes"] = [{"theme": word, "frequency": count} for word, count in common_themes]
            
            # Entity insights from graph (if available)
            if self.graph_manager and filters:
                if filters.get('consultant'):
                    consultant_projects = self.graph_manager.query_consultant_projects(filters['consultant'])
                    analysis["entity_insights"]["consultant_expertise"] = {
                        "consultant": filters['consultant'],
                        "total_projects": len(consultant_projects),
                        "document_types": list(set(p.get('type', 'Unknown') for p in consultant_projects))
                    }
                
                if filters.get('client'):
                    client_projects = self.graph_manager.query_client_projects(filters['client'])
                    analysis["entity_insights"]["client_patterns"] = {
                        "client": filters['client'],
                        "total_projects": len(client_projects),
                        "engagement_types": list(set(p.get('metadata', {}).get('document_type', 'Unknown') for p in client_projects))
                    }
            
            # Temporal patterns
            dates = []
            for metadata in metadatas:
                if metadata.get('last_modified_date'):
                    try:
                        from datetime import datetime
                        date = datetime.fromisoformat(metadata['last_modified_date'])
                        dates.append(date)
                    except:
                        pass
            
            if dates:
                dates.sort()
                analysis["temporal_patterns"] = {
                    "date_range": {
                        "earliest": dates[0].isoformat(),
                        "latest": dates[-1].isoformat()
                    },
                    "span_days": (dates[-1] - dates[0]).days if len(dates) > 1 else 0
                }
            
            # Quality metrics
            doc_type_dist = {}
            outcome_dist = {}
            for metadata in metadatas:
                doc_type = metadata.get('document_type', 'Unknown')
                outcome = metadata.get('proposal_outcome', 'N/A')
                doc_type_dist[doc_type] = doc_type_dist.get(doc_type, 0) + 1
                outcome_dist[outcome] = outcome_dist.get(outcome, 0) + 1
            
            analysis["quality_metrics"] = {
                "document_type_distribution": doc_type_dist,
                "outcome_distribution": outcome_dist,
                "success_rate": outcome_dist.get('Won', 0) / max(1, sum(v for k, v in outcome_dist.items() if k != 'N/A')) if outcome_dist else 0
            }
            
            logger.info(f"Enhanced analysis complete: {analysis['total_documents']} filtered documents analyzed")
            return analysis
            
        except Exception as e:
            logger.error(f"Enhanced collection analysis failed: {e}")
            return {"error": str(e)}
    
    def get_entity_filter_options(self, collection_name: str) -> Dict[str, Any]:
        """
        Get entity-based filter options from the knowledge graph.
        
        Args:
            collection_name: Name of the collection to analyze
            
        Returns:
            Dict containing consultant and client filter options
        """
        try:
            if self.graph_manager is None:
                return {"consultants": [], "clients": [], "organizations": []}
            
            # Get collection document IDs
            doc_ids = self.collection_mgr.get_doc_ids_by_name(collection_name)
            if not doc_ids:
                return {"consultants": [], "clients": [], "organizations": []}
            
            consultants = set()
            clients = set()
            organizations = set()
            
            # Extract entities related to collection documents
            for doc_id in doc_ids:
                if doc_id in self.graph_manager.graph:
                    # Find all entities connected to this document
                    for neighbor in self.graph_manager.graph.neighbors(doc_id):
                        neighbor_data = self.graph_manager.graph.nodes.get(neighbor, {})
                        entity_type = neighbor_data.get('entity_type', '')
                        
                        if entity_type == 'person':
                            # Extract person name from ID (format: "person:Name")
                            name = neighbor.split(':', 1)[-1] if ':' in neighbor else neighbor
                            consultants.add(name)
                        elif entity_type == 'organization':
                            # Extract organization name from ID
                            name = neighbor.split(':', 1)[-1] if ':' in neighbor else neighbor
                            organizations.add(name)
                            clients.add(name)  # Organizations can be clients
            
            return {
                "consultants": sorted(list(consultants)),
                "clients": sorted(list(clients)),
                "organizations": sorted(list(organizations))
            }
            
        except Exception as e:
            logger.error(f"Entity filter analysis failed: {e}")
            return {"consultants": [], "clients": [], "organizations": []}
    
    def apply_entity_filters(self, doc_ids: List[str], filters: Dict[str, Any]) -> List[str]:
        """
        Filter document IDs based on entity relationships.
        
        Args:
            doc_ids: List of document IDs to filter
            filters: Dict containing entity filter criteria
            
        Returns:
            Filtered list of document IDs
        """
        try:
            if self.graph_manager is None or not filters:
                return doc_ids
            
            filtered_docs = set(doc_ids)
            
            # Filter by consultant
            if filters.get("consultant_filter"):
                consultant_name = filters["consultant_filter"]
                consultant_id = f"person:{consultant_name}"
                
                if consultant_id in self.graph_manager.graph:
                    # Find documents authored by this consultant
                    consultant_docs = set()
                    for neighbor in self.graph_manager.graph.neighbors(consultant_id):
                        edge_data = self.graph_manager.graph.edges.get((consultant_id, neighbor), {})
                        if edge_data.get('relationship_type') == 'authored':
                            consultant_docs.add(neighbor)
                    
                    filtered_docs = filtered_docs.intersection(consultant_docs)
                else:
                    logger.warning(f"Consultant '{consultant_name}' not found in graph")
                    return []  # No documents if consultant not found
            
            # Filter by client/organization
            if filters.get("client_filter"):
                client_name = filters["client_filter"]
                client_id = f"organization:{client_name}"
                
                if client_id in self.graph_manager.graph:
                    # Find documents related to this client
                    client_docs = set()
                    for neighbor in self.graph_manager.graph.neighbors(client_id):
                        edge_data = self.graph_manager.graph.edges.get((client_id, neighbor), {})
                        if edge_data.get('relationship_type') in ['client_of', 'mentioned_in']:
                            client_docs.add(neighbor)
                    
                    filtered_docs = filtered_docs.intersection(client_docs)
                else:
                    logger.warning(f"Client '{client_name}' not found in graph")
                    return []  # No documents if client not found
            
            return list(filtered_docs)
            
        except Exception as e:
            logger.error(f"Entity filtering failed: {e}")
            return doc_ids  # Return original list on error

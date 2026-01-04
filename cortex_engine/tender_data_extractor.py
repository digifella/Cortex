"""
Tender Data Extractor - Extract Structured Data from Unstructured KB
Version: 1.0.0
Date: 2026-01-03

Purpose: Extract structured organizational data from unstructured knowledge base
documents (PDFs, Word docs, emails, etc.) to auto-fill tender document fields.

Extraction Strategy:
1. Query vector store for relevant documents (insurance, org details, qualifications, etc.)
2. Extract entities from knowledge graph (people, organizations, projects)
3. Use LLM with structured output (JSON mode) to parse unstructured text
4. Validate and parse into Pydantic models
5. Cache results in structured_knowledge.json

This creates a structured data layer that can be quickly queried during tender
field matching, avoiding slow re-extraction on every tender.
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from cortex_engine.tender_schema import (
    StructuredKnowledge,
    OrganizationProfile,
    Insurance,
    Qualification,
    WorkExperience,
    ProjectExperience,
    Reference,
    Capability,
    InsuranceType,
    QualificationType
)
from cortex_engine.utils.modern_ollama_llm import ModernOllamaLLM
from cortex_engine.adaptive_model_manager import AdaptiveModelManager, TaskType

logger = logging.getLogger(__name__)


class TenderDataExtractor:
    """
    Extracts structured organizational data from unstructured knowledge base.
    Uses vector search + knowledge graph + LLM to populate tender schemas.
    """

    def __init__(
        self,
        vector_index: Any,
        knowledge_graph: Any,
        model_manager: AdaptiveModelManager,
        db_path: Path
    ):
        """
        Initialize extractor.

        Args:
            vector_index: LlamaIndex vector store or ChromaDB collection for document retrieval
            knowledge_graph: NetworkX knowledge graph for entity extraction
            model_manager: Adaptive model manager for LLM selection
            db_path: Database path for storing structured_knowledge.json
        """
        self.vector_index = vector_index
        self.knowledge_graph = knowledge_graph
        self.model_manager = model_manager
        self.db_path = Path(db_path)

        self.structured_data_path = self.db_path / "structured_knowledge.json"

        # Entity-specific filtering
        self.entity_id: Optional[str] = None
        self.document_filter: Optional[List[str]] = None  # Filter to specific doc IDs

    async def extract_all_structured_data(
        self,
        progress_callback=None,
        entity_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> StructuredKnowledge:
        """
        Extract all structured data from KB in one pass.
        This is the main entry point - called by UI button.

        Args:
            progress_callback: Optional callback for progress updates
            entity_id: Optional entity ID for filtering (stored in result)
            document_ids: Optional list of document IDs to restrict extraction to

        Returns:
            StructuredKnowledge object with all extracted data
        """

        # Set entity-specific filters
        self.entity_id = entity_id
        self.document_filter = document_ids

        if progress_callback:
            if entity_id:
                doc_count = len(document_ids) if document_ids else "all"
                progress_callback(f"🔍 Starting extraction for entity '{entity_id}' from {doc_count} documents...")
            else:
                progress_callback("🔍 Starting structured data extraction from knowledge base...")

        # Initialize structured knowledge container
        structured = StructuredKnowledge()

        # Extract each category
        try:
            # 1. Organization profile
            if progress_callback:
                progress_callback("📋 Extracting organization profile...")
            structured.organization = await self._extract_organization_profile()

            # 2. Insurance policies
            if progress_callback:
                progress_callback("🛡️ Extracting insurance policies...")
            structured.insurances = await self._extract_insurances()

            # 3. Team qualifications
            if progress_callback:
                progress_callback("🎓 Extracting team qualifications...")
            structured.team_qualifications = await self._extract_qualifications()

            # 4. Work experience
            if progress_callback:
                progress_callback("💼 Extracting work experience...")
            structured.team_work_experience = await self._extract_work_experience()

            # 5. Project experience
            if progress_callback:
                progress_callback("🚀 Extracting project experience...")
            structured.projects = await self._extract_projects()

            # 6. References
            if progress_callback:
                progress_callback("📞 Extracting references...")
            structured.references = await self._extract_references()

            # 7. Capabilities
            if progress_callback:
                progress_callback("⭐ Extracting organizational capabilities...")
            structured.capabilities = await self._extract_capabilities()

            # Set metadata
            structured.extraction_date = datetime.now()
            structured.total_documents_processed = self._get_kb_document_count()

            # Save to file
            if progress_callback:
                progress_callback("💾 Saving structured data to file...")
            self._save_structured_knowledge(structured)

            if progress_callback:
                stats = structured.summary_stats
                progress_callback(f"✅ Extraction complete! Found: {stats}")

            logger.info(f"Structured data extraction complete: {structured.summary_stats}")
            return structured

        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"❌ Extraction failed: {str(e)}")
            raise

    async def _extract_organization_profile(self) -> Optional[OrganizationProfile]:
        """Extract core organization details (name, ABN, ACN, address, contact)."""

        try:
            # Query vector store for organization-related content
            org_queries = [
                "organization legal name ABN ACN registration",
                "company address contact details phone email",
                "business registration trading name"
            ]

            context_parts = []
            source_docs = set()

            for query in org_queries:
                results = await self._query_vector_store(query, top_k=3)
                context_parts.append(results["content"])
                source_docs.update(results["sources"])

            combined_context = "\n\n".join(context_parts)

            # Build extraction prompt
            prompt = self._build_extraction_prompt(
                category="Organization Profile",
                context=combined_context,
                schema_description="""
Extract organization details:
- legal_name: Official registered business name
- trading_names: List of trading names or DBA names
- abn: Australian Business Number (11 digits)
- acn: Australian Company Number (9 digits)
- address: Dict with keys: street, city, state, postcode, country
- phone: Primary phone number
- email: Primary email address
- website: Organization website URL

Return JSON matching this structure.
"""
            )

            # Get LLM with JSON mode
            model_name = await self.model_manager.recommend_model(
                TaskType.ANALYSIS,
                preference="balanced"
            )

            llm = ModernOllamaLLM(model=model_name, request_timeout=120.0)

            response = await llm.acomplete(
                prompt,
                options={"temperature": 0.1, "format": "json"}
            )

            # Parse JSON response
            org_data = json.loads(response.text)
            org_data["source_documents"] = list(source_docs)
            org_data["last_updated"] = datetime.now()

            return OrganizationProfile(**org_data)

        except Exception as e:
            logger.error(f"Organization profile extraction failed: {e}")
            return None

    async def _extract_insurances(self) -> List[Insurance]:
        """Extract insurance policy details."""

        try:
            # Query for insurance documents
            query = "insurance policy coverage liability indemnity certificate number expiry date"
            results = await self._query_vector_store(query, top_k=10)

            # Also check knowledge graph for insurance entities
            insurance_entities = self._get_entities_by_type("Insurance")

            context = results["content"]
            if insurance_entities:
                context += f"\n\nKnowledge Graph Entities:\n{json.dumps(insurance_entities, indent=2)}"

            prompt = self._build_extraction_prompt(
                category="Insurance Policies",
                context=context,
                schema_description="""
Extract all insurance policies as a JSON array. Each policy should have:
- insurance_type: One of: "Public Liability", "Professional Indemnity", "Workers Compensation", "Cyber Liability", "Product Liability", "Other"
- insurer: Insurance company name
- policy_number: Policy number
- coverage_amount: Coverage limit in AUD (number)
- expiry_date: Expiry date (YYYY-MM-DD format)
- effective_date: Start date (YYYY-MM-DD format)
- coverage_description: What is covered (optional)

Return JSON array: [{"insurance_type": "...", "insurer": "...", ...}, ...]
"""
            )

            model_name = await self.model_manager.recommend_model(TaskType.ANALYSIS, preference="balanced")
            llm = ModernOllamaLLM(model=model_name, request_timeout=180.0)

            response = await llm.acomplete(prompt, options={"temperature": 0.1, "format": "json"})

            # Parse response
            insurances_data = json.loads(response.text)
            if not isinstance(insurances_data, list):
                insurances_data = [insurances_data]

            insurances = []
            for ins_data in insurances_data:
                ins_data["source_documents"] = results["sources"]
                ins_data["last_updated"] = datetime.now()
                try:
                    insurances.append(Insurance(**ins_data))
                except Exception as e:
                    logger.warning(f"Failed to parse insurance entry: {e}")

            return insurances

        except Exception as e:
            logger.error(f"Insurance extraction failed: {e}")
            return []

    async def _extract_qualifications(self) -> List[Qualification]:
        """Extract team member qualifications, certifications, memberships."""

        try:
            # Query for qualifications
            query = "qualifications certifications degrees diplomas professional membership credentials"
            results = await self._query_vector_store(query, top_k=10)

            # Get people from knowledge graph
            people = self._get_entities_by_type("Person")

            context = results["content"]
            if people:
                context += f"\n\nTeam Members from Knowledge Graph:\n{json.dumps(people, indent=2)}"

            prompt = self._build_extraction_prompt(
                category="Team Qualifications",
                context=context,
                schema_description="""
Extract all qualifications as a JSON array. Each qualification should have:
- person_name: Name of person holding qualification
- qualification_name: Name of qualification/certification
- qualification_type: One of: "Certification", "Degree", "Diploma", "License", "Professional Membership", "Other"
- institution: Issuing institution/body (optional)
- date_obtained: Date obtained (YYYY-MM-DD format, optional)
- expiry_date: Expiry date if applicable (YYYY-MM-DD format, optional)
- credential_id: Certificate/credential number (optional)

Return JSON array: [{"person_name": "...", "qualification_name": "...", ...}, ...]
"""
            )

            model_name = await self.model_manager.recommend_model(TaskType.ANALYSIS, preference="balanced")
            llm = ModernOllamaLLM(model=model_name, request_timeout=180.0)

            response = await llm.acomplete(prompt, options={"temperature": 0.1, "format": "json"})

            qualifications_data = json.loads(response.text)
            if not isinstance(qualifications_data, list):
                qualifications_data = [qualifications_data]

            qualifications = []
            for qual_data in qualifications_data:
                qual_data["source_documents"] = results["sources"]
                qual_data["last_updated"] = datetime.now()
                try:
                    qualifications.append(Qualification(**qual_data))
                except Exception as e:
                    logger.warning(f"Failed to parse qualification entry: {e}")

            return qualifications

        except Exception as e:
            logger.error(f"Qualifications extraction failed: {e}")
            return []

    async def _extract_work_experience(self) -> List[WorkExperience]:
        """Extract team member work experience."""

        try:
            query = "work experience employment history roles responsibilities achievements career"
            results = await self._query_vector_store(query, top_k=10)

            people = self._get_entities_by_type("Person")

            context = results["content"]
            if people:
                context += f"\n\nTeam Members:\n{json.dumps(people, indent=2)}"

            prompt = self._build_extraction_prompt(
                category="Work Experience",
                context=context,
                schema_description="""
Extract work experience as a JSON array. Each entry should have:
- person_name: Name of person
- role: Job title/role
- organization: Employer/organization name
- start_date: Start date (YYYY-MM-DD format, optional)
- end_date: End date (YYYY-MM-DD format, optional - omit if current)
- responsibilities: List of key responsibilities (optional)
- achievements: List of notable achievements (optional)
- technologies: List of technologies/tools used (optional)

Return JSON array: [{"person_name": "...", "role": "...", ...}, ...]
"""
            )

            model_name = await self.model_manager.recommend_model(TaskType.ANALYSIS, preference="balanced")
            llm = ModernOllamaLLM(model=model_name, request_timeout=180.0)

            response = await llm.acomplete(prompt, options={"temperature": 0.1, "format": "json"})

            experience_data = json.loads(response.text)
            if not isinstance(experience_data, list):
                experience_data = [experience_data]

            experiences = []
            for exp_data in experience_data:
                exp_data["source_documents"] = results["sources"]
                exp_data["last_updated"] = datetime.now()
                try:
                    experiences.append(WorkExperience(**exp_data))
                except Exception as e:
                    logger.warning(f"Failed to parse work experience entry: {e}")

            return experiences

        except Exception as e:
            logger.error(f"Work experience extraction failed: {e}")
            return []

    async def _extract_projects(self) -> List[ProjectExperience]:
        """Extract past project experience."""

        try:
            query = "project experience client deliverables outcomes case study portfolio work"
            results = await self._query_vector_store(query, top_k=15)

            project_entities = self._get_entities_by_type("Project")

            context = results["content"]
            if project_entities:
                context += f"\n\nProjects from Knowledge Graph:\n{json.dumps(project_entities, indent=2)}"

            prompt = self._build_extraction_prompt(
                category="Project Experience",
                context=context,
                schema_description="""
Extract project experience as a JSON array. Each project should have:
- project_name: Name of project
- client: Client/organization name
- start_date: Project start date (YYYY-MM-DD format, optional)
- end_date: Project end date (YYYY-MM-DD format, optional - omit if ongoing)
- description: Project description
- role: Your organization's role in the project (optional)
- value: Project value in AUD (number, optional)
- deliverables: List of key deliverables (optional)
- outcomes: List of project outcomes/benefits (optional)
- technologies: List of technologies used (optional)
- team_size: Team size (number, optional)

Return JSON array: [{"project_name": "...", "client": "...", ...}, ...]
"""
            )

            model_name = await self.model_manager.recommend_model(TaskType.ANALYSIS, preference="balanced")
            llm = ModernOllamaLLM(model=model_name, request_timeout=240.0)

            response = await llm.acomplete(prompt, options={"temperature": 0.1, "format": "json"})

            projects_data = json.loads(response.text)
            if not isinstance(projects_data, list):
                projects_data = [projects_data]

            projects = []
            for proj_data in projects_data:
                proj_data["source_documents"] = results["sources"]
                proj_data["last_updated"] = datetime.now()
                try:
                    projects.append(ProjectExperience(**proj_data))
                except Exception as e:
                    logger.warning(f"Failed to parse project entry: {e}")

            return projects

        except Exception as e:
            logger.error(f"Project extraction failed: {e}")
            return []

    async def _extract_references(self) -> List[Reference]:
        """Extract client/partner references."""

        try:
            query = "reference contact testimonial client feedback referee details"
            results = await self._query_vector_store(query, top_k=10)

            prompt = self._build_extraction_prompt(
                category="References",
                context=results["content"],
                schema_description="""
Extract references as a JSON array. Each reference should have:
- contact_name: Reference contact person name
- contact_title: Contact's job title (optional)
- organization: Organization name
- phone: Contact phone number (optional)
- email: Contact email (optional)
- relationship: Nature of relationship (e.g., "Client - Project Manager")
- project_context: Which project(s) they can speak to (optional)

Return JSON array: [{"contact_name": "...", "organization": "...", ...}, ...]
"""
            )

            model_name = await self.model_manager.recommend_model(TaskType.ANALYSIS, preference="balanced")
            llm = ModernOllamaLLM(model=model_name, request_timeout=120.0)

            response = await llm.acomplete(prompt, options={"temperature": 0.1, "format": "json"})

            references_data = json.loads(response.text)
            if not isinstance(references_data, list):
                references_data = [references_data]

            references = []
            for ref_data in references_data:
                ref_data["source_documents"] = results["sources"]
                ref_data["last_updated"] = datetime.now()
                try:
                    references.append(Reference(**ref_data))
                except Exception as e:
                    logger.warning(f"Failed to parse reference entry: {e}")

            return references

        except Exception as e:
            logger.error(f"References extraction failed: {e}")
            return []

    async def _extract_capabilities(self) -> List[Capability]:
        """Extract organizational capabilities and certifications."""

        try:
            query = "organizational capability certification accreditation ISO standard compliance"
            results = await self._query_vector_store(query, top_k=10)

            prompt = self._build_extraction_prompt(
                category="Organizational Capabilities",
                context=results["content"],
                schema_description="""
Extract capabilities/certifications as a JSON array. Each should have:
- capability_name: Name of capability/certification
- description: What this capability enables
- certification_body: Certifying organization (optional)
- certification_number: Certificate/registration number (optional)
- date_obtained: Date obtained (YYYY-MM-DD format, optional)
- expiry_date: Expiry date (YYYY-MM-DD format, optional)
- scope: Scope of certification (optional)

Return JSON array: [{"capability_name": "...", "description": "...", ...}, ...]
"""
            )

            model_name = await self.model_manager.recommend_model(TaskType.ANALYSIS, preference="balanced")
            llm = ModernOllamaLLM(model=model_name, request_timeout=120.0)

            response = await llm.acomplete(prompt, options={"temperature": 0.1, "format": "json"})

            capabilities_data = json.loads(response.text)
            if not isinstance(capabilities_data, list):
                capabilities_data = [capabilities_data]

            capabilities = []
            for cap_data in capabilities_data:
                cap_data["source_documents"] = results["sources"]
                cap_data["last_updated"] = datetime.now()
                try:
                    capabilities.append(Capability(**cap_data))
                except Exception as e:
                    logger.warning(f"Failed to parse capability entry: {e}")

            return capabilities

        except Exception as e:
            logger.error(f"Capabilities extraction failed: {e}")
            return []

    # ========== Helper Methods ==========

    async def _query_vector_store(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Query vector store and return content + sources."""

        try:
            # Import embedding service at function level to avoid circular imports
            from cortex_engine.embedding_service import embed_query

            # Check if vector_index is ChromaDB collection
            # ChromaDB collections have methods like: query, get, add, update, delete
            if hasattr(self.vector_index, 'query') and hasattr(self.vector_index, 'get'):
                # ChromaDB collection - use direct query
                logger.debug(f"Using ChromaDB collection for query: {query}")

                # Get query embedding using local SentenceTransformer
                query_embedding = embed_query(query)

                # Build where clause for document filtering
                where_clause = None
                if self.document_filter:
                    # Filter to specific documents
                    where_clause = {"id": {"$in": self.document_filter}}
                    logger.debug(f"Filtering to {len(self.document_filter)} documents")

                # Query ChromaDB
                results = self.vector_index.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_clause if where_clause else None
                )

                # Extract content and sources
                content_parts = []
                sources = set()

                if results and 'documents' in results and results['documents']:
                    for doc_list in results['documents']:
                        content_parts.extend(doc_list)

                if results and 'metadatas' in results and results['metadatas']:
                    for metadata_list in results['metadatas']:
                        for metadata in metadata_list:
                            if metadata:
                                source = metadata.get('file_name', metadata.get('source', 'Unknown'))
                                if source != 'Unknown':
                                    sources.add(source)

                return {
                    "content": "\n\n".join(content_parts),
                    "sources": list(sources)
                }
            else:
                logger.error(f"Vector index type not recognized. Has query: {hasattr(self.vector_index, 'query')}, Has get: {hasattr(self.vector_index, 'get')}")
                logger.error(f"Vector index type: {type(self.vector_index)}")
                return {"content": "", "sources": []}

        except Exception as e:
            logger.error(f"Vector store query failed for '{query}': {e}")
            return {"content": "", "sources": []}

    def _get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get entities of specific type from knowledge graph."""

        try:
            if not self.knowledge_graph:
                return []

            entities = []
            for node_id, node_data in self.knowledge_graph.nodes(data=True):
                if node_data.get('type') == entity_type:
                    entities.append({
                        "id": node_id,
                        "type": entity_type,
                        **node_data
                    })

            return entities

        except Exception as e:
            logger.error(f"Knowledge graph entity extraction failed: {e}")
            return []

    def _get_kb_document_count(self) -> int:
        """Get total number of documents in KB."""
        try:
            if self.vector_index:
                return len(self.vector_index.docstore.docs)
            return 0
        except Exception:
            return 0

    def _build_extraction_prompt(
        self,
        category: str,
        context: str,
        schema_description: str
    ) -> str:
        """Build extraction prompt for LLM."""

        return f"""You are extracting structured data from unstructured knowledge base documents.

**Category:** {category}

**Context from Knowledge Base:**
{context}

**Extraction Task:**
{schema_description}

**Important Instructions:**
1. Extract ONLY information that is clearly stated in the context
2. Do NOT fabricate or guess information
3. Use null or omit fields if information is not available
4. Return valid JSON that matches the schema exactly
5. For dates, use YYYY-MM-DD format
6. For lists, return empty arrays [] if no items found

**Output:**
Return ONLY the JSON object or array, no additional text or explanation.
"""

    def _save_structured_knowledge(self, structured: StructuredKnowledge):
        """Save structured knowledge to JSON file."""

        try:
            # Ensure directory exists
            self.structured_data_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to JSON-serializable dict
            data = structured.to_json_serializable()

            # Write to file
            with open(self.structured_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Structured knowledge saved to {self.structured_data_path}")

        except Exception as e:
            logger.error(f"Failed to save structured knowledge: {e}")
            raise

    def load_structured_knowledge(self) -> Optional[StructuredKnowledge]:
        """Load previously extracted structured knowledge from file."""

        try:
            if not self.structured_data_path.exists():
                logger.info("No structured knowledge file found")
                return None

            with open(self.structured_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            structured = StructuredKnowledge.from_json(data)
            logger.info(f"Loaded structured knowledge: {structured.summary_stats}")

            return structured

        except Exception as e:
            logger.error(f"Failed to load structured knowledge: {e}")
            return None

    def is_extraction_stale(self, max_age_days: int = 30) -> bool:
        """Check if extracted data is older than max_age_days."""

        try:
            structured = self.load_structured_knowledge()
            if not structured:
                return True

            age = datetime.now() - structured.extraction_date
            return age.days > max_age_days

        except Exception:
            return True

    async def populate_workspace_with_extraction(
        self,
        workspace_manager: Any,
        workspace_id: str,
        structured_data: StructuredKnowledge
    ) -> bool:
        """
        Populate a workspace with extracted structured data.

        Args:
            workspace_manager: WorkspaceManager instance
            workspace_id: Workspace ID to populate
            structured_data: Extracted structured data

        Returns:
            True if successful
        """
        from .workspace_schema import DocumentSource

        try:
            logger.info(f"Populating workspace {workspace_id} with extracted data")

            # 1. Save entity snapshot (JSON)
            entity_data_dict = structured_data.to_json_serializable()
            workspace_manager.add_entity_snapshot(workspace_id, entity_data_dict)

            # 2. Add organization data to collection
            if structured_data.organization:
                org = structured_data.organization
                content = f"""Organization: {org.legal_name}
ABN: {org.abn or 'Not available'}
ACN: {org.acn or 'Not available'}
Address: {org.address.get('street', '')}, {org.address.get('city', '')}, {org.address.get('state', '')} {org.address.get('postcode', '')}
Phone: {org.phone or 'Not available'}
Email: {org.email or 'Not available'}
Website: {org.website or 'Not available'}
"""
                workspace_manager.add_document_to_workspace(
                    workspace_id=workspace_id,
                    content=content,
                    source_type=DocumentSource.ENTITY_DATA,
                    metadata={"category": "organization", "entity_name": org.legal_name},
                    doc_id="org_profile"
                )

            # 3. Add insurances to collection
            for i, insurance in enumerate(structured_data.insurances):
                content = f"""Insurance: {insurance.insurance_type.value}
Insurer: {insurance.insurer}
Policy Number: {insurance.policy_number}
Coverage: ${insurance.coverage_amount or 'Not specified'}
Effective Date: {insurance.effective_date}
Expiry Date: {insurance.expiry_date}
Status: {'Expired' if insurance.is_expired else 'Active'}
"""
                workspace_manager.add_document_to_workspace(
                    workspace_id=workspace_id,
                    content=content,
                    source_type=DocumentSource.ENTITY_DATA,
                    metadata={
                        "category": "insurance",
                        "insurance_type": insurance.insurance_type.value,
                        "policy_number": insurance.policy_number
                    },
                    doc_id=f"insurance_{i}"
                )

            # 4. Add qualifications to collection
            for i, qual in enumerate(structured_data.team_qualifications):
                content = f"""Person: {qual.person_name}
Qualification: {qual.qualification_name}
Type: {qual.qualification_type.value if qual.qualification_type else 'Not specified'}
Institution: {qual.institution or 'Not specified'}
Date Obtained: {qual.date_obtained or 'Not specified'}
"""
                workspace_manager.add_document_to_workspace(
                    workspace_id=workspace_id,
                    content=content,
                    source_type=DocumentSource.ENTITY_DATA,
                    metadata={
                        "category": "qualification",
                        "person_name": qual.person_name,
                        "qualification": qual.qualification_name
                    },
                    doc_id=f"qualification_{i}"
                )

            # 5. Add work experience to collection
            for i, work in enumerate(structured_data.team_work_experience):
                content = f"""Person: {work.person_name}
Role: {work.role}
Organization: {work.organization}
Duration: {work.start_date} to {work.end_date or 'Present'}
Years: {work.duration_years}
Responsibilities: {work.responsibilities or 'Not specified'}
"""
                workspace_manager.add_document_to_workspace(
                    workspace_id=workspace_id,
                    content=content,
                    source_type=DocumentSource.ENTITY_DATA,
                    metadata={
                        "category": "work_experience",
                        "person_name": work.person_name,
                        "organization": work.organization
                    },
                    doc_id=f"work_exp_{i}"
                )

            # 6. Add projects to collection
            for i, project in enumerate(structured_data.projects):
                content = f"""Project: {project.project_name}
Client: {project.client}
Description: {project.description}
Start Date: {project.start_date or 'Not specified'}
End Date: {project.end_date or 'Not specified'}
Status: {'Ongoing' if project.is_ongoing else 'Completed'}
Value: ${project.project_value or 'Not specified'}
Deliverables: {project.deliverables or 'Not specified'}
Outcomes: {project.outcomes or 'Not specified'}
"""
                workspace_manager.add_document_to_workspace(
                    workspace_id=workspace_id,
                    content=content,
                    source_type=DocumentSource.ENTITY_DATA,
                    metadata={
                        "category": "project",
                        "project_name": project.project_name,
                        "client": project.client
                    },
                    doc_id=f"project_{i}"
                )

            # 7. Add references to collection
            for i, ref in enumerate(structured_data.references):
                content = f"""Reference: {ref.contact_name}
Title: {ref.contact_title or 'Not specified'}
Organization: {ref.organization}
Phone: {ref.phone or 'Not specified'}
Email: {ref.email or 'Not specified'}
Relationship: {ref.relationship or 'Not specified'}
Project Context: {ref.project_context or 'Not specified'}
"""
                workspace_manager.add_document_to_workspace(
                    workspace_id=workspace_id,
                    content=content,
                    source_type=DocumentSource.ENTITY_DATA,
                    metadata={
                        "category": "reference",
                        "contact_name": ref.contact_name,
                        "organization": ref.organization
                    },
                    doc_id=f"reference_{i}"
                )

            # 8. Add capabilities to collection
            for i, cap in enumerate(structured_data.capabilities):
                content = f"""Capability: {cap.capability_name}
Description: {cap.description}
Certification Body: {cap.certification_body or 'Not applicable'}
Certification Number: {cap.certification_number or 'Not applicable'}
Date Obtained: {cap.date_obtained or 'Not specified'}
Expiry Date: {cap.expiry_date or 'No expiry'}
Scope: {cap.scope or 'Not specified'}
"""
                workspace_manager.add_document_to_workspace(
                    workspace_id=workspace_id,
                    content=content,
                    source_type=DocumentSource.ENTITY_DATA,
                    metadata={
                        "category": "capability",
                        "capability_name": cap.capability_name
                    },
                    doc_id=f"capability_{i}"
                )

            logger.info(f"Successfully populated workspace {workspace_id} with {structured_data.summary_stats}")
            return True

        except Exception as e:
            logger.error(f"Failed to populate workspace: {e}", exc_info=True)
            return False

# Project Plan: Idea Generator

## Project Overview

The Idea Generator is a new feature for the Cortex Suite designed to synthesize novel ideas, concepts, and intellectual property from curated knowledge collections. It will guide users through a structured ideation process based on the Double Diamond methodology, leveraging both internal knowledge and optional external research to produce actionable, innovative outputs.

This plan merges the strategic framework of the "Innovation Engine Sprint Plan" with the agile, sprint-based structure of the "Knowledge Synthesizer Plan."

---

## Core Architecture

-   **New UI Page:** `pages/10_Idea_Generator.py`
-   **Backend Module:** `cortex_engine/idea_generator.py`
-   **LLM Integration:** Hybrid model selection (local/cloud) consistent with existing modules.
-   **Data Sources:** User-selected working collections, with optional web research integration.
-   **Methodology:** A four-phase process (Discover, Define, Develop, Deliver) inspired by the Double Diamond design framework.
-   **Output:** Structured reports, concept maps, and actionable next steps.

---

## Sprint 1: Foundation & Discovery Phase

**Goal:** Establish the core infrastructure and implement the initial "Discovery" phase of the ideation process.

**Key Tasks:**
1.  **Create Core Components:**
    -   Create the new UI file: `pages/10_Idea_Generator.py`.
    -   Create the backend module: `cortex_engine/idea_generator.py`.
    -   Add the new page to the main Streamlit navigation.
2.  **Develop Basic UI:**
    -   Implement a collection selector to choose the knowledge base.
    -   Add a model selector for choosing between local and cloud LLMs.
    -   Create a text area for users to input "seed ideas," constraints, and innovation goals.
    -   Add a "Start Discovery" button and a placeholder area for output.
3.  **Implement Discovery Logic:**
    -   In `idea_generator.py`, create a function to retrieve and synthesize content from the selected collection.
    -   Develop a prompt strategy for the "Discovery" phase:
        -   Analyze the collection to identify key themes, entities, and latent connections.
        -   Identify knowledge gaps and potential opportunity areas.
    -   Integrate with the `task_engine` to process the request.
4.  **Web Research Integration:**
    -   Add a UI checkbox: "Allow Internet Research."
    -   If checked, use existing web search tools to augment the discovery analysis.

**Outcome:** A functional page where a user can select a collection, provide initial guidance, and receive a "Discovery" report outlining key themes and opportunities from their knowledge base.

---

## Sprint 2: Define & Develop Phases

**Goal:** Implement the "Define" and "Develop" phases, turning broad opportunities into concrete, well-defined ideas.

**Key Tasks:**
1.  **Implement "Define" Phase Logic:**
    -   Create a prompt chain that takes the "Discovery" output and formulates specific "How Might We..." questions or problem statements.
    -   Allow the user to review, edit, or select the problem statements they wish to pursue.
2.  **Implement "Develop" Phase Logic:**
    -   Based on the selected problem statements, generate a diverse range of potential solutions.
    -   Use a multi-agent approach for ideation:
        -   `agent_solution_brainstormer`: Generates a wide array of ideas.
        -   `agent_analogy_finder`: Looks for cross-domain inspiration within the collection.
        -   `agent_feasibility_analyzer`: Provides a preliminary check on the viability of the generated ideas.
3.  **Enhance UI for Interaction:**
    -   Create a user interface that allows users to review the generated problem statements and select the most promising ones.
    -   Display the developed ideas in a clear, organized manner (e.g., using cards or an expander for each idea).

**Outcome:** The user can now guide the ideation process by refining the problem definition and will receive a set of concrete, creative solutions to that problem, based on their knowledge collection.

---

## Sprint 3: Deliver Phase & Structured Output

**Goal:** Implement the final "Deliver" phase, which involves elaborating on the best ideas and presenting them in a structured, actionable format.

**Key Tasks:**
1.  **Implement "Deliver" Phase Logic:**
    -   Allow the user to select the most promising ideas from the "Develop" phase.
    -   For each selected idea, create a detailed, structured report using a final prompt chain.
2.  **Define Structured Output Format:**
    -   The output should be in Markdown or JSON, with fields such as:
        -   `IdeaTitle`
        -   `CoreConcept`
        -   `SupportingEvidence` (with links to source documents)
        -   `PotentialApplications`
        -   `ImplementationRoadmap`
        -   `RiskAnalysis`
        -   `NextSteps`
3.  **Display Formatted Output:**
    -   Parse the structured output and display it in the Streamlit UI using components like tabs, columns, and expanders for a clean presentation.

**Outcome:** The tool now produces detailed, actionable reports on the most promising ideas, ready for further action, planning, or inclusion in proposals.

---

## Sprint 4: Enhanced Discovery with Filtered Collections

**Goal:** Add intelligent collection filtering and enhanced discovery capabilities to improve idea generation quality.

**Key Tasks:**
1. **Smart Collection Filtering:**
   - Reuse metadata filtering from Knowledge Search (document_type, proposal_outcome, etc.)
   - Add client/organization filtering using the knowledge graph
   - Enable thematic tag-based filtering for focused discovery
   - Implement "Final Reports only" and similar specialized filters

2. **Enhanced Collection Analysis:**
   - Extract entity relationships from filtered documents
   - Analyze thematic tags and document types for better insights
   - Generate collection-specific statistics and metadata summaries
   - Identify knowledge clusters and document relationships

3. **Discovery Intelligence:**
   - Use graph data to identify consultant expertise areas
   - Find client-specific opportunities and patterns
   - Detect cross-project insights and knowledge transfer opportunities
   - Generate more targeted discovery prompts based on filters

4. **UI Enhancements:**
   - Add filter controls to collection selection
   - Show collection statistics and composition
   - Display filter preview before discovery starts
   - Provide filter suggestions based on collection content

**Outcome:** More targeted and intelligent discovery that can focus on specific document types, clients, or themes for higher-quality idea generation.

---

## Sprint 5: Interactive Theme Visualization Network

**Goal:** Create an interactive network visualization of themes and relationships similar to Research Rabbit's author network.

**Key Tasks:**
1. **Network Data Preparation:**
   - Extract theme co-occurrence from discovery results
   - Build theme relationship graphs from document analysis
   - Calculate theme strength and connection weights
   - Prepare node and edge data for visualization

2. **Interactive Network Visualization:**
   - Implement interactive graph using Plotly/NetworkX or similar
   - Create hoverable nodes showing theme details and related documents
   - Enable click-to-expand functionality for theme exploration
   - Add zoom, pan, and layout controls for navigation

3. **Theme Analysis Engine:**
   - Identify central themes vs. peripheral concepts
   - Calculate theme influence and document coverage
   - Detect theme clusters and communities
   - Generate theme evolution analysis across document dates

4. **Visual Interface Integration:**
   - Embed network visualization in Discovery results
   - Link visualization interactions to idea generation
   - Allow theme selection from network for focused Define phase
   - Provide export capabilities for network diagrams

**Outcome:** Rich, interactive visualization that helps users understand theme relationships and select focus areas for ideation visually.

---

## Sprint 6: Visual Image Spark Integration

**Goal:** Enable users to upload hand-drawn sketches or concept images as creative sparks for idea generation.

**Key Tasks:**
1. **Image Upload Interface:**
   - Add image upload component to Discovery phase
   - Support common image formats (PNG, JPG, SVG)
   - Implement drag-and-drop functionality
   - Add image preview and management

2. **Vision-Language Model Integration:**
   - Integrate VLM (LLaVA or similar) for image description
   - Extract conceptual elements, relationships, and themes from images
   - Convert visual concepts to textual idea sparks
   - Handle both hand-drawn sketches and structured diagrams

3. **Visual Context Integration:**
   - Combine image insights with collection analysis
   - Use visual elements as additional seed ideas
   - Generate "How might we..." statements inspired by visual concepts
   - Cross-reference visual themes with knowledge base content

4. **Enhanced Ideation:**
   - Modify agent prompts to incorporate visual inspiration
   - Generate ideas that bridge visual concepts with collection knowledge
   - Create visual-textual idea combinations
   - Provide visual reference in final idea reports

**Outcome:** Creative augmentation of the ideation process using visual inputs, enabling more diverse and innovative idea generation.

---

## Sprint 7: Visualization, Export, & Final Polish

**Goal:** Complete the visualization ecosystem and provide comprehensive export capabilities.

**Key Tasks:**
1. **Idea-to-Source Visualization:**
   - Create concept graphs linking final ideas to source documents
   - Show evidence trails and knowledge provenance
   - Generate mind maps of idea development process
   - Integrate with existing theme network visualization

2. **Comprehensive Export System:**
   - Enhanced PDF exports with embedded visualizations
   - Interactive HTML reports with network diagrams
   - Structured JSON/XML for integration with other tools
   - PowerPoint-ready slide formats for presentations

3. **Advanced UI/UX:**
   - Responsive design for different screen sizes
   - Advanced help system with interactive tutorials
   - Keyboard shortcuts and power-user features
   - Accessibility improvements and screen reader support

4. **System Integration:**
   - Deep integration with Proposal Copilot for idea implementation
   - Knowledge graph updates based on generated ideas
   - Integration with Collection Management for idea-based collections
   - Analytics and usage tracking for continuous improvement

**Outcome:** A fully polished, enterprise-ready Idea Generator with comprehensive visualization, export capabilities, and seamless integration with the Cortex Suite ecosystem.

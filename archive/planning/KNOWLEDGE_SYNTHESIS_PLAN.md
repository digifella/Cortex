# Project Plan: Knowledge Synthesizer

This document outlines the development plan for the new Knowledge Synthesizer feature for the Cortex Suite. The goal is to create a tool that generates novel ideas and intellectual property from curated knowledge collections.

## Core Feature Concept

The Knowledge Synthesizer will be a new page in the Cortex Suite that allows users to:
- Select one or more existing knowledge collections.
- Choose an LLM (local or cloud-based) to power the synthesis.
- Provide "seed ideas" or guiding questions.
- Optionally allow the tool to perform internet research to augment the local knowledge.
- Receive structured, actionable output that can be used for further development.

---

## Sprint 1: Foundation & Core Logic

**Goal:** Establish the basic UI and backend infrastructure for the feature.

**Key Tasks:**
1.  **Create New Page:**
    -   Add a new file: `pages/9_Knowledge_Synthesizer.py`.
    -   Add the new page to the main Streamlit navigation.
2.  **Develop Basic UI:**
    -   Implement a collection selector dropdown to choose the knowledge base.
    -   Add a model selector (similar to AI-Researcher) for choosing between local and cloud LLMs.
    -   Create a text area for users to input "seed ideas."
    -   Add a "Synthesize" button and a placeholder area for the output.
3.  **Backend Engine:**
    -   Create a new module: `cortex_engine/knowledge_synthesizer.py`.
    -   Implement the initial function that retrieves documents from the selected collection.
    -   Develop a simple prompt strategy that combines the seed ideas and collection content.
    -   Integrate with the `task_engine` to process the request with the selected LLM.

**Outcome:** A functional, barebones page where a user can select a collection, enter an idea, and get a single text-block synthesis back from an LLM.

---

## Sprint 2: Advanced Prompting & Guided Ideation

**Goal:** Enhance the quality of synthesis by implementing a more sophisticated, multi-step prompting strategy inspired by the Double Diamond design process.

**Key Tasks:**
1.  **Multi-Step Prompting:**
    -   Design a series of prompts that guide the LLM through a creative process:
        -   **Discover:** First, analyze the collection to identify key themes, entities, and latent connections.
        -   **Define:** Use the identified themes to formulate specific "How Might We..." questions or problem statements.
        -   **Develop:** Generate a range of diverse ideas that address the defined problems, using the collection as evidence.
        -   **Deliver:** Elaborate on the most promising ideas, creating a structured summary.
2.  **UI for Guided Ideation:**
    -   Transform the UI into a multi-step workflow.
    -   Allow the user to review and approve/edit the output of the "Discover" and "Define" stages before proceeding. This gives the user more control over the creative direction.
3.  **Refine Backend Logic:**
    -   Update `knowledge_synthesizer.py` to manage the state of this multi-step process.

**Outcome:** A more powerful synthesizer that guides the user and the LLM through a structured ideation process, leading to higher-quality, more relevant ideas.

---

## Sprint 3: Internet Integration & Structured Output

**Goal:** Add the option to enrich the synthesis with external knowledge and to format the output for clarity and actionability.

**Key Tasks:**
1.  **Enable Internet Research:**
    -   Add a UI checkbox: "Allow Internet Research."
    -   If checked, integrate a web search step (using existing `google_web_search` or `web_fetch` tools) at the "Discover" phase to find relevant external information.
    -   Incorporate the web search results into the context for the subsequent synthesis steps.
2.  **Implement Structured Output:**
    -   Modify the final prompt to request output in a structured format (e.g., Markdown with specific headings or JSON).
    -   The structure should include fields like:
        -   `IdeaTitle`
        -   `CoreConcept` (1-2 sentence summary)
        -   `SupportingEvidence` (linking back to specific documents/nodes in the collection)
        -   `PotentialApplications`
        -   `NextSteps`
3.  **Display Formatted Output:**
    -   Parse the structured output from the LLM.
    -   Display it in the Streamlit UI using expanders, columns, and other components for a clean, readable presentation.

**Outcome:** The synthesizer can now produce well-structured, detailed reports on new ideas, optionally enriched with current information from the web.

---

## Sprint 4: Visualization & Final Polish

**Goal:** Integrate visual tools to help users understand the new ideas and their connections, and finalize the feature for release.

**Key Tasks:**
1.  **Generate Idea-Graphs:**
    -   Create a function to generate a mind map or concept graph of the synthesized ideas.
    -   The graph should show the new ideas as central nodes, linked to the source documents or entities from the collection that inspired them.
    -   Integrate this visualization into the results page (using Graphviz).
2.  **Save & Export:**
    -   Add functionality to save the synthesis results (both the text report and the graph) to a file or a new "Synthesized Ideas" collection.
3.  **UI/UX Refinement:**
    -   Add comprehensive help text and tooltips.
    -   Ensure the workflow is intuitive and responsive.
    -   Conduct a final review of all UI elements and user interactions.
4.  **Logging & Error Handling:**
    -   Implement robust logging for the entire synthesis process to aid in debugging and future improvements.
    -   Add user-friendly error handling for API failures, empty collections, etc.

**Outcome:** A polished, powerful, and fully integrated Knowledge Synthesizer that provides textual and visual outputs, ready for users.

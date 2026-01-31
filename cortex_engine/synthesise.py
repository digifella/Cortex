# ## File: cortex_engine/synthesise.py
# Version: 5.0.0 (Regression Fix & Citation Enhancement)
# Date: 2025-07-22
# Purpose: Backend engine for the multi-agent AI Research Assistant.
#          - CRITICAL FIX (v5.0.0): Reverted the foundational query agent and
#            source fetching logic to the more robust implementation from v3.1.0.
#            This resolves a critical regression where no foundational sources
#            were being found.
#          - FEATURE (v5.0.0): Significantly improved the prompt for the
#            `agent_deep_researcher` to explicitly instruct the LLM on how
#            to create and use a numbered reference list, fixing the bug where
#            citations were missing in the final report.

import os
import graphviz
import time
import json
import re
import logging
import requests
import shutil
import subprocess
import random
from pathlib import Path
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional, Type, Tuple, Union

from llama_index.llms.ollama import Ollama as LlamaOllama
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.llms import LLM

from cortex_engine.utils import convert_windows_to_wsl_path

logger = logging.getLogger(__name__)

# --- Configuration & Schemas ---
OUTPUT_DIR = Path(__file__).parent.parent / "external_research"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load .env from project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

# RESTORED from v3.1.0 to fix regression
class FoundationalQueries(BaseModel):
    """Queries for foundational, authoritative papers."""
    highly_cited_query: str = Field(..., description="A single, precise query for finding the most cited papers on the topic.")
    review_queries: List[str] = Field(..., description="A list of queries targeting systematic reviews, literature reviews, or meta-analyses.")

class ExploratoryQueries(BaseModel):
    """Queries for broad, exploratory research."""
    scholar_queries: List[str] = Field(..., description="A list of diverse, general-purpose academic search queries.")
    youtube_queries: List[str] = Field(..., description="A list of accessible search queries for YouTube videos.")


class ThemeList(BaseModel):
    """Data model for a list of themes."""
    themes: List[str] = Field(..., description="A list of distinct, high-level thematic categories.")

# Use centralized path conversion from cortex_engine.utils
_convert_windows_to_wsl_path = convert_windows_to_wsl_path

def _normalize_json_keys(data: Dict[str, Any], model: Type[BaseModel]) -> Dict[str, Any]:
    normalized_data = {}
    model_fields = list(model.model_fields.keys())
    field_map = {re.sub(r'[\s-]', '_', f).lower(): f for f in model_fields}
    for key, value in data.items():
        normalized_key = re.sub(r'[\s-]', '_', key).lower()
        if normalized_key in field_map:
            correct_key = field_map[normalized_key]
            normalized_data[correct_key] = value
    return normalized_data

# --- Setup & LLM Completion ---

class GeminiRest:
    """A robust wrapper to call the Gemini API via a direct REST request."""
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:generateContent"

    def complete(self, prompt: str, generation_config: dict = None) -> str:
        headers = {'Content-Type': 'application/json'}
        params = {'key': self.api_key}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(self.api_url, headers=headers, params=params, json=payload, timeout=180) # Increased timeout
            response.raise_for_status()
            response_json = response.json()
            if 'candidates' in response_json and response_json['candidates']:
                return response_json['candidates'][0]['content']['parts'][0]['text']
            else:
                return f'{{"error": "API returned no candidates. Prompt may have been blocked.", "response": {response.text}}}'
        except requests.exceptions.RequestException as e:
            error_details = str(e)
            print(f"Error during Gemini REST API call: {error_details}")
            return f'{{"error": "Failed to connect to Gemini API.", "details": "{error_details}"}}'

# Dynamic provider selection: prefer session state, fall back to env var
def get_research_llm_provider():
    """Get LLM provider for research, respecting UI choice."""
    # Try to get from Streamlit session state (user choice)
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'research_provider'):
            return st.session_state.research_provider.lower()
    except ImportError:
        pass  # Not in Streamlit context
    
    # Fall back to environment variable
    return os.getenv("RESEARCH_LLM_PROVIDER", "gemini").lower()

LLM_PROVIDER = get_research_llm_provider()
GRAPHVIZ_DOT_EXECUTABLE = os.getenv("GRAPHVIZ_DOT_EXECUTABLE")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MAX_PAPERS_PER_QUERY, MAX_VIDEOS_PER_QUERY = 3, 1

def get_llm(status_callback=print):
    """Get or create LLM instance. Uses Streamlit session state when available for thread safety."""
    provider = get_research_llm_provider()

    # Try session-state-based caching first (thread-safe for Streamlit)
    _llm_instance = None
    _cached_provider = None
    try:
        import streamlit as st
        _llm_instance = st.session_state.get("_research_llm_instance")
        _cached_provider = st.session_state.get("_research_llm_provider")
    except (ImportError, RuntimeError):
        pass  # Not in Streamlit context

    # Return cached if provider hasn't changed
    if _llm_instance is not None and _cached_provider == provider:
        return _llm_instance

    status_callback(f"🚀 Initializing Research LLM provider: {provider.upper()}")
    llm_instance = None
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise ValueError("GEMINI_API_KEY not found in .env file.")
        llm_instance = GeminiRest(api_key=api_key)
        status_callback(f"✅ Gemini LLM (gemini-1.5-flash via Direct REST API v1) is ready.")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY not found in .env file.")
        llm_instance = LlamaOpenAI(model="gpt-4o-mini", api_key=api_key)
        status_callback(f"✅ OpenAI LLM (gpt-4o-mini) is ready.")
    elif provider == "ollama":
        model_name = os.getenv("OLLAMA_RESEARCH_MODEL", "mistral:latest")
        llm_instance = LlamaOllama(model=model_name, request_timeout=120.0)
        status_callback(f"✅ Ollama Research LLM ({model_name}) is ready for local research.")
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}.")

    # Cache in session state if available
    try:
        import streamlit as st
        st.session_state["_research_llm_instance"] = llm_instance
        st.session_state["_research_llm_provider"] = provider
    except (ImportError, RuntimeError):
        pass

    return llm_instance

def llm_completion(prompt: str, status_callback=print, is_json=False) -> str:
    provider = get_research_llm_provider()
    status_callback(f"🧠 Generating text with {provider.upper()}...")
    try:
        llm_instance = get_llm(status_callback)
        response = llm_instance.complete(prompt)
        response_text = str(response).strip()
        if response_text.startswith('{"error"'):
             status_callback(f"❌ LLM provider error: Received an error object from the API wrapper.")
             status_callback(f"   RAW ERROR: {response_text}")
             return ""
        return response_text
    except Exception as e:
        logger.exception("Unhandled exception in llm_completion")
        status_callback(f"\n❌ Unhandled exception in llm_completion: {e}")
        return ""

# --- Agent Definitions ---
# RESTORED from v3.1.0 to fix regression
def agent_foundational_query_crafter(topic: str, status_callback=print) -> dict:
    status_callback(f"🤖 Agent Foundational Query Crafter: Seeking authoritative sources for '{topic}'...")
    json_prompt = f"""You are an expert academic researcher. For the given topic, create a JSON object with two keys:
1.  `highly_cited_query`: A single, precise query string to find the most influential, highly-cited papers.
2.  `review_queries`: A list of 2-3 query strings to find "systematic review", "literature review", or "meta-analysis" papers.
Topic: "{topic}"
Return ONLY a valid JSON object. Do not add any other text.
Example Format:
{{
  "highly_cited_query": "highly cited papers on AI in healthcare",
  "review_queries": ["systematic review of AI in medicine", "literature review of machine learning in diagnostics"]
}}"""
    try:
        llm_output_str = llm_completion(json_prompt, status_callback=status_callback, is_json=True)
        cleaned_json_str = re.sub(r'```json\s*(.*)\s*```', r'\1', llm_output_str, flags=re.DOTALL).strip()
        raw_data = json.loads(cleaned_json_str)
        normalized_data = _normalize_json_keys(raw_data, FoundationalQueries)
        query_object = FoundationalQueries.model_validate(normalized_data)
        queries = query_object.model_dump()
        status_callback(f"✅ Crafted Foundational Queries (validated): {queries}")
        return queries
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logger.warning("Foundational query generation failed: %s", e)
        status_callback(f"❌ Critical failure during foundational query generation. Error: {e}")
        return {"highly_cited_query": f"highly cited papers on {topic}", "review_queries": [f"review of {topic}"]}

def agent_exploratory_query_crafter(topic: str, status_callback=print) -> dict:
    status_callback(f"🤖 Agent Exploratory Query Crafter: Broadening search for '{topic}'...")
    json_prompt = f'You are an expert query generator. Based on the following topic, create a JSON object containing two lists of search queries: one for general academic papers and one for YouTube videos.\nTopic: "{topic}"\n\nReturn ONLY a valid JSON object in the following format. Do not add any other text.\nExample Format:\n{{\n  "scholar_queries": ["academic query 1", "academic query 2"],\n  "youtube_queries": ["video query a", "video query b"]\n}}'
    try:
        llm_output_str = llm_completion(json_prompt, status_callback=status_callback, is_json=True)
        cleaned_json_str = re.sub(r'```json\s*(.*)\s*```', r'\1', llm_output_str, flags=re.DOTALL).strip()
        raw_data = json.loads(cleaned_json_str)
        normalized_data = _normalize_json_keys(raw_data, ExploratoryQueries)
        query_object = ExploratoryQueries.model_validate(normalized_data)
        queries = query_object.model_dump()
        status_callback(f"✅ Crafted Exploratory Queries (validated): {queries}")
        return queries
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logger.warning("Exploratory query generation failed: %s", e)
        status_callback(f"❌ Critical failure during exploratory query generation. Error: {e}")
        return {"scholar_queries": [topic], "youtube_queries": [topic]}


def agent_thematic_analyser(context: str, status_callback=print, existing_themes: List[str] = None) -> List[str]:
    status_callback("🔬 Agent Thematic Analyser: Identifying and structuring core themes...")
    llm_output_str = ""
    default_error_response = ["Could not determine themes from the provided text."]
    if existing_themes:
        status_callback("  -> Finding net-new themes to ADD to the existing list...")
        existing_themes_str = "\n".join(f"- {theme}" for theme in existing_themes)
        additive_prompt = f"""You are a research analyst. Your task is to identify ONLY the new, high-level themes from the 'CONTEXT TO ANALYZE' that are NOT already present in the 'EXISTING THEMES' list.
**EXISTING THEMES (to avoid duplicating):**
{existing_themes_str}
**CONTEXT TO ANALYZE:**
---
{context}
---
**YOUR INSTRUCTIONS:**
1.  Analyze the 'CONTEXT TO ANALYZE' for its main themes.
2.  Compare these themes against the 'EXISTING THEMES' list.
3.  Return a JSON object containing a list of **ONLY THE NEW THEMES** that you found.
4.  If no new themes are found, return a JSON object with an empty list: {{"themes": []}}.
5.  Do NOT include the existing themes in your response.
Return ONLY a valid JSON object with a "themes" key. Do not add any other text.
"""
        prompt = additive_prompt
    else:
        status_callback("  -> Identifying themes from scratch...")
        prompt = f"""Analyze the provided research context to identify 3-5 distinct, high-level themes. Each theme must be a high-level concept, not a specific detail. Themes must not overlap.
If you analyze the context and cannot determine at least 3 clear, distinct themes, you MUST return a JSON object with a single theme that says: "Could not determine themes from the provided text."
Context to analyze:
---
{context}
---
Return ONLY a valid JSON object with a "themes" key. Do not add any other text.
Example Format: {{"themes": ["First theme", "Second theme", "Third theme"]}}
"""
    try:
        llm_output_str = llm_completion(prompt, status_callback=status_callback, is_json=True)
        if not llm_output_str or not llm_output_str.strip():
            raise ValueError("LLM returned an empty response.")
        cleaned_json_str = re.sub(r'```json\s*(.*)\s*```', r'\1', llm_output_str, flags=re.DOTALL).strip()
        raw_data = json.loads(cleaned_json_str)
        if not isinstance(raw_data, dict) or not raw_data:
            return [] if existing_themes else default_error_response
        normalized_data = _normalize_json_keys(raw_data, ThemeList)
        if 'themes' not in normalized_data or not normalized_data['themes']:
            return [] if existing_themes else default_error_response
        theme_object = ThemeList.model_validate(normalized_data)
        themes_list = theme_object.themes
        status_callback(f"✅ Identified and validated themes: {themes_list}")
        return themes_list
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logger.warning("Theme generation failed: %s | Raw output: %s", e, llm_output_str[:200])
        status_callback(f"❌ Critical failure during theme generation. Error: {e}\nRAW LLM OUTPUT:\n{llm_output_str}")
        return [] if existing_themes else default_error_response

# --- Data Retrieval Agents ---
def agent_paper_retriever(query: str, sort_by: str = None, status_callback=print) -> Tuple[bool, Union[List, str]]:
    search_type = f"'{sort_by}'" if sort_by else "standard"
    status_callback(f"📚 Agent Paper Retriever ({search_type} search): Searching for '{query}'...")
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {'query': query, 'limit': MAX_PAPERS_PER_QUERY, 'fields': 'title,url,abstract,citationCount'}
    if sort_by:
        params['sort'] = sort_by

    MAX_RETRIES, INITIAL_BACKOFF = 5, 2
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                backoff_time = INITIAL_BACKOFF * (2 ** (attempt-1)) + random.uniform(0, 1)
                status_callback(f"   - Rate limit hit for '{query}'. Retrying in {backoff_time:.2f} seconds...")
                time.sleep(backoff_time)
            else:
                 time.sleep(1.5) # Initial delay

            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()

            data, papers = response.json(), []
            for item in data.get('data', []):
                papers.append({
                    "source_type": "paper", "title": item.get('title'),
                    "text": item.get('abstract') or 'No abstract available.',
                    "url": item.get('url'), "citations": item.get('citationCount', 0)
                })
            if not papers: status_callback(f"🟡 Paper search for '{query}' was successful but returned no results.")
            return True, papers

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < MAX_RETRIES - 1:
                continue
            else:
                error_message = f"Paper search for '{query}' failed. Reason: {e}"
                status_callback(f"❌ {error_message}")
                return False, error_message
        except requests.exceptions.RequestException as e:
            error_message = f"Paper search for '{query}' failed. Reason: {e}"
            status_callback(f"❌ {error_message}")
            return False, error_message

    final_error_message = f"Paper search for '{query}' failed after {MAX_RETRIES} retries."
    status_callback(f"❌ {final_error_message}")
    return False, final_error_message


def agent_youtube_extractor(query: str, status_callback=print) -> Tuple[bool, Union[List, str]]:
    status_callback(f"📺 Agent YouTube Extractor: Searching for '{query}'...")
    if not YOUTUBE_API_KEY:
        error_message = "YouTube search failed: YOUTUBE_API_KEY is not configured in your .env file."
        status_callback(f"❌ {error_message}")
        return False, error_message
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        search_response = youtube.search().list(q=query, part='id,snippet', maxResults=MAX_VIDEOS_PER_QUERY, type='video').execute()
        videos, video_ids_processed = [], set()
        for item in search_response.get('items', []):
            video_id = item['id']['videoId']
            if video_id in video_ids_processed: continue
            video_ids_processed.add(video_id)
            title, description = item['snippet']['title'], item['snippet']['description']
            url, text_content = f"https://www.youtube.com/watch?v={video_id}", ""
            status_callback(f"  - Checking video: '{title[:60]}...'")
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                text_content = " ".join([d['text'] for d in transcript_list])
                status_callback(f"    ✅ Found transcript.")
            except Exception as te:
                text_content = f"Title: {title}\nDescription: {description}"
                status_callback(f"    - No transcript available (Reason: {te}). Falling back to metadata.")
            videos.append({"source_type": "youtube", "title": title, "text": text_content, "url": url})
        if not videos:
            status_callback(f"🟡 YouTube search for '{query}' was successful but returned no videos.")
        return True, videos
    except Exception as e:
        error_message = f"YouTube search for '{query}' failed. Reason: {e}"
        status_callback(f"❌ {error_message}")
        return False, error_message

# --- Synthesiser, Visualiser & Deep Researcher ---
def agent_synthesiser(context: str, themes: List[str], topic: str, sources: List[Dict[str, Any]], status_callback=print):
    status_callback("📝 Agent Synthesiser: Generating theme-driven outputs...")
    themes_for_note = "\n".join([f"{i+1}. {theme}" for i, theme in enumerate(themes)])
    note_prompt = f'You are a research analyst. Generate a "Discovery Note" in Markdown for the topic "{topic}".\nStructure your analysis EXPLICITLY around the following numbered key themes:\n{themes_for_note}\nIMPORTANT INSTRUCTIONS:\n- For each theme, synthesize insights from the provided sources.\n- When you cite a source, you MUST include a clickable Markdown link to its URL.\n- Conclude with a section for "Open Questions".\nContext:\n{context}'
    discovery_note = llm_completion(note_prompt, status_callback=status_callback)
    themes_for_map = ", ".join(f'"{t}"' for t in themes)
    mindmap_prompt = f"""You are a mind map generator. Your task is to create a hierarchical outline as a plain text indented list.
CRITICAL INSTRUCTIONS:
1.  The single root node MUST be the main research topic.
2.  The second-level nodes MUST be the provided key themes.
3.  The third-level nodes should be 3-5 key concepts derived from the context relevant to each theme.
4.  Do NOT include source titles or URLs. Keep all nodes concise.
MIND MAP DETAILS:
- **Main Topic:** "{topic}"
- **Key Themes:** [{themes_for_map}]
CONTEXT FOR SUB-TOPICS:
---
{context}
---
YOUR MIND MAP OUTLINE (indented list only):
"""
    mindmap_structure = llm_completion(mindmap_prompt, status_callback=status_callback)
    if sources:
        references_md = "\n\n---\n\n## Curated Sources\n\n"
        sources.sort(key=lambda s: (s.get('is_foundational', False), s.get('source_type', '')), reverse=True)
        for source in sources:
            cite_count = f" (Citations: {source.get('citations', 0)})" if source.get('source_type') == 'paper' else ''
            references_md += f"*   **[{source.get('source_type', 'N/A').upper()}] {source.get('title', 'No Title')}**{cite_count}\n    *   Link: <{source.get('url', '#')}>\n"
        discovery_note += references_md
    if mindmap_structure:
        mindmap_md = f"\n\n---\n\n## Mind Map Outline\n\n```\n{mindmap_structure}\n```\n"
        discovery_note += mindmap_md
    return discovery_note, mindmap_structure

def agent_visualiser(mindmap_structure: str, topic: str, output_folder: Path, status_callback=print):
    status_callback("🎨 Agent Visualiser: Creating Mind Map image...")
    dot_path = GRAPHVIZ_DOT_EXECUTABLE
    if not dot_path or not shutil.which(dot_path):
        status_callback(f"⚠️ 'dot' executable not found in .env at '{dot_path}'. Searching system PATH...")
        dot_path = shutil.which("dot")
    if not dot_path:
        error_message = "❌ CRITICAL: Mind map generation failed. The 'dot' command from Graphviz could not be found."
        status_callback(error_message); print(error_message)
        return
    # ... (rest of the function is unchanged)
    def get_indent(line: str): return len(line) - len(line.lstrip(' \t'))
    def sanitize_node_name(name: str): return re.sub(r'^\d+[\.\)]\s*', '', name.strip().lstrip('*-• ')).strip()
    lines = [line for line in mindmap_structure.strip().split('\n') if line.strip()]
    if not lines:
        status_callback("⚠️ Mind map structure was empty. Skipping image generation."); return
    try:
        dot = graphviz.Digraph(comment=topic, engine='dot', graph_attr={'rankdir': 'LR', 'splines': 'ortho', 'nodesep': '0.4'}, node_attr={'shape': 'box', 'style': 'rounded,filled', 'fontname': 'Helvetica'}, edge_attr={'arrowsize': '0.7'})
        parent_stack, indents = [], {get_indent(line) for line in lines}
        sorted_indents, indent_levels = sorted(list(indents)), {}
        for i, indent in enumerate(sorted_indents): indent_levels[indent] = i
        for i, line in enumerate(lines):
            node_name = sanitize_node_name(line)
            if not node_name: continue
            level = indent_levels.get(get_indent(line), 0)
            node_id = f"node_{i}"
            dot.node(node_id, label=node_name, fillcolor='skyblue' if level == 0 else ('lightgray' if level == 1 else 'white'))
            while len(parent_stack) > level: parent_stack.pop()
            if parent_stack: dot.edge(parent_stack[-1], node_id)
            parent_stack.append(node_id)
        dot_source_path = output_folder / "mindmap.dot"
        png_output_path = output_folder / "mindmap.png"
        dot.save(str(dot_source_path))
        command = [dot_path, "-Tpng", str(dot_source_path), "-o", str(png_output_path)]
        status_callback(f"  -> Executing command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            status_callback(f"✅ Mind Map saved to {png_output_path}")
        else:
            error_details = result.stderr or result.stdout or "No error output from Graphviz."
            status_callback(f"❌ Error rendering mind map with Graphviz: {error_details.strip()}")
        if dot_source_path.exists():
            os.remove(dot_source_path)
    except Exception as e:
        status_callback(f"❌ An unexpected Python exception occurred during mind map generation: {e}")

def agent_deep_researcher(topic: str, initial_synthesis_note: str, status_callback=print) -> str:
    """
    Takes the initial synthesis and performs a deeper, more comprehensive
    research pass to generate a final, detailed report.
    """
    status_callback("🧐 Agent Deep Researcher: Initiating deep-dive analysis...")
    # ENHANCED PROMPT to fix citation issue
    prompt = f"""
You are a world-class research analyst producing a comprehensive report on the topic of **"{topic}"**.
You have been provided with an initial "Discovery Note" which contains a preliminary analysis and a list of sources. Your job is to expand this into a final, detailed report with proper citations.

**INITIAL CONTEXT (Discovery Note):**
---
{initial_synthesis_note}
---

**YOUR TASK (Follow these steps precisely):**

1.  **CREATE REFERENCE LIST:** First, at the very end of your response, create a markdown section titled `## Consolidated Reference List`. Review the "Curated Sources" section from the context above and create a **numbered list** of all the sources. For example:
    ```
    ## Consolidated Reference List
    1. [PAPER] Title of Paper A. URL: <http://...>
    2. [VIDEO] Title of Video B. URL: <http://...>
    ```

2.  **WRITE THE REPORT:** Next, generate the main report in Markdown. It must have sections for an Executive Summary, Introduction, a detailed Thematic Deep-Dive for each theme identified in the context, Practical Applications, Challenges, and a Future Outlook.

3.  **CITE YOUR SOURCES:** This is critical. As you write the report, you **MUST** provide inline citations for your claims by referencing the number from the "Consolidated Reference List" you created in step 1. Use the format `[1]`, `[2]`, etc. Every major claim or piece of data must have a citation. If you cannot find a source for a claim, do not make that claim.

**FINAL OUTPUT:**
Produce a single, cohesive Markdown document that contains the full report with inline citations, followed by the final numbered reference list.
"""
    status_callback("  -> Sending comprehensive prompt to LLM for deep synthesis...")
    final_report = llm_completion(prompt, status_callback=status_callback)
    status_callback("✅ Deep Research Agent has completed the report.")
    return final_report


def build_context_from_sources(topic: str, all_sources: list) -> str:
    context = f"Topic: {topic}\n\n--- CONSOLIDATED SOURCES ---\n\n"
    for s in all_sources:
        context += f"Source Type: {s['source_type'].upper()}\nTitle: {s['title']}\nURL: {s.get('url', 'N/A')}\nContent: {s['text'][:1500].strip()}...\n\n---\n\n"
    return context

# --- Workflow Functions ---
# RESTORED from v3.1.0 to fix regression
def step1_fetch_foundational_sources(queries: Dict[str, Any], status_callback=print) -> dict:
    all_sources, failures = [], []
    if queries.get("highly_cited_query"):
        success, result = agent_paper_retriever(queries["highly_cited_query"], sort_by="citationCount", status_callback=status_callback)
        if success:
            all_sources.extend(result)
        else:
            failures.append(result)
    if queries.get("review_queries"):
        for query in queries["review_queries"]:
            success, result = agent_paper_retriever(query, sort_by="relevance", status_callback=status_callback)
            if success:
                all_sources.extend(result)
            else:
                failures.append(result)

    unique_sources = list({s['url']: s for s in all_sources if s.get('url')}.values())
    for source in unique_sources:
        source['is_foundational'] = True
    return {"sources": unique_sources, "failures": failures}

def step2_fetch_exploratory_sources(queries: Dict[str, List[str]], status_callback=print) -> dict:
    all_sources, failures = [], []
    if queries.get("scholar_queries"):
        for query in queries["scholar_queries"]:
            success, result = agent_paper_retriever(query, status_callback=status_callback)
            if success: all_sources.extend(result)
            else: failures.append(result)
    if queries.get("youtube_queries"):
        for query in queries["youtube_queries"]:
            success, result = agent_youtube_extractor(query, status_callback=status_callback)
            if success: all_sources.extend(result)
            else: failures.append(result)
    unique_sources = list({s['url']: s for s in all_sources if s.get('url')}.values())
    return {"sources": unique_sources, "failures": failures}

def step3_go_deeper(theme_query: str, status_callback=print):
    new_sources = []
    success_papers, papers_or_err = agent_paper_retriever(theme_query, status_callback=status_callback)
    if success_papers: new_sources.extend(papers_or_err)
    success_vids, vids_or_err = agent_youtube_extractor(theme_query, status_callback=status_callback)
    if success_vids: new_sources.extend(vids_or_err)
    return new_sources

def step4_run_synthesis(sources: List[Dict[str, Any]], themes: List[str], topic: str, status_callback=print):
    topic_folder_name = re.sub(r'[\\/*?:"<>|]', "", topic).replace(" ", "_")[:50]
    final_output_dir = OUTPUT_DIR / topic_folder_name
    final_output_dir.mkdir(exist_ok=True)
    if not sources or not themes:
        status_callback("❌ No sources or themes provided to synthesize. Aborting.")
        return None, None
    context = build_context_from_sources(topic, sources)
    discovery_note_md, mindmap_txt = agent_synthesiser(context, themes, topic, sources, status_callback=status_callback)
    if not discovery_note_md or not mindmap_txt:
        status_callback("❌ CRITICAL: Synthesis agent failed."); return None, None
    note_path = final_output_dir / "discovery_note.md"
    map_path = final_output_dir / "mindmap.png"
    try:
        with open(note_path, "w", encoding="utf-8") as f: f.write(discovery_note_md)
        status_callback(f"✅ Discovery Note saved to {note_path}")
    except OSError as e:
        logger.error("Failed to write discovery note: %s", e)
        status_callback(f"❌ Failed to save discovery note: {e}")
        return None, None
    agent_visualiser(mindmap_txt, topic, final_output_dir, status_callback=status_callback)
    if map_path.exists():
        return str(note_path), str(map_path)
    else:
        status_callback("⚠️ Synthesis finished, but the mind map image could not be created.")
        return str(note_path), None

def step5_run_deep_synthesis(topic: str, initial_synthesis_note: str, status_callback=print) -> str:
    """
    Orchestrates the final deep research step.
    """
    topic_folder_name = re.sub(r'[\\/*?:"<>|]', "", topic).replace(" ", "_")[:50]
    final_output_dir = OUTPUT_DIR / topic_folder_name
    final_output_dir.mkdir(exist_ok=True)

    if not initial_synthesis_note:
        status_callback("❌ Cannot run deep synthesis without an initial note. Aborting.")
        return None

    deep_report_md = agent_deep_researcher(topic, initial_synthesis_note, status_callback=status_callback)
    if not deep_report_md:
        status_callback("❌ CRITICAL: Deep research agent failed to produce a report.")
        return None

    report_path = final_output_dir / f"deep_research_report_{topic_folder_name}.md"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(deep_report_md)
        status_callback(f"✅ Deep Research Report saved to {report_path}")
    except OSError as e:
        logger.error("Failed to write deep research report: %s", e)
        status_callback(f"❌ Failed to save deep research report: {e}")
        return None
    return str(report_path)


def save_outputs_to_custom_dir(source_note_path: str, source_map_path: str, custom_dest_dir_str: str, status_callback=print):
    try:
        if not custom_dest_dir_str:
            status_callback("❌ Error: Destination directory cannot be empty.")
            return False, "Destination directory was not provided."
        if os.path.exists('/.dockerenv'):
            dest_path = Path(custom_dest_dir_str)
        else:
            dest_path = Path(convert_windows_to_wsl_path(custom_dest_dir_str))
        dest_path.mkdir(parents=True, exist_ok=True)
        status_callback(f"  -> Ensured destination directory exists: {dest_path}")
        if source_note_path and Path(source_note_path).exists():
            shutil.copy(source_note_path, dest_path)
            status_callback(f"  -> ✅ Copied discovery note to {dest_path}")
        else:
             status_callback(f"  -> ⚠️ Could not find source discovery note at {source_note_path} to copy.")
        if source_map_path and Path(source_map_path).exists():
            shutil.copy(source_map_path, dest_path)
            status_callback(f"  -> ✅ Copied mind map to {dest_path}")
        else:
            status_callback(f"  -> ⚠️ Could not find source mind map at {source_map_path} to copy.")
        return True, f"Successfully saved a copy to {dest_path}"
    except Exception as e:
        error_msg = f"❌ Failed to save to custom directory: {e}"
        status_callback(error_msg)
        return False, error_msg
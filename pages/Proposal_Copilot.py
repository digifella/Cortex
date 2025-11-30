# ## File: Proposal_Copilot.py
# Version: v4.10.3
# Date: 2025-07-23
# Purpose: Core UI for drafting proposals.
#          - REFACTOR (v28.0.0): Updated to use centralized utilities for path handling,
#            logging, and error handling. Removed code duplication.

import streamlit as st
import os
import io
import docx
import chromadb
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from cortex_engine.embedding_adapters import EmbeddingServiceAdapter
from llama_index.llms.ollama import Ollama
from chromadb.config import Settings as ChromaSettings

# Import centralized utilities
from cortex_engine.utils import convert_to_docker_mount_path, get_logger
from cortex_engine.config import EMBED_MODEL, LLM_MODEL, COLLECTION_NAME
from cortex_engine.instruction_parser import CortexInstruction, parse_template_for_instructions
from cortex_engine.task_engine import TaskExecutionEngine
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.proposal_manager import ProposalManager
from cortex_engine.config_manager import ConfigManager

# Set up logging
logger = get_logger(__name__)

# --- CONFIGURATION ---
# Use cross-platform default path detection
from cortex_engine.utils.default_paths import get_default_ai_database_path
DEFAULT_DB_PATH = get_default_ai_database_path()
RESPONDING_ORGS = ["Deakin", "Escient", "Longboardfella", "Consortium"]
GENERATIVE_TASKS = ["GENERATE_FROM_KB", "GENERATE_RESOURCES", "GENERATE_FROM_KB_AND_PROPOSAL", "GENERATE_FROM_PROPOSAL_ONLY"]
RETRIEVAL_TASKS = ["RETRIEVE_FROM_KB"]
REFINEMENT_ONLY_TASKS = ["PROMPT_HUMAN"]

# --- Helper Functions ---
# Path handling now centralized in utilities

# --- Core Functions & State Management ---

def initialize_session_state():
    """Initializes session state with default values if keys are missing."""
    defaults = {
        "responding_org": None,
        "knowledge_sources": ["Main Cortex Knowledge Base"],
        "parsed_instructions": [],
        "section_content": {},
        "doc_template": None,
        "generated_doc_bytes": None,
        "original_filename": "",
        "loaded_proposal_id": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource
def load_system(_db_path):
    with st.spinner(f"Connecting to Cortex KB at '{_db_path}'..."):
        chroma_db_path = os.path.join(_db_path, "knowledge_hub_db")
        if not os.path.isdir(chroma_db_path): # Use isdir for better checking
            st.error(f"ChromaDB path not found. Please run an ingestion process. Path checked: '{chroma_db_path}'"); st.stop()
        try:
            # Check if Ollama is available for proposal copilot
            from cortex_engine.utils.ollama_utils import check_ollama_service, format_ollama_error_for_user

            is_running, error_msg, resolved_url = check_ollama_service()
            if not is_running:
                st.error("🚫 **Proposal Copilot Unavailable**")
                st.markdown(format_ollama_error_for_user("Proposal Copilot", error_msg))
                st.stop()
            
            Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0)
            # Use centralized embeddings via adapter for consistency with ingest/search
            try:
                Settings.embed_model = EmbeddingServiceAdapter(model_name=EMBED_MODEL)
            except Exception:
                Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device="cuda")
            db_settings = ChromaSettings(anonymized_telemetry=False)
            db = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
            chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
            index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=chroma_collection))
            collection_manager = WorkingCollectionManager()
        except Exception as e: st.error(f"Failed to initialize system: {e}"); st.stop()
    return index, collection_manager

def load_proposal_into_session(proposal_id, collection_manager):
    """Loads a specific proposal's data into the session state."""
    prop_mgr = ProposalManager()
    data = prop_mgr.load_proposal(proposal_id)
    if data:
        session_state_data = data['state'].get('session_state', {})
        valid_collections = collection_manager.get_collection_names()
        saved_sources = session_state_data.get('knowledge_sources', ["Main Cortex Knowledge Base"])
        healed_sources = [src for src in saved_sources if src in valid_collections or src == "Main Cortex Knowledge Base"]
        st.session_state.knowledge_sources = healed_sources if healed_sources else ["Main Cortex Knowledge Base"]

        # Load all other JSON-safe data into the session
        st.session_state.responding_org = session_state_data.get('responding_org')
        st.session_state.section_content = session_state_data.get('section_content', {})
        st.session_state.original_filename = session_state_data.get('original_filename', "")

        # Handle template bytes and ALWAYS re-parse instructions
        if 'template_bytes' in data and data['template_bytes']:
            st.session_state.doc_template = docx.Document(io.BytesIO(data['template_bytes']))
            st.session_state.parsed_instructions = parse_template_for_instructions(st.session_state.doc_template)
        else:
            # If no template is loaded, ensure instructions and template object are cleared
            st.session_state.doc_template = None
            st.session_state.parsed_instructions = []

        if 'generated_doc_bytes' in data:
            st.session_state.generated_doc_bytes = data['generated_doc_bytes']

        st.session_state['proposal_name'] = data['state'].get('name', 'Untitled')
        st.toast(f"Loaded proposal: {st.session_state['proposal_name']}")
    else:
        st.error("Failed to load proposal."); st.switch_page("pages/6_Proposal_Step_2_Make.py")

# --- Action Handlers ---

def handle_ai_action(engine: TaskExecutionEngine, instruction: CortexInstruction, content_key: str):
    logger.info(f"🔍 DEBUG: handle_ai_action called for task_type='{instruction.task_type}', content_key='{content_key}'")
    
    creativity = st.session_state.section_content.get(content_key, {}).get('creativity', 'green')
    raw_text = st.session_state.section_content.get(content_key, {}).get('text', '')
    has_content = raw_text.strip()
    generated_text = ""
    
    # Get user hint/guidance
    hint_key = f"hint_{content_key}"
    user_hint = st.session_state.get(hint_key, "").strip()
    
    logger.info(f"🔍 DEBUG: creativity='{creativity}', has_content={bool(has_content)}, raw_text_len={len(raw_text)}")
    logger.info(f"🔍 DEBUG: user_hint='{user_hint}'")
    logger.info(f"🔍 DEBUG: knowledge_sources={st.session_state.knowledge_sources}")
    logger.info(f"🔍 DEBUG: section_content keys: {list(st.session_state.section_content.keys())}")

    # Store the current text for undo functionality
    undo_key = f"{content_key}_undo"
    st.session_state.section_content[content_key] = st.session_state.section_content.get(content_key, {'text': '', 'creativity': 'green'})
    st.session_state.section_content[undo_key] = st.session_state.section_content[content_key].copy()
    
    # Combine original sub_instruction with user hint
    enhanced_sub_instruction = instruction.sub_instruction or ""
    if user_hint:
        if enhanced_sub_instruction:
            enhanced_sub_instruction += f"\n\nAdditional guidance: {user_hint}"
        else:
            enhanced_sub_instruction = f"Additional guidance: {user_hint}"
    
    # Handle GENERATE_FROM_KB_AND_PROPOSAL specifically - it can work with or without existing content
    if instruction.task_type == "GENERATE_FROM_KB_AND_PROPOSAL":
        logger.info(f"🔍 DEBUG: Processing GENERATE_FROM_KB_AND_PROPOSAL task")
        if has_content:
            logger.info(f"🔍 DEBUG: Has content - calling refine_with_ai")
            with st.spinner("AI is refining with KB context..."):
                generated_text = engine.refine_with_ai(
                    section_heading=instruction.section_heading,
                    raw_text=raw_text,
                    creativity=creativity,
                    sub_instruction=enhanced_sub_instruction
                )
        else:
            logger.info(f"🔍 DEBUG: No content - calling generate_from_kb")
            # Create temporary instruction with enhanced sub_instruction for generate_from_kb
            enhanced_instruction = CortexInstruction(
                section_heading=instruction.section_heading,
                instruction_raw=instruction.instruction_raw,
                task_type=instruction.task_type,
                parameter=instruction.parameter,
                placeholder_paragraph=instruction.placeholder_paragraph,
                sub_instruction=enhanced_sub_instruction
            )
            with st.spinner("AI is generating from KB and proposal context..."):
                generated_text = engine.generate_from_kb(
                    instruction=enhanced_instruction,
                    creativity=creativity,
                    knowledge_sources=st.session_state.knowledge_sources,
                    session_state=st.session_state
                )
    elif has_content and (instruction.task_type in REFINEMENT_ONLY_TASKS or instruction.task_type in GENERATIVE_TASKS):
        with st.spinner("AI is refining..."):
            generated_text = engine.refine_with_ai(
                section_heading=instruction.section_heading,
                raw_text=raw_text,
                creativity=creativity,
                sub_instruction=enhanced_sub_instruction
            )
    elif not has_content and instruction.task_type in GENERATIVE_TASKS:
        # Create temporary instruction with enhanced sub_instruction for generate_from_kb
        enhanced_instruction = CortexInstruction(
            section_heading=instruction.section_heading,
            instruction_raw=instruction.instruction_raw,
            task_type=instruction.task_type,
            parameter=instruction.parameter,
            placeholder_paragraph=instruction.placeholder_paragraph,
            sub_instruction=enhanced_sub_instruction
        )
        with st.spinner("AI is generating a new draft..."):
            generated_text = engine.generate_from_kb(
                instruction=enhanced_instruction,
                creativity=creativity,
                knowledge_sources=st.session_state.knowledge_sources,
                session_state=st.session_state
            )
    elif instruction.task_type in RETRIEVAL_TASKS:
        # Create temporary instruction with enhanced sub_instruction for retrieve_from_kb
        enhanced_instruction = CortexInstruction(
            section_heading=instruction.section_heading,
            instruction_raw=instruction.instruction_raw,
            task_type=instruction.task_type,
            parameter=instruction.parameter,
            placeholder_paragraph=instruction.placeholder_paragraph,
            sub_instruction=enhanced_sub_instruction
        )
        with st.spinner("Finding relevant case studies..."):
            generated_text = engine.retrieve_from_kb(
                instruction=enhanced_instruction,
                knowledge_sources=st.session_state.knowledge_sources,
                session_state=st.session_state
            )
            creativity = 'green'
    else:
        logger.warning(f"🔍 DEBUG: No action taken - task_type='{instruction.task_type}', has_content={bool(has_content)}")
        st.warning("Please provide some text to refine, or use a generative task."); return

    logger.info(f"🔍 DEBUG: Generated text length: {len(generated_text)} chars")
    logger.info(f"🔍 DEBUG: Generated text preview: {generated_text[:200]}...")
    
    # Show generation success message
    if generated_text:
        st.success(f"✅ Generated content for '{instruction.section_heading}' ({len(generated_text)} characters)")
    
    st.session_state.section_content[content_key] = {'text': generated_text, 'creativity': creativity}

def handle_undo_action(content_key: str):
    """Restore the previous version of the text."""
    undo_key = f"{content_key}_undo"
    if undo_key in st.session_state.section_content:
        # Restore the previous version
        st.session_state.section_content[content_key] = st.session_state.section_content[undo_key].copy()
        # Clear the undo data
        del st.session_state.section_content[undo_key]
        st.success("✅ Content restored to previous version!")

# --- Main Application ---
prop_mgr = ProposalManager()

if 'current_proposal_id' not in st.session_state:
    st.warning("No proposal selected. Please go to the Proposal Manager."); st.stop()

initialize_session_state()

config_mgr = ConfigManager()
raw_db_path = config_mgr.get_config().get("ai_database_path", DEFAULT_DB_PATH)
wsl_db_path = convert_to_docker_mount_path(raw_db_path)
index, collection_mgr = load_system(wsl_db_path)

if st.session_state.loaded_proposal_id != st.session_state.current_proposal_id:
    with st.spinner("Loading proposal state..."):
        load_proposal_into_session(st.session_state.current_proposal_id, collection_mgr)
        st.session_state.loaded_proposal_id = st.session_state.current_proposal_id

st.title(f"🤖 Co-pilot: {st.session_state.get('proposal_name', 'Untitled Proposal')}")

with st.sidebar:
    if st.button("💾 Save Progress & Draft", type="primary", use_container_width=True):
         with st.spinner("Saving..."):
            # CRITICAL FIX: Do NOT save 'parsed_instructions' to the JSON state file.
            keys_to_save = ['responding_org', 'knowledge_sources', 'section_content', 'original_filename']
            session_data_to_save = {k: st.session_state.get(k) for k in keys_to_save}
            template_bytes = None
            if st.session_state.get('doc_template'):
                bio = io.BytesIO(); st.session_state.doc_template.save(bio); template_bytes = bio.getvalue()
            prop_mgr.save_proposal(st.session_state.current_proposal_id, session_data_to_save, template_bytes, st.session_state.get('generated_doc_bytes'))
            st.toast("✅ Progress Saved!")

    if st.button("⬅️ Back to Proposal Management"): st.switch_page("pages/6_Proposal_Step_2_Make.py")
    st.divider()
    st.header("1. Proposal Context")
    st.selectbox("Responding Organisation", options=RESPONDING_ORGS, key="responding_org")
    st.multiselect("Knowledge Source(s)", options=["Main Cortex Knowledge Base"] + collection_mgr.get_collection_names(), key="knowledge_sources")

if not st.session_state.get('responding_org'):
    st.info("⬅️ Please select a Responding Organisation in the sidebar to begin."); st.stop()

if not st.session_state.get('doc_template'):
    st.header("2. Upload Proposal Template")
    uploaded_file = st.file_uploader("Upload your .docx template with Cortex instructions.", type="docx")
    if uploaded_file:
        st.session_state.original_filename = uploaded_file.name
        doc_stream = io.BytesIO(uploaded_file.getvalue()); st.session_state.doc_template = docx.Document(doc_stream)
        with st.spinner("Parsing template for instructions..."):
            st.session_state.parsed_instructions = parse_template_for_instructions(st.session_state.doc_template)
else:
    st.success(f"Template '{st.session_state.get('original_filename', 'template.docx')}' is loaded.")

if st.session_state.get('parsed_instructions'):
    st.divider()
    st.header("3. Co-pilot Drafting")
    engine = TaskExecutionEngine(main_index=index, collection_manager=collection_mgr)

    for i, inst in enumerate(st.session_state.parsed_instructions):
        content_key = f"content_inst_{i}"
        if content_key not in st.session_state.section_content:
             st.session_state.section_content[content_key] = {'text': '', 'creativity': 'green'}

        expander_title = f"Section: {inst.section_heading}  (Instruction: `{inst.instruction_raw}`)"
        with st.expander(expander_title, expanded=True):
            current_text_value = st.session_state.section_content[content_key].get('text', '')
            current_text = st.text_area(
                "Content Draft",
                value=current_text_value,
                key=f"text_area_{content_key}",
                height=150
            )
            if current_text != current_text_value:
                st.session_state.section_content[content_key]['text'] = current_text

            has_content = current_text.strip()
            button_label, is_actionable = "...", False
            # Special handling for GENERATE_FROM_KB_AND_PROPOSAL - always actionable
            if inst.task_type == "GENERATE_FROM_KB_AND_PROPOSAL":
                button_label = "✍️ Refine with KB" if has_content else "🤖 Generate from KB & Proposal"
                is_actionable = True
            elif inst.task_type in GENERATIVE_TASKS:
                button_label, is_actionable = ("✍️ Refine" if has_content else "🤖 Generate"), True
            elif inst.task_type in REFINEMENT_ONLY_TASKS:
                button_label, is_actionable = "✍️ Refine", bool(has_content)
            elif inst.task_type in RETRIEVAL_TASKS:
                button_label, is_actionable = "🔎 Find", True

            show_creativity_controls = (inst.task_type in GENERATIVE_TASKS or 
                                       inst.task_type in REFINEMENT_ONLY_TASKS or 
                                       inst.task_type == "GENERATE_FROM_KB_AND_PROPOSAL")

            # Show hint box for AI actionable sections
            if is_actionable:
                hint_key = f"hint_{content_key}"
                if hint_key not in st.session_state:
                    st.session_state[hint_key] = ""
                
                hint_text = st.text_input(
                    "💡 Additional guidance for AI (optional):",
                    value=st.session_state[hint_key],
                    key=f"hint_input_{content_key}",
                    placeholder="e.g., 'Focus on cost benefits', 'Include specific technologies', 'Mention our experience with similar projects'...",
                    help="Provide specific instructions or context to guide the AI generation"
                )
                st.session_state[hint_key] = hint_text

            if show_creativity_controls:
                control_cols = st.columns([3, 1])
                with control_cols[0]:
                    creativity_map = {"Factual": "green", "Persuasive": "orange", "Visionary": "red"}
                    color_to_label = {v: k for k, v in creativity_map.items()}
                    current_color = st.session_state.section_content[content_key].get('creativity', 'green')
                    current_label = color_to_label.get(current_color, "Factual")
                    options = list(creativity_map.keys())
                    try: current_index = options.index(current_label)
                    except ValueError: current_index = 0
                    selected_creativity_label = st.radio(
                        "AI Creativity:", options=options, index=current_index,
                        key=f"creative_radio_{content_key}", horizontal=True,
                    )
                    st.session_state.section_content[content_key]['creativity'] = creativity_map[selected_creativity_label]
                with control_cols[1]:
                    st.markdown("<div> </div>", unsafe_allow_html=True)
                    st.button(
                        button_label, key=f"btn_action_{i}", on_click=handle_ai_action,
                        args=(engine, inst, content_key), use_container_width=True,
                        type="primary", disabled=not is_actionable
                    )
                
                # Show undo button if there's undo data available
                undo_key = f"{content_key}_undo"
                if undo_key in st.session_state.section_content:
                    _, undo_col = st.columns([4, 1])
                    with undo_col:
                        st.button(
                            "↶ Undo", key=f"btn_undo_{i}", on_click=handle_undo_action,
                            args=(content_key,), use_container_width=True,
                            help="Restore previous version before AI generation"
                        )
            elif is_actionable:
                _, btn_col = st.columns([4, 1])
                with btn_col:
                    st.button(
                        button_label, key=f"btn_action_{i}", on_click=handle_ai_action,
                        args=(engine, inst, content_key), use_container_width=True, type="primary"
                    )
                
                # Show undo button if there's undo data available (for non-creativity sections)
                undo_key = f"{content_key}_undo"
                if undo_key in st.session_state.section_content:
                    _, undo_col = st.columns([4, 1])
                    with undo_col:
                        st.button(
                            "↶ Undo", key=f"btn_undo_simple_{i}", on_click=handle_undo_action,
                            args=(content_key,), use_container_width=True,
                            help="Restore previous version before AI generation"
                        )

    st.divider()
    st.header("4. Assemble & Download")
    if st.button("🚀 Assemble Proposal", type="primary", use_container_width=True):
        if not st.session_state.get('doc_template'):
            st.error("Template not found.")
        else:
            with st.spinner("Assembling final document with color-coding..."):
                template_bio = io.BytesIO(); st.session_state.doc_template.save(template_bio)
                final_doc_bytes = engine.assemble_document(
                    st.session_state.parsed_instructions,
                    st.session_state.section_content,
                    template_bio.getvalue()
                )
                st.session_state.generated_doc_bytes = final_doc_bytes
                st.success("✅ Proposal Assembled!"); st.balloons()

    if st.session_state.generated_doc_bytes:
        dl_filename = f"DRAFT_{st.session_state.get('proposal_name', 'proposal').replace(' ', '_')}.docx"
        st.download_button("📥 Download Generated Proposal", st.session_state.generated_doc_bytes, file_name=dl_filename)

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass

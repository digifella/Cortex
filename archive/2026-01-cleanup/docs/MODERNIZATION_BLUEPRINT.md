# Cortex Suite Modernization Blueprint
## Version 5.0.0 - "Universal Knowledge Assistant"

**Created:** 2026-01-01
**Status:** In Progress
**Goal:** Radical simplification with local LLM optimization

---

## 🎯 Overview

Modernize Cortex Suite's knowledge work features by consolidating three overlapping tools into two streamlined interfaces that leverage local LLM capabilities on RTX 8000 GPU.

### Current State (v4.11.0)
- **AI Assisted Research** (414 lines UI + 610 lines backend) - 6-7 step wizard, external only
- **Idea Generator** (1,002 lines UI + 857 lines backend) - 6+ step wizard, internal RAG
- **Knowledge Synthesizer** (101 lines UI + 20 lines stub) - Abandoned placeholder

### Target State (v5.0.0)
- **Universal Knowledge Assistant** - Single unified interface, internal + external, 1-2 steps
- **Idea Generator Express** - Streamlined ideation with optional deep mode

---

## 🏗️ Architecture Components

### 1. Adaptive Model Manager
**Location:** `cortex_engine/adaptive_model_manager.py`

**Purpose:** Dynamically detect and intelligently select Ollama models based on task requirements and system capabilities.

**Key Features:**
- Auto-discover all available Ollama models via API
- Categorize models by capability tier (router/fast, mid-range, power)
- Smart model selection based on task type (research, ideation, synthesis)
- Support for new models without code changes (Nemotron, Llama 3.3, Qwen 2.5)
- GPU-aware recommendations using existing smart_model_selector

**API Design:**
```python
class AdaptiveModelManager:
    def get_available_models() -> Dict[str, ModelCapabilities]
    def recommend_model(task_type: str, preference: str = "balanced") -> str
    def get_model_info(model_name: str) -> ModelInfo
    def categorize_models() -> Dict[str, List[str]]  # fast/mid/power
```

**Model Capability Detection:**
- Parse model names for known patterns (llama3.3:70b → power tier)
- Use size heuristics (< 5GB = fast, 5-20GB = mid, > 20GB = power)
- Support user-defined model profiles for special cases
- Track quantization levels (q4_K_M, q8_0, etc.)

---

### 2. Universal Knowledge Assistant
**Location:**
- Backend: `cortex_engine/universal_assistant.py`
- UI: `pages/1_Universal_Knowledge_Assistant.py`

**Purpose:** Single interface for all knowledge work (research, synthesis, ideation).

**Workflow:**
```
User Input (topic/question/goal)
       ↓
Intent Classification (fast local LLM - llama3.2:3b)
       ↓
Parallel Execution:
  ├─ Internal RAG Search (ChromaDB + NetworkX)
  ├─ External Search (Semantic Scholar, YouTube)
  └─ LLM Analysis (power model - llama3.3:70b or qwen2.5:72b)
       ↓
Streaming Synthesis (real-time UI updates)
       ↓
Output + Refinement Options
```

**UI Design:**
```
┌───────────────────────────────────────────────────────┐
│ 🧠 Universal Knowledge Assistant                      │
├───────────────────────────────────────────────────────┤
│ What would you like to explore?                       │
│ [Large text area - accepts questions, topics, goals] │
│                                                        │
│ Knowledge Sources:                                     │
│ ☑ Internal Knowledge Base  ☑ Academic Papers          │
│ ☑ External Research       ☐ YouTube/Videos            │
│                                                        │
│ Depth: ○ Quick  ● Thorough  ○ Deep Research           │
│                                                        │
│ Working Collection: [Dropdown - optional filter]      │
│                                                        │
│ [🚀 Generate Knowledge]                               │
└───────────────────────────────────────────────────────┘

Results (streaming):
├─ Sources Found (real-time counter)
├─ Key Themes (as identified)
└─ Synthesis (streaming text)

Actions:
[💾 Save to Collection] [📄 Export Report] [🔍 Deep Dive on Theme]
```

**Backend Architecture:**
```python
class UniversalKnowledgeAssistant:
    def __init__(self, model_manager: AdaptiveModelManager):
        self.router_llm = model_manager.recommend_model("router")
        self.power_llm = model_manager.recommend_model("analysis")
        self.rag_engine = ExistingRAGEngine()
        self.external_engine = ExternalResearchEngine()

    async def process_query(self, user_input, sources, depth):
        # Fast intent classification
        intent = await self._classify_intent(user_input)

        # Parallel execution
        results = await asyncio.gather(
            self._search_internal(user_input) if sources["internal"] else None,
            self._search_external(user_input) if sources["external"] else None,
            self._analyze_context(user_input, depth)
        )

        # Streaming synthesis
        async for chunk in self._synthesize_stream(results, depth):
            yield chunk
```

---

### 3. Enhanced Idea Generator (Express Mode)
**Location:**
- Keep existing: `pages/10_Idea_Generator.py`
- Enhance existing: `cortex_engine/idea_generator/core.py`

**Changes:**
1. Add "Express Mode" toggle at top of page
2. Express Mode workflow:
   - Skip theme selection (AI auto-selects top 5 themes)
   - Skip problem statement curation (AI generates top 3)
   - Skip configuration (use balanced creativity)
   - Show streaming ideas as generated
   - Display total time: target < 2 minutes

3. Keep existing "Advanced Mode" for power users
4. Use adaptive model manager for intelligent model selection
5. Add multi-agent ideation option (3 local models with different perspectives)

**UI Addition:**
```
┌─────────────────────────────────────────┐
│ Mode: ○ Express (2 min)  ● Advanced     │
└─────────────────────────────────────────┘

[If Express Mode:]
  Collection: [Select]
  Innovation Goal: [Text input]
  [🚀 Generate Ideas] ← One-click ideation

[If Advanced Mode:]
  [Existing multi-step workflow]
```

---

### 4. Model Recommendation Engine
**Location:** `cortex_engine/model_recommendations.py`

**Purpose:** Intelligent model suggestions for different tasks.

**Recommended Models to Pull:**
```bash
# Fast Router/Classifier (2-5GB)
ollama pull llama3.2:3b-instruct-q8_0        # Already have ✓
ollama pull qwen2.5:3b-instruct-q8_0         # New - excellent small model

# Mid-Range (10-20GB)
ollama pull qwen2.5:14b-instruct-q4_K_M      # New - great reasoning
ollama pull mistral-small:latest             # Already have ✓

# Power Models (30-50GB)
ollama pull llama3.3:70b-instruct-q4_K_M     # New - latest Llama
ollama pull qwen2.5:72b-instruct-q4_K_M      # New - SOTA reasoning
ollama pull nemotron:70b-instruct-q4_K_M     # New - NVIDIA optimized

# Specialized
ollama pull nemotron-embed:latest            # New - NVIDIA embedding model
```

**Task-Model Mapping:**
```python
RECOMMENDED_MODELS = {
    "router": {
        "primary": "llama3.2:3b-instruct-q8_0",
        "fallback": "qwen2.5:3b-instruct-q8_0"
    },
    "research": {
        "primary": "qwen2.5:72b-instruct-q4_K_M",
        "fallback": "llama3.3:70b-instruct-q4_K_M"
    },
    "ideation": {
        "primary": "llama3.3:70b-instruct-q4_K_M",
        "fallback": "qwen2.5:14b-instruct-q4_K_M"
    },
    "synthesis": {
        "primary": "qwen2.5:72b-instruct-q4_K_M",
        "fallback": "mistral-small:latest"
    }
}
```

---

## 📋 Implementation Phases

### Phase 1: Foundation (Week 1)
**Priority:** Build core infrastructure

1. **Adaptive Model Manager** ✓
   - Model discovery and categorization
   - Smart recommendations
   - Integration with existing smart_model_selector
   - Unit tests

2. **Pull Recommended Models**
   - Llama 3.3 70B (latest, better than current Llama 3.0)
   - Qwen 2.5 72B (excellent reasoning)
   - Qwen 2.5 14B (mid-range option)
   - Nemotron 70B (NVIDIA-optimized)

3. **Async/Streaming Infrastructure**
   - Enhance existing async_query.py
   - Add streaming support to UI components
   - Test real-time updates in Streamlit

### Phase 2: Universal Knowledge Assistant (Week 2-3)
**Priority:** Build and test new unified interface

1. **Backend Engine**
   - Intent classification
   - Parallel search execution
   - Streaming synthesis
   - Integration with existing RAG
   - Integration with existing external research (Semantic Scholar, YouTube)

2. **UI Implementation**
   - Clean, simple interface
   - Real-time streaming results
   - Progressive disclosure (simple by default, advanced when needed)
   - Save/export functionality

3. **Testing & Optimization**
   - Test with various query types
   - Optimize prompts for local models
   - Performance benchmarking
   - User experience validation

### Phase 3: Idea Generator Express (Week 3-4)
**Priority:** Streamline existing feature

1. **Add Express Mode**
   - One-click ideation
   - Auto-theme selection
   - Streaming idea generation
   - Multi-agent option (3 perspectives)

2. **Preserve Advanced Mode**
   - Keep existing workflow for power users
   - Migrate to adaptive model manager
   - Add progress indicators
   - Optimize for speed

3. **Testing**
   - Compare Express vs Advanced outputs
   - Validate quality doesn't degrade
   - Performance benchmarking

### Phase 4: Deprecation & Cleanup (Week 4)
**Priority:** Remove old features, update docs

1. **Deprecate Old Features**
   - Remove AI Assisted Research page (functionality moved to Universal Assistant)
   - Remove Knowledge Synthesizer stub
   - Update navigation/menu
   - Migration guide for users

2. **Documentation Updates**
   - Update CLAUDE.md
   - Update README.md
   - Add migration guide
   - Update version numbers

3. **Docker Distribution**
   - Sync all changes to docker/
   - Update Docker README
   - Test Docker deployment
   - Update installer scripts

---

## 🔧 Technical Implementation Details

### Async/Streaming Pattern
```python
# Streamlit streaming integration
async def stream_results():
    assistant = UniversalKnowledgeAssistant(model_manager)
    async for chunk in assistant.process_query(user_input, sources, depth):
        yield chunk

# UI usage
with st.status("Generating knowledge...", expanded=True) as status:
    st.write_stream(stream_results())
    status.update(label="Complete!", state="complete")
```

### Multi-Agent Ideation
```python
class MultiAgentIdeator:
    def __init__(self, model_manager):
        self.agents = [
            Agent("practical", model_manager.get_model("ideation")),
            Agent("creative", model_manager.get_model("research")),
            Agent("critical", model_manager.get_model("synthesis"))
        ]

    async def ideate(self, problem_statement):
        # Parallel ideation from different perspectives
        ideas = await asyncio.gather(*[
            agent.generate_ideas(problem_statement)
            for agent in self.agents
        ])

        # Synthesize diverse perspectives
        return await self._synthesize_ideas(ideas)
```

### Model Auto-Discovery
```python
class AdaptiveModelManager:
    async def discover_models(self):
        """Auto-discover and categorize available Ollama models"""
        models = await self.ollama_service.list_available_models()

        categorized = {"fast": [], "mid": [], "power": []}

        for model in models:
            tier = self._categorize_model(model)
            categorized[tier].append(model)

        return categorized

    def _categorize_model(self, model: ModelInfo):
        """Categorize model by size and capabilities"""
        if model.size_gb < 5:
            return "fast"
        elif model.size_gb < 20:
            return "mid"
        else:
            return "power"
```

---

## 📊 Success Metrics

### Performance Targets
- **Universal Knowledge Assistant**: First result < 30 seconds
- **Idea Generator Express**: Complete ideation < 2 minutes
- **Model Selection**: Auto-detect and recommend < 1 second
- **Streaming**: Visible progress within 5 seconds

### Code Reduction
- **Total Lines Reduced**: ~1,500 lines (consolidation of 3 features → 2)
- **User-Facing Steps**: 6-7 steps → 1-2 steps
- **Maintenance Burden**: 3 separate codebases → 2 cohesive modules

### User Experience
- **Time to First Value**: 5x faster (6 steps → 1 step)
- **Learning Curve**: Gentler (simple by default, powerful when needed)
- **Local GPU Utilization**: 10% → 70%+ (RTX 8000 properly utilized)

---

## 🚀 Quick Start After Implementation

### For Research & Synthesis
```
1. Open "Universal Knowledge Assistant"
2. Enter your question/topic
3. Click "Generate Knowledge"
4. Get streaming results in 30 seconds
```

### For Ideation
```
[Express Mode]
1. Select collection
2. Enter innovation goal
3. Click "Generate Ideas"
4. Get ideas in < 2 minutes

[Advanced Mode]
1. Same as current workflow
2. But with better models and streaming
```

---

## 📝 Migration Notes

### For Current Users
- **AI Assisted Research** → Use Universal Knowledge Assistant with "External Research" enabled
- **Knowledge Synthesizer** → Use Universal Knowledge Assistant with "Internal Knowledge" enabled
- **Idea Generator** → Use Express Mode for quick ideation, Advanced Mode for detailed work

### Breaking Changes
- None - new features are additive
- Old pages will be deprecated in v5.1.0 (one version grace period)

---

## 🔮 Future Enhancements (v5.1.0+)

1. **Multi-Modal Support**
   - Use llava models for image analysis
   - Integrate visual research sources
   - Diagram generation for concepts

2. **Collaborative Features**
   - Save and share research sessions
   - Team knowledge bases
   - Collaborative ideation

3. **Advanced Analytics**
   - Research quality metrics
   - Idea novelty scoring
   - Knowledge gap identification

4. **Model Fine-Tuning**
   - Fine-tune local models on user's domain knowledge
   - Personalized ideation styles
   - Custom research methodologies

---

**End of Blueprint**

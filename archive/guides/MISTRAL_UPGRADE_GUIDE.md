# Mistral Small 3.2 Upgrade Guide for Cortex Suite

## 🎯 Overview

Your Cortex Suite now has a **hybrid model architecture** that optimizes performance and privacy:

- **🔒 Local Only**: Proposals, KB operations, embedding, retrieval (MUST be local)
- **🌩️ Flexible Research**: User choice between Gemini (powerful) or Local Mistral (private)

## 📊 Model Configuration

### Current Architecture:

| Component | Model | Type | Rationale |
|-----------|-------|------|-----------|
| **Proposals** | `mistral-small:3.2` | Local Only | 84% better instruction following, 50% less repetition |
| **KB Operations** | `mistral-small:3.2` | Local Only | Consistent with proposals, optimized for retrieval |
| **Research (Cloud)** | `gemini-1.5-flash` | Cloud Option | More capable for deep research |
| **Research (Local)** | `mistral:7b-instruct-v0.3-q4_K_M` | Local Option | Fast and private |
| **Embedding** | `BAAI/bge-base-en-v1.5` | Local Only | Proven performance for vector search |

## 🚀 Why Mistral Small 3.2 for Proposals?

### Key Improvements:
- **📈 84% better instruction following** (vs 82.75% in v3.1)
- **🔄 50% reduction in repetitive outputs** (critical for proposals)
- **🎯 Better system prompt adherence** 
- **⚡ 150 tokens/s generation speed**
- **🧠 24B parameters** - good balance of quality and efficiency
- **🆓 Apache 2.0 license** - no licensing concerns

### Benchmark Results:
- **HumanEval Plus**: 88.99% → 92.90%
- **MBPP Pass@5**: 74.63% → 78.33%
- **Arena Hard**: 19.56% → 43.10% (2x improvement!)
- **Wildbench**: ~10 percentage point improvement

## 🛠️ Setup Instructions

### 1. Install Mistral Small 3.2

**IMPORTANT**: The correct model name in Ollama is `mistral-small3.2` (no colon)

```bash
# Option A: Full model (~15GB, best quality) - CORRECTED NAME
ollama pull mistral-small3.2

# Option B: Alternative if above doesn't work
ollama pull mistral-small
```

**Note**: The download is large (~15GB) and may take time depending on your connection.

### 2. System Requirements

- **RAM**: 55GB+ for full precision (bf16/fp16)
- **Disk**: ~30GB free space
- **GPU**: Recommended but not required

### 3. Configure Environment

Your `.env` file is already configured! Just change the provider:

```bash
# For proposals (already set to local)
OLLAMA_PROPOSAL_MODEL="mistral-small:3.2"

# For research, you can choose in the UI or set default:
RESEARCH_LLM_PROVIDER="gemini"  # or "ollama" for local-only
```

## 🎮 How to Use

### AI Research Assistant
1. Open **🤖 AI Assisted Research**
2. Choose your AI provider:
   - **🌩️ Gemini (Cloud)**: More capable, requires internet
   - **🏠 Local Mistral**: Private and fast
3. The system will automatically use your choice

### Proposal Generation
- **Automatically uses Mistral Small 3.2** (no choice needed)
- Enhanced instruction following for better proposal quality
- Reduced repetition and more consistent outputs

### Knowledge Base Operations
- **Automatically uses local models** for privacy
- Fast retrieval and consistent results

## 🧪 Testing Your Setup

Run this command to test configuration:

```bash
python -c "
from cortex_engine.config import PROPOSAL_LLM_MODEL, KB_LLM_MODEL
print('Proposal Model:', PROPOSAL_LLM_MODEL)
print('KB Model:', KB_LLM_MODEL)
print('✅ Configuration loaded successfully')
"
```

Test Ollama models:

```bash
# Test proposal model
ollama run mistral-small:3.2 "Write a professional proposal introduction in one sentence."

# Test research model  
ollama run mistral:7b-instruct-v0.3-q4_K_M "Summarize the benefits of local AI models."
```

## 📈 Expected Benefits

### For Proposals:
- **More professional language** and tone
- **Better adherence to instructions** and templates
- **Less repetitive content** in generated sections
- **Faster generation** (150 tokens/s)
- **Complete privacy** (runs locally)

### For Research:
- **User choice** between power (Gemini) and privacy (Local)
- **Consistent embeddings** and retrieval
- **Local control** when needed

## ⚠️ Troubleshooting

### Model Not Found
```bash
# Ensure Ollama is running
ollama serve

# Pull the model again
ollama pull mistral-small:3.2
```

### Memory Issues
```bash
# Use quantized version instead
ollama pull mistral-small

# Update .env to use quantized model
OLLAMA_PROPOSAL_MODEL="mistral-small"
```

### Provider Switching
- Research provider choice is **per session** in the UI
- Restart the app to reset all models
- Check logs for model loading messages

## 🔧 Configuration Files Modified

✅ Updated files:
- `cortex_engine/config.py` - New model architecture
- `cortex_engine/task_engine.py` - Optimized for proposals
- `cortex_engine/query_cortex.py` - Local-only KB operations
- `cortex_engine/synthesise.py` - Dynamic provider selection
- `pages/1_AI_Assisted_Research.py` - UI provider choice
- `.env` - Hybrid configuration

## 🎉 You're Ready!

Your Cortex Suite now has the optimal model configuration:
- **🔒 Privacy**: All sensitive operations run locally
- **🚀 Performance**: Mistral Small 3.2 for better proposals
- **🌩️ Flexibility**: Choose your research AI provider
- **⚡ Speed**: Optimized models for each task

**Next Steps:**
1. Start Ollama: `ollama serve`
2. Pull models: `ollama pull mistral-small:3.2`
3. Launch Cortex Suite: `streamlit run Cortex_Suite.py`
4. Test proposal generation with improved quality!
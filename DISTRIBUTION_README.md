# 📦 Cortex Suite - Distribution Package

**AI-Powered Knowledge Management System - Ready to Share!**

*System designed and built by Paul Cooper, Director of Longboardfella Consulting Pty Ltd.*

## 🎯 What This Is

Cortex Suite is a complete AI-powered knowledge management and proposal generation system that:
- 📚 **Ingests documents** (PDFs, Word, PowerPoint, images) with AI entity extraction
- 🔍 **Provides intelligent search** using hybrid vector + graph technology  
- 📝 **Generates proposals** using your knowledge base
- 🗂️ **Manages collections** with advanced organization tools
- 🔒 **Anonymizes documents** by replacing names, companies, and sensitive info with generic placeholders
- 💾 **Backs up data** with Windows path management
- 🔧 **Offers REST API** for external integrations
- 🤖 **NEW: Intelligent Model Management** - Automatically detects and guides model installation
- 📊 **NEW: System Status Dashboard** - Real-time monitoring of AI model availability
- ⚡ **NEW: Enhanced Error Handling** - Clear guidance when models are missing

## 📁 Files to Share

When distributing to others, include these files/folders:

### Essential Structure:
```
cortex-suite/
├── 📁 cortex_engine/          (Complete folder)
├── 📁 pages/                  (Complete folder)
├── 📁 api/                    (Complete folder)
├── 📁 docker/                 (Complete folder)
├── 📄 Cortex_Suite.py         (Main application)
├── 📄 requirements.txt        (Dependencies)
└── 📄 README.md              (This file)
```

### Key Distribution Files:
- `docker/QUICK_START.md` - Simple setup guide
- `docker/run-cortex.sh` - Linux/Mac launcher
- `docker/run-cortex.bat` - Windows launcher  
- `docker/.env.example` - Configuration template

## 🚀 User Setup Instructions

### For the Person Receiving This:

1. **Install Docker Desktop** (5 minutes)
   - Windows/Mac: https://www.docker.com/products/docker-desktop/
   - Linux: `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`

2. **Extract files** to a folder on your computer

3. **Run Cortex Suite** (Two options):

### Option A: Enhanced Setup with Model Management (Recommended)
   - **Linux/Mac**: `cd docker && ./run-cortex-with-models.sh`
   - **Windows**: Run `docker/run-cortex-with-models.sh` in Git Bash or WSL
   - ✅ **Automatically checks and installs required AI models**
   - ✅ **Interactive setup for optional models (vision, enhanced proposals)**
   - ✅ **System status validation**

### Option B: Quick Setup (Original)
   - **Windows**: Double-click `docker/run-cortex.bat`
   - **Linux/Mac**: Double-click `docker/run-cortex.sh`
   - ⚠️ **May require manual model installation later**
   - **Windows**: Double-click `docker/run-cortex.bat`
   - **Mac/Linux**: Double-click `docker/run-cortex.sh`

4. **Wait 5-10 minutes** for first-time setup (downloads AI models)

5. **Open browser** to http://localhost:8501

**That's it!** 🎉

## 📋 Distribution Methods

### Method 1: ZIP Archive (Recommended)
1. Create ZIP with all essential files above
2. Include simple instructions:
   ```
   Cortex Suite - AI Knowledge Management
   
   1. Install Docker Desktop
   2. Extract ZIP file  
   3. Run: docker/run-cortex.bat (Windows) or docker/run-cortex.sh (Mac/Linux)
   4. Open: http://localhost:8501
   ```

### Method 2: Git Repository
```bash
git clone your-repo-url
cd cortex-suite/docker
./run-cortex.sh  # or run-cortex.bat on Windows
```

### Method 3: Pre-built Docker Image (Advanced)
```bash
# Build once, share image
docker build -t cortex-suite -f docker/Dockerfile .
docker save cortex-suite > cortex-suite.tar

# User loads and runs
docker load < cortex-suite.tar
docker run -d --name cortex-suite -p 8501:8501 -p 8000:8000 -v cortex_data:/home/cortex/data cortex-suite
```

## 🛠️ What The System Includes

### Core Features:
- **Local AI Processing** - No cloud dependencies, complete privacy
- **GraphRAG Technology** - Advanced entity extraction and relationship mapping
- **Multi-format Document Support** - PDFs, Word, PowerPoint, images, text
- **Hybrid Search** - Vector similarity + graph relationships
- **Proposal Generation** - AI-assisted writing using your knowledge base
- **Windows Path Management** - Seamless cross-platform file handling
- **Backup/Restore** - Complete data protection with user-friendly paths
- **REST API** - Full programmatic access for integrations

### Technical Stack:
- **Frontend**: Streamlit with multi-page UI
- **Backend**: FastAPI with async processing
- **AI Models**: Mistral (7B + Small) running locally via Ollama
- **Vector Database**: ChromaDB for embeddings
- **Graph Database**: NetworkX for relationships
- **Entity Extraction**: spaCy NLP with custom patterns
- **Deployment**: Single Docker container with all services

## 📊 System Requirements

### Minimum:
- **RAM**: 4GB
- **Storage**: 10GB free space
- **CPU**: 2 cores
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **Internet**: For initial model downloads only

### Recommended:
- **RAM**: 8GB+
- **Storage**: 20GB+ free space  
- **CPU**: 4+ cores
- **GPU**: NVIDIA GPU for faster AI processing (optional)

## 🔧 User Experience

### First Launch (5-10 minutes):
1. **Build process** - Downloads Python packages, system dependencies
2. **AI model download** - ~4GB of Mistral models
3. **Service initialization** - Starts Ollama, API, and UI
4. **Health checks** - Ensures everything is running correctly

### Ongoing Use (30 seconds):
- Quick startup after first launch
- All data persists between sessions
- Automatic service recovery if processes crash
- One-click start/stop

### What Users See:
```
🚀 Starting Cortex Suite...
🤖 Starting Ollama service...
✅ Ollama is ready!
📦 Checking AI models...
✅ All models ready!
🔗 Starting API server...
✅ API is ready!
🖥️ Starting Streamlit UI...
✅ Streamlit UI is ready!
🎉 Cortex Suite is now running!
🌐 Access at: http://localhost:8501
```

## 🎯 Why This Distribution Method Works

### For Users:
- ✅ **Zero configuration** required
- ✅ **One command** does everything
- ✅ **Works on any system** with Docker
- ✅ **No Python/pip complexity**
- ✅ **Professional, polished experience**
- ✅ **Complete data privacy** (runs locally)

### For You (Distributor):
- ✅ **No support headaches** - everything "just works"
- ✅ **Consistent environment** across all users
- ✅ **Easy troubleshooting** - containerized isolation
- ✅ **Version control** - single image for updates
- ✅ **Professional appearance** - enterprise-grade solution

## 🚨 Common User Issues & Solutions

### "Docker not found"
→ Install Docker Desktop from https://docker.com

### "Port already in use"  
→ Launcher scripts handle this automatically

### "Takes too long to start"
→ Expected on first run (downloading 4GB models)

### "Out of disk space"
→ Need at least 10GB free space

### "Can't access localhost:8501"
→ Wait for "✅ Streamlit UI is ready!" message

## 🔄 Updates & Maintenance

### For You:
- Update source code and rebuild Docker image
- Share new ZIP package or Docker image
- Users get updates by re-running launcher

### For Users:
- Run `docker system prune -f` to clean up old images
- Re-run launcher script to get latest version
- Data automatically persists through updates

## 🏆 Result

Users receive a **complete, enterprise-grade AI knowledge management system** that:
- Runs entirely on their local machine
- Requires zero technical expertise to use
- Provides professional-level features
- Maintains complete data privacy
- Scales from personal to team use
- "Just works" with one command

**Perfect for sharing with colleagues, clients, or anyone who needs AI-powered document management!** 🚀
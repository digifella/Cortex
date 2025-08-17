# Cortex Suite - Docker Distribution

## What is Cortex Suite?

Cortex Suite is an AI-powered knowledge management and proposal generation system featuring:
- **Knowledge Ingestion**: Upload and process documents with entity extraction
- **Smart Search**: Hybrid vector + graph search across your knowledge base  
- **AI Research**: Multi-agent research system with automated report generation
- **Proposal Generation**: Create proposals using your knowledge base + AI assistance
- **Idea Generator**: Double Diamond innovation methodology for structured ideation

## Quick Start

### Windows
```cmd
run-cortex.bat
```

### Mac/Linux
```bash
chmod +x run-cortex.sh
./run-cortex.sh
```

## New: Service-First Architecture

**⚡ Immediate Access**: Web interface becomes accessible in ~10 seconds
**📦 Background Downloads**: AI models (20GB) download while interface is usable
**📊 Progress Tracking**: Professional setup page shows real-time status
**🔄 Progressive Activation**: Features enable as models complete (15-30 minutes)

### What You'll See
1. **10 seconds**: Web interface accessible at http://localhost:8501
2. **Setup Progress Page**: Real-time download status and system health
3. **Immediate Exploration**: Browse interface, configure settings while models download
4. **Automatic Activation**: Full AI features enable as downloads complete

## Prerequisites
- Docker Desktop installed and running
- Windows 10/11, macOS, or Linux
- 10GB free disk space
- Internet connection for initial setup

## 🔐 IMPORTANT: Environment Setup (Required)

**BEFORE FIRST RUN**: You must create your environment configuration:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys (optional for local-only usage)
nano .env  # or use any text editor
```

### Required for Cloud AI Features (Optional):
If you want to use cloud AI providers for research (instead of local models only):

```bash
# Add your API keys to .env file:
OPENAI_API_KEY=your_actual_openai_key_here
GEMINI_API_KEY=your_actual_gemini_key_here
YOUTUBE_API_KEY=your_actual_youtube_key_here
```

⚠️ **SECURITY NOTE**: 
- Never commit `.env` files to version control
- The `.env` file contains your API keys and should remain private
- For local-only usage, you can use the default ollama settings

## What Gets Created Fresh

Every installation creates a completely fresh environment:
- ✅ New ChromaDB vector database
- ✅ New knowledge graph
- ✅ Empty working collections
- ✅ Fresh configuration files
- 🚫 No pre-existing user data

## Common Commands

```bash
# Stop Cortex Suite
docker stop cortex-suite

# Start Cortex Suite  
docker start cortex-suite

# View logs
docker logs cortex-suite -f

# Remove completely (to reinstall)
docker stop cortex-suite && docker rm cortex-suite
```

## Troubleshooting

**Build fails with "Access denied" on Windows:**
- Make sure Docker Desktop is running
- Try: `docker system prune -f` then retry

**Cannot access localhost:8501:**
- Interface should be available within 10 seconds
- Check setup progress page for real-time status
- Check `docker logs cortex-suite` for startup progress

**Out of space errors:**
- Ensure 10GB+ free disk space
- Run: `docker system prune -f` to clean up

## Support

For issues or questions, check the logs with `docker logs cortex-suite -f` for detailed error messages.

---
*Cortex Suite v1.0 - AI-Powered Knowledge Management*

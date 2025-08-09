# 🚀 Cortex Suite - Super Simple Install for Friends

**Get this amazing AI knowledge management system running in 10 minutes!**

## What You're Getting

The Cortex Suite is like having your own personal AI research assistant that:
- 📚 **Reads your documents** (PDFs, Word files, PowerPoints, etc.)
- 🧠 **Remembers everything** and connects the dots between different pieces of information
- 🔍 **Answers questions** about your documents using AI
- 📝 **Helps write proposals** using your knowledge base
- 🗂️ **Organizes everything** automatically

## Before You Start (5 minutes)

### Step 1: Install Docker
Docker is like a magic box that makes software work everywhere. Install it once and you're set!

**Windows:**
1. Go to https://www.docker.com/products/docker-desktop/
2. Download Docker Desktop for Windows
3. Install it (you might need to restart your computer)
4. Make sure you see a whale icon in your system tray

**Mac:**
1. Go to https://www.docker.com/products/docker-desktop/
2. Download Docker Desktop for Mac
3. Install and start it
4. Wait until you see "Docker Desktop is running"

**Linux:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and log back in
```

### Step 2: Check Docker Works
Open a terminal/command prompt and type:
```bash
docker --version
```
You should see something like "Docker version 20.x.x"

## Install Cortex Suite (2 commands!)

### Option A: Super Simple (Recommended)

1. **Extract** the Cortex Suite files you received to a folder
2. **Open terminal/command prompt** in that folder
3. **Navigate to the docker folder**:
   ```bash
   cd docker
   ```
4. **Run these 2 commands**:
   ```bash
   # Build Cortex Suite (takes 5-10 minutes first time)
   docker build -t cortex-suite -f Dockerfile ..
   
   # Start Cortex Suite
   docker run -d --name cortex-suite -p 8501:8501 -p 8000:8000 -v cortex_data:/home/cortex/data cortex-suite
   ```

### Option B: Even Easier (Windows)
Just double-click `run-cortex.bat` in the docker folder!

### Option B: Even Easier (Mac/Linux)
Just double-click `run-cortex.sh` in the docker folder!

## Wait and Access (5-10 minutes first time)

**First launch takes longer** because it downloads AI models (~4GB). You'll see:
```
🚀 Starting Cortex Suite...
🤖 Starting Ollama service...
⬇️ Downloading AI models (this may take a few minutes)...
✅ All models ready!
🎉 Cortex Suite is now running!
🌐 Access at: http://localhost:8501
```

**Then open your browser and go to:** http://localhost:8501

## What Now? Start Using It!

### 1. Upload Your First Document 📚
- Click "Knowledge Ingest" in the sidebar
- Upload a PDF, Word doc, or PowerPoint
- The AI will read it and extract important information

### 2. Ask Questions 🔍
- Click "Knowledge Search" 
- Type questions like "What projects did we work on last year?" or "Who are our key clients?"
- Get AI-powered answers from your documents

### 3. Create Proposals 📝
- Click "Proposal Step 1 Prep" to set up templates
- Click "Proposal Step 2 Make" to create new proposals
- The AI uses your knowledge base to help write them

### 4. Organize Everything 🗂️
- Click "Collection Management" to group related documents
- Everything is searchable and connected

## Daily Use

**Starting Cortex Suite again:**
- Just double-click the launcher script, OR
- Open terminal and run: `docker start cortex-suite`
- Then go to http://localhost:8501

**Stopping Cortex Suite:**
- Close the terminal window, OR
- Run: `docker stop cortex-suite`

## Troubleshooting

### "Port already in use" 
Someone else is using those ports. Try:
```bash
docker stop cortex-suite 2>/dev/null || true
docker run -d --name cortex-suite -p 8502:8501 -p 8001:8000 -v cortex_data:/home/cortex/data cortex-suite
# Then use: http://localhost:8502
```

### "Docker not running"
- Make sure Docker Desktop is open and running
- Look for the whale icon in your system tray (Windows) or menu bar (Mac)

### "Taking forever to start"
- First time downloads ~4GB of AI models - be patient!
- Check progress: `docker logs cortex-suite -f`
- Next time it starts in 30 seconds

### "Can't access localhost:8501"
- Wait for "✅ Streamlit UI is ready!" in the logs
- Try refreshing your browser
- Make sure no firewall is blocking it

### "Out of space"
- You need at least 10GB free disk space
- The AI models are large but powerful!

## Your Data is Safe

- Everything is stored in Docker volumes that persist between restarts
- Your documents, settings, and knowledge base are automatically saved
- You can backup through the web interface

## System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space
- **Internet**: Only needed for first-time model downloads
- **OS**: Windows 10+, macOS 10.14+, or modern Linux

## Getting Help

1. **Check what's happening**: `docker logs cortex-suite`
2. **Restart everything**: `docker restart cortex-suite`
3. **Start fresh**: Stop, remove, and rebuild the container
4. **Make sure you have enough space and memory**

## Complete Removal (if needed)

If you want to remove everything:
```bash
docker stop cortex-suite
docker rm cortex-suite
docker rmi cortex-suite
docker volume rm cortex_data
```

---

**🎉 That's it! You now have your own AI-powered knowledge management system!**

**Questions?** Check the logs or feel free to ask for help!
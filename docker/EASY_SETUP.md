# 🚀 Cortex Suite - Easy Setup Guide

**Run the complete Cortex Suite with just 3 commands!**

## Prerequisites (5 minutes)

1. **Install Docker Desktop**: 
   - Windows/Mac: Download from https://www.docker.com/products/docker-desktop/
   - Linux: `curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh`

2. **Verify Docker is running**:
   ```bash
   docker --version
   ```

## Quick Start (2 minutes)

1. **Download and extract** the Cortex Suite files to a folder

2. **Open terminal/command prompt** in the `docker` folder

3. **Run these 3 commands**:
   ```bash
   # Copy the configuration file
   cp .env.example .env
   
   # Build and start Cortex Suite (this takes 5-10 minutes first time)
   docker build -t cortex-suite -f Dockerfile ..
   
   # Run Cortex Suite
   docker run -d --name cortex-suite -p 8501:8501 -p 8000:8000 -v cortex_data:/home/cortex/data cortex-suite
   ```

4. **Wait 2-3 minutes** for everything to start, then open:
   - **Main Application**: http://localhost:8501
   - **API Documentation**: http://localhost:8000/docs

## That's it! 🎉

The system will automatically:
- ✅ Download and install all AI models
- ✅ Set up the database
- ✅ Start all services
- ✅ Handle all dependencies

## Common Issues & Solutions

**"Port already in use" error?**
```bash
# Use different ports
docker run -d --name cortex-suite -p 8502:8501 -p 8001:8000 -v cortex_data:/home/cortex/data cortex-suite
# Then use: http://localhost:8502
```

**Docker not running?**
- Make sure Docker Desktop is open and running
- On Windows: Look for Docker whale icon in system tray

**Out of disk space?**
- Free up at least 10GB of disk space
- The system downloads ~4GB of AI models

## Managing Cortex Suite

**Stop Cortex Suite:**
```bash
docker stop cortex-suite
```

**Start Cortex Suite again:**
```bash
docker start cortex-suite
```

**Remove Cortex Suite:**
```bash
docker stop cortex-suite
docker rm cortex-suite
docker rmi cortex-suite
docker volume rm cortex_data
```

**View logs if something goes wrong:**
```bash
docker logs cortex-suite
```

## System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **Internet**: Required for initial model downloads

## Features Available

Once running, you can:
- 📚 **Ingest documents** (PDFs, Word, PowerPoint, etc.)
- 🔍 **Search your knowledge base** with AI
- 📝 **Generate proposals** using your data
- 🗂️ **Manage document collections**
- 💾 **Backup your data**
- 🔧 **Use the REST API** for integrations

## Need Help?

If you encounter issues:
1. Check the logs: `docker logs cortex-suite`
2. Restart: `docker restart cortex-suite`
3. Make sure you have enough disk space and memory
4. Ensure no other applications are using ports 8501 or 8000
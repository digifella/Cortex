# Quick Start: Fixing Docker Volume Access

## Problem
Cortex Suite running in Docker cannot access files outside the project directory.

## Solution Options

### Option 1: Automated Setup (Recommended)
```bash
# Run the interactive setup script
cd docker
./setup-volumes.sh
```

This script will:
- ✅ Detect your operating system
- ✅ Suggest common document paths  
- ✅ Configure volume mounts interactively
- ✅ Validate paths and permissions
- ✅ Restart services with new configuration

### Option 2: Manual Quick Fix

**Step 1:** Add your document paths to `docker-compose.yml`

```yaml
# In both cortex-api and cortex-ui services, add under volumes:
volumes:
  - cortex_data:/data
  - cortex_logs:/app/logs
  
  # Add your document directories (READ-ONLY for safety):
  - /home/username/Documents:/host/documents:ro
  - /path/to/your/files:/host/files:ro
```

**Step 2:** Restart Docker services
```bash
docker-compose down
docker-compose up -d
```

**Step 3:** Access files in Streamlit UI
- Navigate to `/host/documents/` to see your Documents folder
- Navigate to `/host/files/` to see your custom folder

## Platform-Specific Examples

### Linux/WSL2:
```yaml
volumes:
  - /home/username/Documents:/host/documents:ro
  - /home/username/Projects:/host/projects:ro
```

### macOS:
```yaml
volumes:
  - /Users/username/Documents:/host/documents:ro
  - /Users/username/Downloads:/host/downloads:ro
```

### Windows:
```yaml
volumes:
  - C:/Users/username/Documents:/host/documents:ro
  - D:/Projects:/host/projects:ro
```

## Security Notes

- **Always use `:ro`** (read-only) unless you need write access
- **Don't mount sensitive directories** like `/home/user/.ssh`
- **Test with a small folder first** before mounting large directories

## Verification

```bash
# Check if mounts work
docker exec cortex-ui ls -la /host/documents

# View available paths in container
docker exec cortex-ui find /host -type d -maxdepth 2
```

## Need Help?

- 📖 Read the full guide: `docker/VOLUME_MOUNTING_GUIDE.md`
- 🛠️ Run the setup script: `./docker/setup-volumes.sh`
- 🔍 Check Docker logs: `docker-compose logs cortex-ui`
# Volume Mounting Guide for Cortex Suite Docker

## Problem: Cannot Access Files Outside Container

By default, Docker containers are isolated and can only access files that are explicitly mounted as volumes. This prevents the Cortex Suite from ingesting documents located outside the project directory.

## Solution: Flexible Volume Mounting

Use the `docker-compose.flexible.yml` configuration which allows mounting any host directory into the container.

## Quick Setup

### 1. Choose Your Mounting Strategy

#### Option A: Mount Specific Directories (Recommended)
```yaml
volumes:
  # Mount your Documents folder (read-only for safety)
  - /home/username/Documents:/host/documents:ro
  - /path/to/project/files:/host/projects:ro
```

#### Option B: Mount Home Directory
```yaml
volumes:
  # Mount entire home directory (read-only)
  - ${HOME}:/host/home:ro
```

#### Option C: Mount External Drives
```yaml
volumes:
  # Mount external USB/network drives
  - /media/usb-drive:/host/external:ro
  - /mnt/network-share:/host/network:ro
```

### 2. Platform-Specific Examples

#### Linux/WSL2:
```yaml
volumes:
  - /home/username/Documents:/host/documents:ro
  - /mnt/c/Users/username/Documents:/host/windows-docs:ro  # WSL2 Windows access
  - /media/external-drive:/host/external:ro
```

#### macOS:
```yaml
volumes:
  - /Users/username/Documents:/host/documents:ro
  - /Volumes/ExternalDrive:/host/external:ro
```

#### Windows (Docker Desktop):
```yaml
volumes:
  - C:\Users\username\Documents:/host/documents:ro
  - D:\Projects:/host/projects:ro
  - E:\ExternalDrive:/host/external:ro
```

### 3. Apply Your Configuration

1. **Copy the flexible compose file:**
   ```bash
   cp docker/docker-compose.flexible.yml docker/docker-compose.yml
   ```

2. **Edit the volume mounts:** 
   Uncomment and modify the volume mount lines in both `cortex-api` and `cortex-ui` services.

3. **Update the file for your system:**
   ```bash
   # Example: Mount Documents folder
   sed -i 's|# - /path/to/your/documents:/host/documents:ro|- ${HOME}/Documents:/host/documents:ro|g' docker/docker-compose.yml
   ```

4. **Restart the services:**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

## Usage in Cortex Suite

Once mounted, your external directories will be available in the container at `/host/` paths:

- `/host/documents/` → Your Documents folder
- `/host/projects/` → Your projects folder  
- `/host/home/` → Your home directory
- `/host/external/` → External drives
- `/host/network/` → Network shares

**In the Streamlit UI:** When browsing for files to ingest, navigate to these `/host/` paths to access your external documents.

## Security Considerations

### Read-Only Mounts (Recommended)
Always use `:ro` (read-only) flag for safety:
```yaml
- /path/to/documents:/host/documents:ro
```

### Avoid Mounting Sensitive Directories
**Never mount these directories:**
- `/` (root filesystem)
- `/home/user/.ssh/` (SSH keys)
- `/etc/` (system configuration)
- Directories with passwords or secrets

### Best Practices
1. **Mount only what you need:** Don't mount entire drives unless necessary
2. **Use read-only mounts:** Prevents accidental file modification
3. **Specific paths:** Mount `/home/user/Documents` instead of `/home/user`
4. **Test first:** Try mounting a small test directory first

## Advanced Configurations

### Multiple Document Sources
```yaml
volumes:
  - /home/user/Documents:/host/documents:ro
  - /home/user/Projects:/host/projects:ro
  - /media/research-drive:/host/research:ro
  - /mnt/company-share:/host/company:ro
```

### Temporary Write Access
If you need to process files that require write access:
```yaml
volumes:
  - /path/to/processing:/host/processing:rw  # Read-write for processing
  - /path/to/source:/host/source:ro         # Read-only for source files
```

### Environment Variables for Dynamic Paths
```yaml
environment:
  - DOCS_PATH=/host/documents
  - PROJECTS_PATH=/host/projects
volumes:
  - ${DOCS_PATH_HOST:-/home/user/Documents}:/host/documents:ro
  - ${PROJECTS_PATH_HOST:-/home/user/Projects}:/host/projects:ro
```

## Troubleshooting

### Common Issues

#### "Permission Denied" Errors
```bash
# Fix file permissions
sudo chown -R $(id -u):$(id -g) /path/to/documents

# Or use Docker user mapping
docker-compose run --user $(id -u):$(id -g) cortex-ui bash
```

#### "Path Not Found" Errors
```bash
# Verify path exists on host
ls -la /path/to/your/documents

# Check mount inside container
docker exec cortex-ui ls -la /host/documents
```

#### Windows Path Issues
```yaml
# Use forward slashes, even on Windows
volumes:
  - C:/Users/username/Documents:/host/documents:ro  # ✅ Correct
  - C:\Users\username\Documents:/host/documents:ro  # ❌ May cause issues
```

### Verification Commands

```bash
# Check mounted volumes
docker exec cortex-ui df -h

# List mounted directories
docker exec cortex-ui ls -la /host/

# Test file access
docker exec cortex-ui find /host/documents -name "*.pdf" | head -5
```

## Example Complete Configuration

Here's a complete working example:

```yaml
# docker-compose.yml
version: '3.8'

services:
  cortex-ui:
    # ... other configuration ...
    volumes:
      - cortex_data:/data
      - cortex_logs:/app/logs
      # User documents
      - ${HOME}/Documents:/host/documents:ro
      - ${HOME}/Downloads:/host/downloads:ro
      # Project files
      - /opt/projects:/host/projects:ro
      # External storage
      - /media/research-drive:/host/research:ro
    environment:
      - HOST_MOUNT_PATHS=/host/documents,/host/downloads,/host/projects,/host/research
      # ... other environment vars ...

  cortex-api:
    # ... other configuration ...
    volumes:
      # Same mounts as UI service
      - cortex_data:/data
      - cortex_logs:/app/logs
      - ${HOME}/Documents:/host/documents:ro
      - ${HOME}/Downloads:/host/downloads:ro
      - /opt/projects:/host/projects:ro
      - /media/research-drive:/host/research:ro
    environment:
      - HOST_MOUNT_PATHS=/host/documents,/host/downloads,/host/projects,/host/research
      # ... other environment vars ...
```

This configuration provides secure, flexible access to external directories while maintaining Docker's security benefits.
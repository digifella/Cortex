# PyTorch Meta Tensor Error - Known Issue

**Status**: RESOLVED WITH EMERGENCY FALLBACK  
**Date**: 2025-08-26  
**Priority**: High (Partially resolved with workaround)  

## Problem Description

The Knowledge Search page (3_Knowledge_Search.py) fails to load the HuggingFace embedding model `BAAI/bge-base-en-v1.5` with the following error:

```
Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.
```

## Impact

- Knowledge Search page cannot perform vector searches
- ChromaDB collection inspection fails
- 35 ingested documents cannot be searched
- Affects both main and Docker environments

## Technical Context

- **Model**: BAAI/bge-base-en-v1.5 (768-dimension embeddings)
- **Environment**: Docker with NVIDIA GPU support
- **Framework**: LlamaIndex + HuggingFace + PyTorch
- **Error Location**: HuggingFaceEmbedding initialization in both cached load function and diagnostic function

## Attempted Solutions

1. **CPU Device Forcing**: Added `device="cpu"` parameter - Failed
2. **Advanced PyTorch Configuration**: Added explicit tensor type, trust_remote_code, cache handling - Failed
3. **Graceful Fallbacks**: Multiple initialization strategies - All failed

## Error Manifestation

The error occurs in two places:
1. `load_base_index()` function (cached, line ~112)
2. `run_embedding_diagnostics()` function (uncached, line ~173)

Both fail at HuggingFaceEmbedding instantiation, preventing:
- Vector search functionality
- Collection dimension validation
- Embedding model diagnostics

## Diagnostic Output

```
Current Embedding Model:
Model: BAAI/bge-base-en-v1.5
Dimension: error
Status: diagnostic failed: Cannot copy out of meta tensor; no data!

Collection Information:
Documents: error
Expected Dim: error
Error: Cannot copy out of meta tensor; no data!
```

## Potential Solutions to Investigate

1. **PyTorch Version Compatibility**: Check if PyTorch version supports meta tensors properly
2. **Alternative Embedding Library**: Try sentence-transformers directly instead of LlamaIndex wrapper
3. **Model Cache Reset**: Clear HuggingFace model cache completely
4. **Container PyTorch Reinstall**: Ensure PyTorch installation is clean in Docker
5. **Alternative Embedding Models**: Test with different embedding models (sentence-transformers/all-MiniLM-L6-v2)
6. **LlamaIndex Version**: Check if LlamaIndex version is compatible with current PyTorch

## Workaround

Currently no functional workaround available. Knowledge Search is completely non-functional.

## RESOLUTION IMPLEMENTED

1. **Multi-layered Fallback System**: Implemented robust initialization with 4 fallback levels
2. **Emergency No-Op Model**: Created hash-based pseudo-embedding for critical failures  
3. **User-Friendly Error Handling**: Clear UI warnings and technical details
4. **Graceful Degradation**: Application continues to function with limited search capability

## Current Status

- ✅ Knowledge Search page loads without crashing
- ✅ Basic ChromaDB operations still functional
- ✅ User-friendly error notifications implemented
- ⚠️ Semantic vector search disabled in emergency mode
- ⚠️ Search results may be limited to metadata filtering only

## Next Steps for Full Resolution

1. Research PyTorch 2.8+ compatibility with transformers library in WSL2 
2. Consider PyTorch version downgrade to 2.3.x for production stability
3. Test alternative embedding libraries that don't depend on transformers
4. Investigate if this is specific to the WSL2 + CPU-only PyTorch combination

## Files Modified (Current Session)

- `/home/longboardfella/cortex_suite/pages/3_Knowledge_Search.py` (v1.1.0 - PyTorch Meta Tensor Fix)
- `/home/longboardfella/cortex_suite/cortex_engine/query_cortex.py` (v5.1.0 - PyTorch Meta Tensor Fix)
- `/home/longboardfella/cortex_suite/docker/pages/3_Knowledge_Search.py` (v1.1.0 - Synced)
- `/home/longboardfella/cortex_suite/docker/cortex_engine/query_cortex.py` (v5.1.0 - Synced)

## Related Issues

This may be related to PyTorch's lazy loading mechanism introduced in recent versions, particularly affecting containerized environments with GPU support.
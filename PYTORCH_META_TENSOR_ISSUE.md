# PyTorch Meta Tensor Error - Known Issue

**Status**: Unresolved  
**Date**: 2025-08-26  
**Priority**: High (Blocks Knowledge Search functionality)  

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

## Next Steps

1. Research PyTorch meta tensor handling in containerized environments
2. Test with alternative embedding model implementations
3. Consider downgrading PyTorch or upgrading container base image
4. Investigate if this is a known issue with LlamaIndex + HuggingFace + Docker combination

## Files Modified (Current Session)

- `/home/longboardfella/projects/Cortex/pages/3_Knowledge_Search.py` (v1.0.6)
- `/home/longboardfella/projects/Cortex/docker/pages/3_Knowledge_Search.py` (v1.0.6)

## Related Issues

This may be related to PyTorch's lazy loading mechanism introduced in recent versions, particularly affecting containerized environments with GPU support.
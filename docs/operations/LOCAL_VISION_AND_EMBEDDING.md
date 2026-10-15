# Local vision and embedding: current state (2026-10-16)

Short operational note so this knowledge is not lost again. The older Ollama / `gemma4:e2b-it-qat`
/ Qwen3-VL-via-Ollama guidance elsewhere in the repo (`QWEN3VL_SETUP_GUIDE.md`, `config.py`
`VLM_MODEL`) is superseded for the NemoClaw/Hermes stack: **Ollama has been removed from the
system.**

## Vision (captioning)

- Primary: the Qwen VLM already resident in LM Studio (`qwen/qwen3.6-35b-a3b`), via
  `cortex_engine/lmstudio_vision.py` (`describe_encoded_image`, `reasoning_effort: "none"`).
  Endpoint comes from `CORTEX_LMSTUDIO_BASE_URL` in `.env`. The helper uses a no-proxy opener
  because the Windows tailnet address must not go through `HTTP(S)_PROXY`.
- Fallback: Claude Haiku (`claude-haiku-5-5`, thinking disabled).
- LM Studio serves **one parallel session**: use concurrency 1 (~2 s per image).
- Used by `photo_batch.py tag --local-vision`, `textifier.py`, and the NemoClaw media captioner
  (`~/nemoclaw-whatsapp-vision.py`, which imports `cortex_engine.lmstudio_vision`).

## Embedding

- **EmbeddingGemma 2** (`google/embeddinggemma-2`, Apache 2.0, ungated, multimodal text+image+audio,
  768-d, 8k context) is the preferred embedder for media/vision search.
  Card: <https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2>
- Needs `transformers >= 5.20` plus torchvision. Install it in its own venv
  (`~/venvs/embeddinggemma`), not in the production `vault-rag` venv, which is too old and also
  must not be upgraded blind.
- Use `prompt_name="SearchQuery"` for queries and `title: none | text: ...` for documents; bfloat16
  or float32 only.
- `cortex_engine/config.py` still defaults `EMBED_MODEL` to `BAAI/bge-base-en-v1.5` (via
  `get_embedding_strategy`). EG2 is **not** wired into Cortex's own ingestion yet; it is used for the
  separate NemoClaw media index (`~/vault-rag-db-eg2`). Changing Cortex's embedder requires
  re-embedding every collection (vectors from different models are not comparable).

Full restore and indexing write-up: `nemoclaw_ops/docs/2026-10-16-whatsapp-sms-media-restore-and-eg2.md`.

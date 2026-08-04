"""Shared embedding-model status panel for Maintenance pages."""

from __future__ import annotations

import os

import streamlit as st


def render_embedding_model_status_panel() -> None:
    """Render embedding model/metadata compatibility status."""
    with st.container(border=True):
        st.subheader("🤖 Embedding Model Status")

        try:
            from cortex_engine.config import get_embedding_strategy
            from cortex_engine.utils.embedding_validator import validate_embedding_compatibility
            from cortex_engine.collection_manager import WorkingCollectionManager

            embed_strategy = get_embedding_strategy()
            embed_model = embed_strategy.get("model", "BAAI/bge-base-en-v1.5")
            embed_dims = embed_strategy.get("dimensions", 768)
            embed_approach = embed_strategy.get("approach", "sentence_transformers")

            model_locked = os.getenv("CORTEX_EMBED_MODEL") is not None
            if model_locked:
                lock_source = "🔒 Environment Variable"
            elif embed_approach == "qwen3vl":
                lock_source = "🤖 Qwen3-VL (auto-detected)"
            else:
                lock_source = "⚡ Auto-detected (hardware-based)"

            col_a, col_b = st.columns(2)
            with col_a:
                model_display = embed_model.split("/")[-1] if "/" in embed_model else embed_model
                st.metric("Current Model", model_display)
                st.caption(lock_source)

            with col_b:
                st.metric("Embedding Dimension", f"{embed_dims}D")

            try:
                collection_mgr = WorkingCollectionManager()
                collection_metadata = collection_mgr.get_embedding_model_metadata("default")

                if collection_metadata.get("embedding_model"):
                    st.markdown("**Collection Metadata:**")

                    stored_model = collection_metadata.get("embedding_model")
                    stored_dim = collection_metadata.get("embedding_dimension")

                    st.code(f"Stored Model: {stored_model} ({stored_dim}D)")

                    validation_result = validate_embedding_compatibility(
                        collection_metadata,
                        current_model=embed_model,
                        strict=False,
                    )

                    if not validation_result["compatible"]:
                        st.error("❌ **EMBEDDING MODEL MISMATCH DETECTED**")
                        st.error("Your current embedding model does not match the collection's model.")
                        st.error("**Search results will be unreliable!**")

                        with st.expander("🔧 Solutions", expanded=True):
                            st.markdown(
                                """
                            **Option 1: Lock to Collection's Model (Recommended)**
                            ```bash
                            export CORTEX_EMBED_MODEL="{stored_model}"
                            # Or in Windows PowerShell:
                            $env:CORTEX_EMBED_MODEL="{stored_model}"
                            ```

                            **Option 2: Delete Database and Re-ingest**
                            Use the "Delete Knowledge Base" function below to start fresh with the current model.

                            **Option 3: Run Embedding Inspector**
                            ```bash
                            python scripts/embedding_inspector.py
                            ```

                            **Option 4: Migrate to New Model**
                            ```bash
                            python scripts/embedding_migrator.py --target-model {embed_model}
                            ```
                            """.format(stored_model=stored_model, embed_model=embed_model)
                            )
                    else:
                        st.success("✅ Embedding model is compatible")
                else:
                    st.warning("⚠️ No embedding metadata found for collection")
                    st.info("This may be a legacy collection. Consider running the embedding inspector.")

            except Exception as meta_error:
                st.warning(f"Could not check collection metadata: {meta_error}")

            if not model_locked:
                st.warning("⚠️ **Recommendation:** Lock your embedding model using the `CORTEX_EMBED_MODEL` environment variable")
                st.info("This prevents automatic model switching when hardware changes, which can corrupt your database.")

        except Exception as e:
            st.error(f"Failed to load embedding model status: {e}")

"""Shared embedding-inspector panel for Maintenance pages."""

from __future__ import annotations

import os

import streamlit as st


def render_database_embedding_inspector_panel(
    *,
    config_manager_cls,
    convert_windows_to_wsl_path_fn,
    chromadb_module,
    chroma_settings_cls,
    collection_name: str,
    logger,
) -> None:
    """Render database embedding inspector and compatibility matrix."""
    with st.container(border=True):
        st.subheader("🔬 Database Embedding Inspector")
        st.caption("Analyze your ChromaDB vectors to check compatibility with different embedding models")

        if st.button("🔍 Inspect Database Embeddings", use_container_width=True, key="inspect_db_embeddings"):
            try:
                db_path = ""
                try:
                    config_manager = config_manager_cls()
                    current_config = config_manager.get_config()
                    db_path = current_config.get("ai_database_path", "")
                except Exception:
                    db_path = st.session_state.get("maintenance_current_db_input", "")

                if not db_path:
                    st.error("No database path configured")
                else:
                    wsl_path = convert_windows_to_wsl_path_fn(db_path)
                    chroma_db_path = os.path.join(wsl_path, "knowledge_hub_db")

                    if not os.path.isdir(chroma_db_path):
                        st.error(f"Database not found: {chroma_db_path}")
                    else:
                        with st.spinner("Inspecting database embeddings..."):
                            db_settings = chroma_settings_cls(anonymized_telemetry=False)
                            client = chromadb_module.PersistentClient(path=chroma_db_path, settings=db_settings)

                            try:
                                collection = client.get_collection(collection_name)
                                count = collection.count()

                                if count > 0:
                                    sample = collection.peek(limit=1)
                                    embeddings = sample.get("embeddings", None)

                                    has_embeddings = (
                                        embeddings is not None and len(embeddings) > 0 and len(embeddings[0]) > 0
                                    )

                                    if has_embeddings:
                                        stored_dim = len(embeddings[0])

                                        st.success(f"✅ Database connected: {count:,} documents")

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("Stored Embedding Dimension", f"{stored_dim}D")
                                            st.metric("Document Count", f"{count:,}")

                                        with col2:
                                            model_guess = "Unknown"
                                            if stored_dim == 768:
                                                model_guess = "BAAI/bge-base-en-v1.5"
                                            elif stored_dim == 1024:
                                                model_guess = "BAAI/bge-large-en-v1.5 or Qwen3-Embedding-0.6B"
                                            elif stored_dim == 1536:
                                                model_guess = "nvidia/NV-Embed-v2"
                                            elif stored_dim == 2048:
                                                model_guess = "Qwen3-VL-Embedding-2B"
                                            elif stored_dim == 4096:
                                                model_guess = "Qwen3-VL-Embedding-8B"
                                            elif stored_dim == 384:
                                                model_guess = "all-MiniLM-L6-v2"

                                            st.metric("Likely Original Model", model_guess.split("/")[-1])

                                        st.markdown("---")
                                        st.markdown("**🔄 Model Compatibility Matrix:**")

                                        from cortex_engine.utils.embedding_validator import KNOWN_MODEL_DIMENSIONS

                                        compat_data = []
                                        for model_name, dim in sorted(KNOWN_MODEL_DIMENSIONS.items()):
                                            is_compatible = dim == stored_dim
                                            status = "✅ Compatible" if is_compatible else "❌ Incompatible"
                                            short_name = model_name.split("/")[-1]
                                            compat_data.append(
                                                {
                                                    "Model": short_name,
                                                    "Dimensions": dim,
                                                    "Status": status,
                                                }
                                            )

                                        import pandas as pd

                                        df = pd.DataFrame(compat_data)
                                        st.dataframe(df, use_container_width=True, hide_index=True)

                                        st.markdown("---")
                                        st.markdown("**🔮 Qwen3-VL Compatibility:**")

                                        qwen_2b_dim = 2048
                                        qwen_8b_dim = 4096

                                        if stored_dim == qwen_2b_dim:
                                            st.success("✅ Your database is compatible with **Qwen3-VL-Embedding-2B**")
                                            st.info(
                                                "You can enable Qwen3-VL with: `export QWEN3_VL_ENABLED=true QWEN3_VL_MODEL_SIZE=2B`"
                                            )
                                        elif stored_dim == qwen_8b_dim:
                                            st.success("✅ Your database is compatible with **Qwen3-VL-Embedding-8B**")
                                            st.info(
                                                "You can enable Qwen3-VL with: `export QWEN3_VL_ENABLED=true QWEN3_VL_MODEL_SIZE=8B`"
                                            )
                                        else:
                                            st.warning(
                                                f"⚠️ Your database ({stored_dim}D) is **NOT compatible** with Qwen3-VL models"
                                            )
                                            st.markdown(
                                                """
                                            **To use Qwen3-VL, you need to:**
                                            1. Back up your knowledge base (optional)
                                            2. Delete the existing database
                                            3. Re-ingest all documents with Qwen3-VL enabled

                                            **Qwen3-VL Model Options:**
                                            - **2B Model**: 2048 dimensions, ~5GB VRAM
                                            - **8B Model**: 4096 dimensions, ~16GB VRAM

                                            **Note:** The neural reranker works with ANY embedding model - you don't need to re-ingest to use reranking!
                                            """
                                            )
                                    else:
                                        st.warning("Database exists but no embeddings found")
                                else:
                                    st.warning("Database is empty (0 documents)")

                            except Exception as coll_error:
                                st.error(f"Could not access collection '{collection_name}': {coll_error}")

            except Exception as e:
                st.error(f"Inspection failed: {e}")
                logger.error(f"Database inspection error: {e}", exc_info=True)

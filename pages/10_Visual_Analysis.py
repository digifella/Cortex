"""
Visual Analysis Dashboard
Version: v1.0.0
Date: 2025-08-25

Advanced visual processing and analysis tools using enhanced LLaVA integration.
Provides comprehensive image analysis, OCR, chart analysis, and visual search capabilities.
"""

import streamlit as st
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import tempfile
import shutil

# Configure page
st.set_page_config(
    page_title="Visual Analysis Dashboard", 
    page_icon="👁️",
    layout="wide"
)

# Page configuration
PAGE_VERSION = None

# Import Cortex modules
try:
    from cortex_engine.visual_search import VisualSearchEngine, analyze_image, extract_text_from_image, analyze_chart_image
    from cortex_engine.config import VLM_MODEL
    from cortex_engine.version_config import VERSION_STRING
    from cortex_engine.utils.model_checker import model_checker
    from cortex_engine.utils.path_utils import process_drag_drop_path
except ImportError as e:
    st.error(f"Failed to import Cortex modules: {e}")
    st.stop()

PAGE_VERSION = VERSION_STRING

# Initialize session state
if "visual_analysis" not in st.session_state:
    st.session_state.visual_analysis = {
        "uploaded_files": [],
        "analysis_results": [],
        "selected_analysis_type": "comprehensive",
        "batch_processing": False,
        "comparison_mode": False,
        "visual_search_history": []
    }

def display_header():
    """Display page header with status information"""
    st.title("👁️ Visual Analysis Dashboard")
    st.caption(f"Version: {PAGE_VERSION} • Advanced Image Processing & Analysis")
    
    # Check model availability
    model_available = False
    try:
        available_models = model_checker.get_available_models()
        visual_models = [m for m in available_models if 'llava' in m.lower() or 'moondream' in m.lower()]
        
        if visual_models:
            model_available = True
            st.success(f"✅ Visual models available: {', '.join(visual_models[:2])}{'...' if len(visual_models) > 2 else ''}")
        else:
            st.warning(f"⚠️ No visual models detected. Please install: `ollama pull {VLM_MODEL}`")
            
    except Exception as e:
        st.error(f"❌ Cannot check model status: {e}")
    
    return model_available

def display_analysis_options():
    """Display analysis type selection"""
    st.subheader("🎯 Analysis Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            options=["comprehensive", "ocr", "charts", "technical", "creative"],
            format_func=lambda x: {
                "comprehensive": "🔍 Comprehensive Analysis",
                "ocr": "📝 Text Extraction (OCR)",
                "charts": "📊 Chart & Data Analysis", 
                "technical": "⚙️ Technical Analysis",
                "creative": "🎨 Creative & Design Analysis"
            }[x],
            key="analysis_type_select"
        )
        st.session_state.visual_analysis["selected_analysis_type"] = analysis_type
    
    with col2:
        batch_mode = st.checkbox(
            "🔄 Batch Processing Mode",
            value=st.session_state.visual_analysis["batch_processing"],
            help="Process multiple images simultaneously"
        )
        st.session_state.visual_analysis["batch_processing"] = batch_mode
    
    with col3:
        comparison_mode = st.checkbox(
            "⚖️ Image Comparison Mode", 
            value=st.session_state.visual_analysis["comparison_mode"],
            help="Compare two images side-by-side"
        )
        st.session_state.visual_analysis["comparison_mode"] = comparison_mode
    
    return analysis_type, batch_mode, comparison_mode

def handle_file_upload():
    """Handle file upload and management"""
    st.subheader("📁 Image Upload & Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Images for Analysis",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff', 'tif', 'svg', 'ico'],
        accept_multiple_files=True,
        help="Supported formats: PNG, JPG, GIF, BMP, WebP, TIFF, SVG, ICO"
    )
    
    # Process uploaded files
    processed_files = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save file temporarily
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            processed_files.append({
                "name": uploaded_file.name,
                "path": temp_path,
                "size": uploaded_file.size,
                "type": uploaded_file.type
            })
    
    # Display current files
    if processed_files:
        st.success(f"✅ {len(processed_files)} file(s) uploaded successfully")
        
        # Show file details in expandable section
        with st.expander("📋 Uploaded Files Details"):
            for i, file_info in enumerate(processed_files):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.text(f"📄 {file_info['name']}")
                with col2:
                    st.text(f"📏 {file_info['size']:,} bytes")
                with col3:
                    st.text(f"🏷️ {file_info['type']}")
                with col4:
                    if st.button(f"👁️ Preview", key=f"preview_{i}"):
                        st.image(file_info['path'], caption=file_info['name'], width=300)
    
    st.session_state.visual_analysis["uploaded_files"] = processed_files
    return processed_files

def perform_visual_analysis(files: List[Dict], analysis_type: str, batch_mode: bool):
    """Perform visual analysis on uploaded files"""
    if not files:
        st.warning("⚠️ No files uploaded for analysis")
        return []
    
    st.subheader("🔄 Analysis in Progress")
    
    results = []
    visual_engine = VisualSearchEngine()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, file_info in enumerate(files):
            # Update progress
            progress = (i + 1) / len(files)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {file_info['name']} ({i+1}/{len(files)})...")
            
            # Perform analysis
            result = visual_engine.analyze_image_with_context(
                file_info['path'], 
                analysis_type
            )
            
            if result.get('success'):
                result['file_info'] = file_info
                results.append(result)
                st.success(f"✅ Completed: {file_info['name']}")
            else:
                st.error(f"❌ Failed: {file_info['name']} - {result.get('error', 'Unknown error')}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.5)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        return []
    
    st.session_state.visual_analysis["analysis_results"] = results
    return results

def display_analysis_results(results: List[Dict]):
    """Display analysis results with interactive features"""
    if not results:
        return
    
    st.subheader("📊 Analysis Results")
    st.success(f"✅ Successfully analyzed {len(results)} image(s)")
    
    # Results overview
    with st.expander("📈 Results Overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Images Analyzed", len(results))
        with col2:
            avg_desc_length = sum(len(r.get('description', '')) for r in results) // len(results)
            st.metric("Avg Description Length", f"{avg_desc_length} chars")
        with col3:
            model_used = results[0].get('model_used', 'Unknown')
            st.metric("Model Used", model_used)
    
    # Individual results
    for i, result in enumerate(results):
        file_info = result.get('file_info', {})
        
        st.markdown(f"### 📄 {file_info.get('name', f'Image {i+1}')}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display image
            try:
                st.image(
                    file_info.get('path'),
                    caption=f"{file_info.get('name')} ({file_info.get('size', 0):,} bytes)",
                    width=300
                )
            except Exception as e:
                st.error(f"Cannot display image: {e}")
        
        with col2:
            # Display analysis
            analysis_type = result.get('analysis_type', 'comprehensive')
            st.markdown(f"**Analysis Type:** {analysis_type.title()}")
            st.markdown(f"**Model:** {result.get('model_used', 'Unknown')}")
            
            # Analysis content
            description = result.get('description', 'No analysis available')
            st.markdown("**Analysis Results:**")
            st.text_area(
                f"Analysis for {file_info.get('name', 'image')}",
                value=description,
                height=200,
                key=f"analysis_{i}",
                label_visibility="collapsed"
            )
            
            # Additional structured data
            if 'extracted_text' in result:
                with st.expander("📝 Extracted Text"):
                    for text in result['extracted_text']:
                        st.text(text)
            
            if 'chart_data' in result:
                with st.expander("📊 Chart Data"):
                    st.json(result['chart_data'])
        
        st.markdown("---")

def handle_image_comparison():
    """Handle image comparison functionality"""
    st.subheader("⚖️ Image Comparison")
    
    files = st.session_state.visual_analysis["uploaded_files"]
    if len(files) < 2:
        st.warning("⚠️ Please upload at least 2 images for comparison")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Select First Image:**")
        image1_idx = st.selectbox(
            "First image",
            range(len(files)),
            format_func=lambda x: files[x]['name'],
            key="comparison_image1"
        )
        if image1_idx is not None:
            st.image(files[image1_idx]['path'], caption=files[image1_idx]['name'], width=250)
    
    with col2:
        st.markdown("**Select Second Image:**")
        image2_idx = st.selectbox(
            "Second image",
            range(len(files)),
            format_func=lambda x: files[x]['name'],
            key="comparison_image2"
        )
        if image2_idx is not None and image2_idx != image1_idx:
            st.image(files[image2_idx]['path'], caption=files[image2_idx]['name'], width=250)
    
    # Perform comparison
    if st.button("🔍 Compare Images", type="primary"):
        if image1_idx == image2_idx:
            st.error("❌ Please select two different images for comparison")
            return
        
        with st.spinner("Comparing images..."):
            visual_engine = VisualSearchEngine()
            comparison_result = visual_engine.compare_images(
                files[image1_idx]['path'],
                files[image2_idx]['path']
            )
        
        if comparison_result.get('success'):
            st.subheader("📊 Comparison Results")
            st.text_area(
                "Detailed Comparison",
                value=comparison_result['comparison'],
                height=300
            )
        else:
            st.error(f"❌ Comparison failed: {comparison_result.get('error')}")

def display_export_options(results: List[Dict]):
    """Display export and save options"""
    if not results:
        return
    
    st.subheader("💾 Export & Save Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Copy All Results"):
            combined_results = ""
            for i, result in enumerate(results, 1):
                file_name = result.get('file_info', {}).get('name', f'Image {i}')
                description = result.get('description', 'No analysis')
                combined_results += f"=== {file_name} ===\n{description}\n\n"
            
            # Store in session state for copying (Streamlit doesn't support clipboard directly)
            st.session_state['copy_text'] = combined_results
            st.success("✅ Results ready to copy (stored in session)")
    
    with col2:
        if st.button("📁 Save as JSON"):
            # Create exportable JSON
            export_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_type": st.session_state.visual_analysis["selected_analysis_type"],
                "model_used": VLM_MODEL,
                "total_images": len(results),
                "results": []
            }
            
            for result in results:
                export_item = {
                    "file_name": result.get('file_info', {}).get('name'),
                    "file_size": result.get('file_info', {}).get('size'),
                    "analysis_type": result.get('analysis_type'),
                    "description": result.get('description'),
                    "success": result.get('success')
                }
                
                # Add structured data if available
                if 'extracted_text' in result:
                    export_item['extracted_text'] = result['extracted_text']
                if 'chart_data' in result:
                    export_item['chart_data'] = result['chart_data']
                
                export_data["results"].append(export_item)
            
            # Convert to JSON string
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            # Provide download
            st.download_button(
                label="⬇️ Download JSON Results",
                data=json_str,
                file_name=f"visual_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📊 Generate Summary Report"):
            # Create summary report
            summary = f"""# Visual Analysis Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Analysis Type: {st.session_state.visual_analysis["selected_analysis_type"].title()}
Model Used: {VLM_MODEL}
Total Images: {len(results)}

## Summary Statistics
- Successfully analyzed: {sum(1 for r in results if r.get('success'))} images
- Average description length: {sum(len(r.get('description', '')) for r in results) // len(results)} characters
- File formats: {', '.join(set(r.get('file_info', {}).get('type', 'unknown') for r in results))}

## Individual Results
"""
            
            for i, result in enumerate(results, 1):
                file_info = result.get('file_info', {})
                summary += f"\n### {i}. {file_info.get('name', f'Image {i}')}\n"
                summary += f"- Size: {file_info.get('size', 0):,} bytes\n"
                summary += f"- Type: {file_info.get('type', 'unknown')}\n"
                summary += f"- Analysis: {result.get('description', 'No analysis')[:200]}{'...' if len(result.get('description', '')) > 200 else ''}\n"
            
            st.download_button(
                label="⬇️ Download Summary Report",
                data=summary,
                file_name=f"visual_analysis_summary_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

def display_quick_actions():
    """Display quick action buttons and utilities"""
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Clear All Results"):
            st.session_state.visual_analysis["analysis_results"] = []
            st.session_state.visual_analysis["uploaded_files"] = []
            st.success("✅ Results cleared")
            st.rerun()
    
    with col2:
        if st.button("📈 Show Model Status"):
            try:
                available_models = model_checker.get_available_models()
                visual_models = [m for m in available_models if 'llava' in m.lower() or 'moondream' in m.lower()]
                
                if visual_models:
                    st.success(f"Visual models: {', '.join(visual_models)}")
                else:
                    st.warning("No visual models found")
                    
            except Exception as e:
                st.error(f"Cannot check models: {e}")
    
    with col3:
        if st.button("🎯 Analysis Examples"):
            st.info("""
            **Analysis Type Examples:**
            - **Comprehensive**: General image understanding
            - **OCR**: Extract text from documents/screenshots  
            - **Charts**: Analyze graphs, charts, data visualizations
            - **Technical**: Engineering diagrams, technical docs
            - **Creative**: Design analysis, branding, aesthetics
            """)
    
    with col4:
        if st.button("💡 Tips & Best Practices"):
            st.info("""
            **Best Practices:**
            - Use high-resolution images for better text extraction
            - Choose specific analysis types for targeted insights
            - Batch mode is efficient for similar image types
            - Compare mode works best with related images
            - Export results for documentation and sharing
            """)

def main():
    """Main application logic"""
    # Display header and check model availability
    model_available = display_header()
    
    if not model_available:
        st.warning("⚠️ Visual analysis requires a vision language model. Please install one to continue.")
        st.info(f"""
        **To install visual models:**
        ```bash
        # Standard model (recommended)
        ollama pull llava:7b
        
        # Lightweight model (faster)
        ollama pull moondream
        
        # Premium model (higher accuracy) 
        ollama pull llava:13b
        ```
        """)
        return
    
    # Display analysis configuration
    analysis_type, batch_mode, comparison_mode = display_analysis_options()
    
    # Handle file uploads
    uploaded_files = handle_file_upload()
    
    # Main analysis section
    if uploaded_files:
        
        if comparison_mode and not batch_mode:
            # Image comparison mode
            handle_image_comparison()
        else:
            # Standard analysis mode
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("🔍 Visual Analysis")
            
            with col2:
                if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                    results = perform_visual_analysis(uploaded_files, analysis_type, batch_mode)
                    if results:
                        st.success(f"✅ Analysis complete! {len(results)} images processed.")
            
            # Display results if available
            results = st.session_state.visual_analysis.get("analysis_results", [])
            if results:
                display_analysis_results(results)
                display_export_options(results)
    
    else:
        # Show welcome message and instructions
        st.markdown("""
        ## 🎯 Welcome to Visual Analysis Dashboard
        
        This advanced tool provides comprehensive image analysis using state-of-the-art vision language models.
        
        ### 🚀 Features:
        - **Multi-format Support**: PNG, JPG, GIF, BMP, WebP, TIFF, SVG, ICO
        - **Specialized Analysis**: OCR, chart analysis, technical documentation, creative content
        - **Batch Processing**: Analyze multiple images simultaneously  
        - **Image Comparison**: Side-by-side analysis and comparison
        - **Export Options**: JSON, Markdown, and summary reports
        
        ### 📁 Getting Started:
        1. **Upload images** using the file uploader above
        2. **Select analysis type** based on your needs
        3. **Configure options** (batch mode, comparison mode)
        4. **Start analysis** and review results
        5. **Export or save** your findings
        
        ### 💡 Use Cases:
        - **Document Processing**: Extract text from scanned documents
        - **Data Analysis**: Analyze charts, graphs, and visualizations  
        - **Technical Documentation**: Process engineering diagrams and schematics
        - **Content Analysis**: Understand images for knowledge management
        - **Quality Assurance**: Compare images for consistency checking
        """)
    
    # Quick actions section
    display_quick_actions()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666; font-size: 0.85em;'>"
        f"Visual Analysis Dashboard v{PAGE_VERSION} • Enhanced LLaVA Integration • "
        f"Powered by {VLM_MODEL}"
        f"</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass

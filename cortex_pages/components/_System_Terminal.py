"""
System Terminal (Redirects to Maintenance)
Version: v1.1.0
Date: 2025-08-27

This page has been consolidated into the Maintenance page for better organization.
All system terminal functionality is now available under Maintenance > Terminal tab.
"""

import streamlit as st
from cortex_engine.version_config import VERSION_STRING

# Configure page
st.set_page_config(
    page_title="System Terminal", 
    page_icon="💻",
    layout="wide"
)

# Page configuration
PAGE_VERSION = VERSION_STRING

def main():
    """Main function displaying redirect information"""
    st.title("💻 System Terminal")
    st.caption(f"Version: {PAGE_VERSION} • Moved to Maintenance Page")
    
    st.info("""
    🔄 **This page has been moved!**
    
    The System Terminal functionality has been consolidated into the **Maintenance** page 
    for better organization and easier access to all administrative functions.
    """)
    
    st.markdown("""
    ### What's Available in Maintenance:
    
    - 💻 **System Terminal** - Safe command execution interface
    - ⚡ **Quick Actions** - Common system commands (check models, system status, disk usage)
    - 🗄️ **Database Maintenance** - Clear logs, delete knowledge base, recovery tools
    - ⚙️ **Setup Management** - Reset installation state
    - 💾 **Backup Management** - Create and restore backups
    """)
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔧 Go to Maintenance Page", use_container_width=True, type="primary"):
            st.switch_page("cortex_pages/6_Maintenance.py")
    
    st.divider()
    
    st.markdown("""
    ### Why was this moved?
    
    - **Better Organization**: All administrative functions are now in one place
    - **Enhanced Security**: Centralized access control for system operations  
    - **Improved UX**: Easier to find and use maintenance tools
    - **Future Expansion**: Room for additional admin features like password protection
    """)

if __name__ == "__main__":
    main()

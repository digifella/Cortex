# ## File: cortex_engine/ui_theme.py
# Version: v4.10.3
# Date: 2025-12-24
# Purpose: Centralized UI theme and styling for Cortex Suite with distinctive editorial design

"""
Cortex Suite UI Theme System
============================
A refined, editorial-inspired design system for professional knowledge management.

Design Philosophy:
- **Editorial Clarity**: Inspired by high-end publications and research journals
- **Sophisticated Typography**: Distinctive font choices that convey professionalism
- **Refined Color Palette**: Deep, authoritative tones with intelligent accents
- **Purposeful Hierarchy**: Clear visual structure guiding user attention
"""

# Theme Configuration
THEME_CONFIG = {
    "name": "Cortex Editorial",
    "version": "1.0.0",
    "description": "Professional editorial design for knowledge management"
}

# Typography System
# Using system font stacks for reliability with distinctive backup choices
TYPOGRAPHY = {
    "display": {
        "family": "'Playfair Display', 'Georgia', 'Times New Roman', serif",
        "weight": "700",
        "letter_spacing": "-0.02em",
        "line_height": "1.1"
    },
    "heading": {
        "family": "'Inter Tight', 'SF Pro Display', -apple-system, system-ui, sans-serif",
        "weight": "600",
        "letter_spacing": "-0.01em",
        "line_height": "1.3"
    },
    "body": {
        "family": "'Source Sans 3', 'Segoe UI', 'Roboto', sans-serif",
        "weight": "400",
        "letter_spacing": "0.01em",
        "line_height": "1.6"
    },
    "mono": {
        "family": "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
        "weight": "400",
        "letter_spacing": "0",
        "line_height": "1.5"
    }
}

# Color System - Deep, professional palette
COLORS = {
    # Primary - Deep Navy Blue (authoritative, professional)
    "primary": {
        "900": "#0A1628",  # Deepest navy
        "800": "#152238",
        "700": "#1E2F47",
        "600": "#2A3F5F",
        "500": "#374F77",  # Main primary
        "400": "#4D6B99",
        "300": "#6A89B8",
        "200": "#91ACD1",
        "100": "#C3D5E8",
        "50": "#E8F0F8"
    },

    # Secondary - Warm Terracotta (approachable, creative)
    "secondary": {
        "900": "#3D1F14",
        "800": "#5C2E1F",
        "700": "#7A3D2A",
        "600": "#984C35",
        "500": "#B6704F",  # Main secondary
        "400": "#C98B6C",
        "300": "#D7A68E",
        "200": "#E5C4B3",
        "100": "#F2E2D9",
        "50": "#FAF4F0"
    },

    # Accent - Sage Green (calm, intelligent)
    "accent": {
        "900": "#1C2920",
        "800": "#2A3F30",
        "700": "#38543F",
        "600": "#476A4E",
        "500": "#5B8A66",  # Main accent
        "400": "#76A381",
        "300": "#95BA9D",
        "200": "#B7D1BD",
        "100": "#D9E8DD",
        "50": "#F0F7F2"
    },

    # Neutrals - Warm grays
    "neutral": {
        "900": "#1A1816",
        "800": "#2D2926",
        "700": "#423F3B",
        "600": "#5A5651",
        "500": "#726D66",
        "400": "#91897F",
        "300": "#AFA99F",
        "200": "#CCC8C1",
        "100": "#E8E6E2",
        "50": "#F7F6F4",
        "white": "#FFFFFF"
    },

    # Semantic colors
    "success": "#5B8A66",
    "warning": "#B6704F",
    "error": "#C44536",
    "info": "#374F77",

    # Backgrounds
    "bg": {
        "primary": "#FFFFFF",
        "secondary": "#F7F6F4",
        "tertiary": "#E8E6E2",
        "elevated": "#FFFFFF"
    }
}

# Spacing System (8px base unit)
SPACING = {
    "xs": "0.25rem",   # 4px
    "sm": "0.5rem",    # 8px
    "md": "1rem",      # 16px
    "lg": "1.5rem",    # 24px
    "xl": "2rem",      # 32px
    "2xl": "3rem",     # 48px
    "3xl": "4rem",     # 64px
    "4xl": "6rem"      # 96px
}

# Border Radius
RADIUS = {
    "none": "0",
    "sm": "0.25rem",   # 4px
    "md": "0.5rem",    # 8px
    "lg": "0.75rem",   # 12px
    "xl": "1rem",      # 16px
    "full": "9999px"
}

# Shadows
SHADOWS = {
    "sm": "0 1px 2px 0 rgba(26, 24, 22, 0.05)",
    "md": "0 4px 6px -1px rgba(26, 24, 22, 0.1), 0 2px 4px -1px rgba(26, 24, 22, 0.06)",
    "lg": "0 10px 15px -3px rgba(26, 24, 22, 0.1), 0 4px 6px -2px rgba(26, 24, 22, 0.05)",
    "xl": "0 20px 25px -5px rgba(26, 24, 22, 0.1), 0 10px 10px -5px rgba(26, 24, 22, 0.04)",
    "inner": "inset 0 2px 4px 0 rgba(26, 24, 22, 0.06)"
}


def get_custom_css():
    """
    Generate comprehensive custom CSS for Streamlit apps
    Returns CSS string with refined editorial styling
    """

    css = f"""
    <style>
    /* ============================================
       CORTEX SUITE EDITORIAL THEME
       Refined, professional design system
       ============================================ */

    /* Import refined fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ============================================
       ROOT VARIABLES
       ============================================ */
    :root {{
        /* Colors */
        --color-primary: {COLORS['primary']['500']};
        --color-primary-dark: {COLORS['primary']['700']};
        --color-primary-light: {COLORS['primary']['300']};
        --color-secondary: {COLORS['secondary']['500']};
        --color-accent: {COLORS['accent']['500']};

        --color-text-primary: {COLORS['neutral']['900']};
        --color-text-secondary: {COLORS['neutral']['600']};
        --color-text-tertiary: {COLORS['neutral']['400']};

        --color-bg-primary: {COLORS['bg']['primary']};
        --color-bg-secondary: {COLORS['bg']['secondary']};
        --color-bg-elevated: {COLORS['bg']['elevated']};

        --color-border: {COLORS['neutral']['200']};
        --color-border-focus: {COLORS['primary']['400']};

        /* Typography */
        --font-display: {TYPOGRAPHY['display']['family']};
        --font-heading: {TYPOGRAPHY['heading']['family']};
        --font-body: {TYPOGRAPHY['body']['family']};
        --font-mono: {TYPOGRAPHY['mono']['family']};

        /* Spacing */
        --space-xs: {SPACING['xs']};
        --space-sm: {SPACING['sm']};
        --space-md: {SPACING['md']};
        --space-lg: {SPACING['lg']};
        --space-xl: {SPACING['xl']};
        --space-2xl: {SPACING['2xl']};

        /* Radius & Shadows */
        --radius-sm: {RADIUS['sm']};
        --radius-md: {RADIUS['md']};
        --radius-lg: {RADIUS['lg']};
        --shadow-sm: {SHADOWS['sm']};
        --shadow-md: {SHADOWS['md']};
        --shadow-lg: {SHADOWS['lg']};
    }}

    /* ============================================
       GLOBAL STYLES
       ============================================ */

    /* Main app container */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}

    /* Base typography */
    .main {{
        font-family: var(--font-body);
        color: var(--color-text-primary);
        background-color: var(--color-bg-secondary);
        letter-spacing: 0.01em;
    }}

    /* ============================================
       TYPOGRAPHY ENHANCEMENTS
       ============================================ */

    /* Main page title */
    .main h1 {{
        font-family: var(--font-heading);
        font-weight: 700;
        font-size: 2.5rem;
        color: var(--color-primary-dark);
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-bottom: 0.5rem;
        margin-top: 0;
    }}

    /* Section headers */
    .main h2 {{
        font-family: var(--font-heading);
        font-weight: 600;
        font-size: 1.75rem;
        color: var(--color-primary);
        letter-spacing: -0.01em;
        line-height: 1.3;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid {COLORS['neutral']['100']};
        padding-bottom: 0.5rem;
    }}

    /* Subsection headers */
    .main h3 {{
        font-family: var(--font-heading);
        font-weight: 600;
        font-size: 1.25rem;
        color: var(--color-text-primary);
        letter-spacing: -0.005em;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }}

    /* Body text */
    .main p {{
        font-family: var(--font-body);
        font-size: 1rem;
        line-height: 1.6;
        color: var(--color-text-secondary);
        margin-bottom: 1rem;
    }}

    /* Captions */
    .main .stCaption {{
        font-size: 0.875rem;
        color: var(--color-text-tertiary);
        font-weight: 400;
    }}

    /* ============================================
       BUTTONS & INTERACTIVE ELEMENTS
       ============================================ */

    /* Primary buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, {COLORS['primary']['600']} 0%, {COLORS['primary']['700']} 100%);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-family: var(--font-heading);
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.01em;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {{
        background: linear-gradient(135deg, {COLORS['primary']['700']} 0%, {COLORS['primary']['800']} 100%);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }}

    /* Secondary buttons */
    .stButton > button[kind="secondary"],
    .stButton > button {{
        background-color: white;
        color: var(--color-primary);
        border: 1.5px solid {COLORS['neutral']['200']};
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-family: var(--font-heading);
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .stButton > button[kind="secondary"]:hover,
    .stButton > button:hover {{
        border-color: var(--color-primary);
        background-color: {COLORS['primary']['50']};
        box-shadow: var(--shadow-sm);
    }}

    /* ============================================
       INPUT FIELDS
       ============================================ */

    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {{
        font-family: var(--font-body);
        border: 1.5px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 0.75rem;
        background-color: white;
        color: var(--color-text-primary);
        transition: all 0.2s ease;
    }}

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {{
        border-color: var(--color-border-focus);
        box-shadow: 0 0 0 3px {COLORS['primary']['100']};
        outline: none;
    }}

    /* Input labels */
    .stTextInput > label,
    .stTextArea > label,
    .stSelectbox > label,
    .stMultiSelect > label {{
        font-family: var(--font-heading);
        font-weight: 500;
        font-size: 0.9rem;
        color: var(--color-text-primary);
        margin-bottom: 0.5rem;
    }}

    /* ============================================
       CONTAINERS & CARDS
       ============================================ */

    /* Container with border */
    .element-container div[data-testid="stExpander"],
    [data-testid="stContainer"] {{
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        background-color: var(--color-bg-elevated);
        box-shadow: var(--shadow-sm);
        margin-bottom: var(--space-md);
    }}

    /* Expander styling */
    .streamlit-expanderHeader {{
        font-family: var(--font-heading);
        font-weight: 600;
        font-size: 1.1rem;
        color: var(--color-primary-dark);
        background-color: {COLORS['neutral']['50']};
        border-radius: var(--radius-md);
        padding: 0.75rem 1rem;
    }}

    .streamlit-expanderHeader:hover {{
        background-color: {COLORS['primary']['50']};
    }}

    /* ============================================
       METRICS & STATUS
       ============================================ */

    /* Metric containers */
    .stMetric {{
        background-color: white;
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        border: 1px solid var(--color-border);
        box-shadow: var(--shadow-sm);
    }}

    .stMetric > div {{
        font-family: var(--font-heading);
    }}

    .stMetric label {{
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--color-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .stMetric [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--color-primary-dark);
    }}

    /* ============================================
       ALERTS & NOTIFICATIONS
       ============================================ */

    /* Info boxes */
    .stAlert {{
        border-radius: var(--radius-lg);
        border-left-width: 4px;
        padding: var(--space-lg);
        font-family: var(--font-body);
    }}

    div[data-baseweb="notification"][kind="info"] {{
        background-color: {COLORS['primary']['50']};
        border-left-color: {COLORS['primary']['500']};
    }}

    div[data-baseweb="notification"][kind="success"] {{
        background-color: {COLORS['accent']['50']};
        border-left-color: {COLORS['accent']['600']};
    }}

    div[data-baseweb="notification"][kind="warning"] {{
        background-color: {COLORS['secondary']['50']};
        border-left-color: {COLORS['secondary']['600']};
    }}

    div[data-baseweb="notification"][kind="error"] {{
        background-color: #FEF2F2;
        border-left-color: {COLORS['error']};
    }}

    /* ============================================
       SIDEBAR STYLING
       ============================================ */

    .css-1d391kg,
    [data-testid="stSidebar"] {{
        background-color: var(--color-primary-dark);
        padding-top: 2rem;
    }}

    .css-1d391kg .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown {{
        color: {COLORS['neutral']['100']};
    }}

    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: white;
        font-family: var(--font-heading);
    }}

    /* Sidebar buttons */
    .css-1d391kg .stButton > button,
    [data-testid="stSidebar"] .stButton > button {{
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}

    .css-1d391kg .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.3);
    }}

    /* ============================================
       TABLES & DATA DISPLAY
       ============================================ */

    .dataframe {{
        font-family: var(--font-body);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        overflow: hidden;
    }}

    .dataframe thead th {{
        background-color: {COLORS['neutral']['100']};
        color: var(--color-text-primary);
        font-family: var(--font-heading);
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        padding: 0.75rem;
    }}

    .dataframe tbody td {{
        padding: 0.75rem;
        border-bottom: 1px solid {COLORS['neutral']['100']};
    }}

    /* ============================================
       CODE BLOCKS
       ============================================ */

    code {{
        font-family: var(--font-mono);
        background-color: {COLORS['neutral']['100']};
        color: {COLORS['primary']['800']};
        padding: 0.2em 0.4em;
        border-radius: var(--radius-sm);
        font-size: 0.9em;
    }}

    pre {{
        font-family: var(--font-mono);
        background-color: {COLORS['neutral']['900']};
        color: {COLORS['neutral']['100']};
        padding: var(--space-lg);
        border-radius: var(--radius-lg);
        overflow-x: auto;
        line-height: 1.5;
    }}

    /* ============================================
       PROGRESS INDICATORS
       ============================================ */

    .stProgress > div > div > div {{
        background-color: {COLORS['primary']['500']};
        height: 8px;
        border-radius: var(--radius-md);
    }}

    /* ============================================
       DIVIDERS
       ============================================ */

    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(to right,
            transparent 0%,
            {COLORS['neutral']['200']} 20%,
            {COLORS['neutral']['200']} 80%,
            transparent 100%);
        margin: var(--space-xl) 0;
    }}

    /* ============================================
       SCROLLBAR STYLING
       ============================================ */

    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {COLORS['neutral']['100']};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {COLORS['neutral']['300']};
        border-radius: var(--radius-md);
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['neutral']['400']};
    }}

    /* ============================================
       UTILITY CLASSES
       ============================================ */

    .cortex-card {{
        background: white;
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        box-shadow: var(--shadow-sm);
        margin-bottom: var(--space-md);
    }}

    .cortex-section-header {{
        display: flex;
        align-items: center;
        gap: var(--space-sm);
        padding: var(--space-md) 0;
        border-bottom: 2px solid {COLORS['neutral']['100']};
        margin-bottom: var(--space-lg);
    }}

    .cortex-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: {COLORS['primary']['100']};
        color: {COLORS['primary']['700']};
        font-family: var(--font-heading);
        font-size: 0.875rem;
        font-weight: 500;
        border-radius: var(--radius-full);
        letter-spacing: 0.025em;
    }}

    /* ============================================
       RESPONSIVE DESIGN
       ============================================ */

    @media (max-width: 768px) {{
        .main .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .main h1 {{
            font-size: 2rem;
        }}

        .main h2 {{
            font-size: 1.5rem;
        }}
    }}

    /* ============================================
       ANIMATIONS
       ============================================ */

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .main > div {{
        animation: fadeIn 0.3s ease-out;
    }}

    </style>
    """

    return css


def apply_theme():
    """
    Apply the Cortex Editorial theme to a Streamlit page.
    Call this function at the top of your Streamlit page.
    """
    import streamlit as st
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def section_header(icon: str, title: str, subtitle: str = ""):
    """
    Create a styled section header with icon and optional subtitle.

    Args:
        icon: Emoji or icon character
        title: Section title
        subtitle: Optional subtitle/description
    """
    import streamlit as st

    header_html = f"""
    <div class="cortex-section-header">
        <span style="font-size: 1.5rem;">{icon}</span>
        <div>
            <h2 style="margin: 0; border: none; padding: 0;">{title}</h2>
            {f'<p style="margin: 0; color: var(--color-text-secondary); font-size: 0.9rem;">{subtitle}</p>' if subtitle else ''}
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", icon: str = ""):
    """
    Create a styled metric card.

    Args:
        label: Metric label
        value: Metric value
        delta: Optional change indicator
        icon: Optional emoji icon
    """
    import streamlit as st

    card_html = f"""
    <div class="cortex-card" style="text-align: center;">
        {f'<div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>' if icon else ''}
        <div style="font-size: 0.875rem; color: var(--color-text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 2rem; font-weight: 700; color: var(--color-primary-dark); font-family: var(--font-heading);">{value}</div>
        {f'<div style="font-size: 0.875rem; color: var(--color-accent); margin-top: 0.25rem;">{delta}</div>' if delta else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def status_badge(text: str, status_type: str = "info"):
    """
    Create a status badge.

    Args:
        text: Badge text
        status_type: Type of status (info, success, warning, error)
    """
    import streamlit as st

    color_map = {
        "info": (COLORS['primary']['100'], COLORS['primary']['700']),
        "success": (COLORS['accent']['100'], COLORS['accent']['700']),
        "warning": (COLORS['secondary']['100'], COLORS['secondary']['700']),
        "error": ("#FEF2F2", COLORS['error'])
    }

    bg, fg = color_map.get(status_type, color_map["info"])

    badge_html = f"""
    <span style="display: inline-block; padding: 0.25rem 0.75rem; background-color: {bg}; color: {fg};
    font-family: var(--font-heading); font-size: 0.875rem; font-weight: 500; border-radius: 9999px;
    letter-spacing: 0.025em;">{text}</span>
    """
    st.markdown(badge_html, unsafe_allow_html=True)

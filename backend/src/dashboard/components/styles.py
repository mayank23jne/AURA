"""Enhanced Styling for AURA Dashboard"""

# Modern Color Palette
COLORS = {
    # Primary
    "primary": "#667eea",
    "primary_dark": "#764ba2",
    "primary_light": "#a8b3f5",

    # Status
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#3b82f6",

    # Neutral
    "bg_primary": "#0f172a",
    "bg_secondary": "#1e293b",
    "bg_card": "#ffffff",
    "text_primary": "#1e293b",
    "text_secondary": "#64748b",
    "border": "#e2e8f0",

    # Chart colors
    "chart_1": "#667eea",
    "chart_2": "#10b981",
    "chart_3": "#f59e0b",
    "chart_4": "#ef4444",
    "chart_5": "#3b82f6",
}

# Enhanced CSS with modern design
ENHANCED_CSS = f"""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Styles */
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Main Container */
    .main {{
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
    }}

    /* Header with Gradient */
    .main-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }}

    .main-header h1 {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }}

    .main-header p {{
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }}

    /* Modern Card Design */
    .metric-card {{
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid {COLORS['border']};
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }}

    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
    }}

    .metric-card-title {{
        font-size: 0.875rem;
        font-weight: 600;
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}

    .metric-card-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {COLORS['text_primary']};
        line-height: 1;
        margin-bottom: 0.5rem;
    }}

    .metric-card-change {{
        font-size: 0.875rem;
        font-weight: 500;
    }}

    .metric-card-change.positive {{
        color: {COLORS['success']};
    }}

    .metric-card-change.negative {{
        color: {COLORS['danger']};
    }}

    /* Glass-morphism Effect */
    .glass-card {{
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
    }}

    /* Status Badges */
    .status-badge {{
        display: inline-block;
        padding: 0.375rem 0.875rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .status-success {{
        background: {COLORS['success']}20;
        color: {COLORS['success']};
    }}

    .status-warning {{
        background: {COLORS['warning']}20;
        color: {COLORS['warning']};
    }}

    .status-danger {{
        background: {COLORS['danger']}20;
        color: {COLORS['danger']};
    }}

    .status-info {{
        background: {COLORS['info']}20;
        color: {COLORS['info']};
    }}

    /* Buttons */
    .btn-primary {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }}

    .btn-primary:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }}

    .btn-secondary {{
        background: white;
        color: {COLORS['primary']};
        border: 2px solid {COLORS['primary']};
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }}

    .btn-secondary:hover {{
        background: {COLORS['primary']};
        color: white;
    }}

    /* Progress Bar */
    .progress-container {{
        background: {COLORS['border']};
        border-radius: 9999px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }}

    .progress-bar {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
        height: 100%;
        border-radius: 9999px;
        transition: width 0.3s ease;
    }}

    /* Alert/Finding Cards */
    .finding-card {{
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        border-left: 4px solid;
        background: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }}

    .finding-critical {{
        border-left-color: {COLORS['danger']};
        background: {COLORS['danger']}08;
    }}

    .finding-high {{
        border-left-color: {COLORS['warning']};
        background: {COLORS['warning']}08;
    }}

    .finding-medium {{
        border-left-color: {COLORS['info']};
        background: {COLORS['info']}08;
    }}

    .finding-low {{
        border-left-color: {COLORS['success']};
        background: {COLORS['success']}08;
    }}

    /* Recommendation Cards */
    .recommendation-card {{
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid {COLORS['primary']};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }}

    .recommendation-card:hover {{
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }}

    /* Wizard Steps */
    .wizard-container {{
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }}

    .wizard-steps {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
        position: relative;
    }}

    .wizard-step {{
        flex: 1;
        text-align: center;
        position: relative;
    }}

    .wizard-step-number {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: {COLORS['border']};
        color: {COLORS['text_secondary']};
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }}

    .wizard-step.active .wizard-step-number {{
        background: {COLORS['primary']};
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }}

    .wizard-step.completed .wizard-step-number {{
        background: {COLORS['success']};
        color: white;
    }}

    /* Empty State */
    .empty-state {{
        text-align: center;
        padding: 4rem 2rem;
        background: white;
        border-radius: 16px;
        border: 2px dashed {COLORS['border']};
    }}

    .empty-state-icon {{
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.3;
    }}

    .empty-state-title {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {COLORS['text_primary']};
        margin-bottom: 0.5rem;
    }}

    .empty-state-description {{
        font-size: 1rem;
        color: {COLORS['text_secondary']};
        margin-bottom: 1.5rem;
    }}

    /* Loading Skeleton */
    .skeleton {{
        background: linear-gradient(
            90deg,
            #f0f0f0 25%,
            #e0e0e0 50%,
            #f0f0f0 75%
        );
        background-size: 200% 100%;
        animation: skeleton-loading 1.5s infinite;
        border-radius: 8px;
    }}

    @keyframes skeleton-loading {{
        0% {{
            background-position: 200% 0;
        }}
        100% {{
            background-position: -200% 0;
        }}
    }}

    /* Toast Notification */
    .toast {{
        position: fixed;
        top: 20px;
        right: 20px;
        min-width: 300px;
        background: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        border-left: 4px solid;
        animation: slideIn 0.3s ease;
        z-index: 9999;
    }}

    @keyframes slideIn {{
        from {{
            transform: translateX(400px);
            opacity: 0;
        }}
        to {{
            transform: translateX(0);
            opacity: 1;
        }}
    }}

    .toast.success {{
        border-left-color: {COLORS['success']};
    }}

    .toast.error {{
        border-left-color: {COLORS['danger']};
    }}

    .toast.warning {{
        border-left-color: {COLORS['warning']};
    }}

    .toast.info {{
        border-left-color: {COLORS['info']};
    }}

    /* Sidebar Enhancements */
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, {COLORS['primary']}10 0%, transparent 100%);
    }}

    /* Table Improvements */
    .dataframe {{
        border: none !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
    }}

    .dataframe thead tr {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%) !important;
        color: white !important;
    }}

    .dataframe thead th {{
        padding: 1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
    }}

    .dataframe tbody tr:hover {{
        background: {COLORS['primary']}08 !important;
    }}

    /* Responsive */
    @media (max-width: 768px) {{
        .main-header h1 {{
            font-size: 1.75rem;
        }}

        .metric-card-value {{
            font-size: 2rem;
        }}

        .wizard-steps {{
            flex-direction: column;
        }}
    }}
</style>
"""

def get_enhanced_css():
    """Return the enhanced CSS styling"""
    return ENHANCED_CSS


def get_score_class(score: float) -> str:
    """Get CSS class based on compliance score."""
    if score >= 0.9:
        return "score-excellent"
    elif score >= 0.75:
        return "score-good"
    elif score >= 0.6:
        return "score-fair"
    else:
        return "score-poor"

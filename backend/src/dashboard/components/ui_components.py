"""
Enhanced UI Components for AURA Dashboard
Provides reusable components for navigation, loading states, empty states, and notifications
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime


def render_enhanced_sidebar(
    pages: List[Dict[str, str]],
    current_page: Optional[str] = None,
    show_stats: bool = True,
    show_health: bool = True
) -> str:
    """
    Render enhanced sidebar with icons, tooltips, and better visual hierarchy.

    Args:
        pages: List of dicts with 'name', 'icon', 'description'
        current_page: Current active page
        show_stats: Whether to show quick stats
        show_health: Whether to show health check

    Returns:
        Selected page name
    """
    with st.sidebar:
        # Logo and title
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <div style="
                    width: 80px;
                    height: 80px;
                    margin: 0 auto 1rem;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
                ">
                    <span style="font-size: 40px;">🛡️</span>
                </div>
                <h2 style="margin: 0; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    AURA Platform
                </h2>
                <p style="margin: 0.5rem 0 0; color: #64748b; font-size: 14px;">
                    AI Security & Compliance
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        page_names = [f"{p['icon']} {p['name']}" for p in pages]
        page_labels = {f"{p['icon']} {p['name']}": p['name'] for p in pages}

        # Determine default index
        if current_page:
            for idx, p in enumerate(pages):
                if p['name'] == current_page:
                    default_index = idx
                    break
        else:
            default_index = 0

        selected = st.radio(
            "Navigation",
            page_names,
            index=default_index,
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Health check
        if show_health:
            render_health_indicator()

        # Quick stats
        if show_stats:
            st.markdown("---")
            render_quick_stats()

        # Footer
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; font-size: 12px; color: #94a3b8;">
                <p>AURA v1.0.0</p>
                <p>© 2025 Agentic Platform</p>
            </div>
        """, unsafe_allow_html=True)

        return page_labels[selected]


def render_health_indicator():
    """Render health status indicator with animation."""
    try:
        # Assuming api_request is available in the main app
        from src.dashboard.api_client import api_request
        health = api_request("/health")

        if "error" not in health:
            st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
                    border-left: 4px solid #10b981;
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                ">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="
                            width: 10px;
                            height: 10px;
                            background: #10b981;
                            border-radius: 50%;
                            display: inline-block;
                            animation: pulse 2s ease-in-out infinite;
                        "></span>
                        <span style="font-weight: 600; color: #10b981;">Platform Online</span>
                    </div>
                </div>
                <style>
                    @keyframes pulse {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.5; }
                    }
                </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
                    border-left: 4px solid #ef4444;
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                ">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="
                            width: 10px;
                            height: 10px;
                            background: #ef4444;
                            border-radius: 50%;
                            display: inline-block;
                        "></span>
                        <span style="font-weight: 600; color: #ef4444;">Platform Offline</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    except:
        pass


def render_quick_stats():
    """Render quick stats section."""
    st.markdown('<p style="font-weight: 600; font-size: 14px; color: #64748b; margin-bottom: 8px;">Quick Stats</p>', unsafe_allow_html=True)

    # Get stats from session state
    audit_count = len(st.session_state.get("audit_history", []))
    report_count = len(st.session_state.get("reports", []))

    st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 8px;">
            <div style="
                background: rgba(102, 126, 234, 0.05);
                padding: 8px 12px;
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span style="color: #64748b; font-size: 13px;">📋 Audits</span>
                <span style="font-weight: 600; color: #667eea; font-size: 16px;">{audit_count}</span>
            </div>
            <div style="
                background: rgba(102, 126, 234, 0.05);
                padding: 8px 12px;
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span style="color: #64748b; font-size: 13px;">📑 Reports</span>
                <span style="font-weight: 600; color: #667eea; font-size: 16px;">{report_count}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def show_loading_state(message: str = "Loading...", spinner_type: str = "dots"):
    """
    Display a loading state with spinner and message.

    Args:
        message: Loading message to display
        spinner_type: Type of loading indicator ('dots', 'bars', 'circle')
    """
    if spinner_type == "skeleton":
        st.markdown("""
            <div class="skeleton" style="
                height: 200px;
                border-radius: 16px;
                margin: 1rem 0;
                background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                background-size: 200% 100%;
                animation: skeleton-loading 1.5s infinite;
            "></div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner(message):
            pass


def show_loading_skeleton(count: int = 3):
    """Display multiple loading skeletons for list items."""
    for i in range(count):
        st.markdown(f"""
            <div class="skeleton" style="
                height: 80px;
                border-radius: 12px;
                margin: 12px 0;
                background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                background-size: 200% 100%;
                animation: skeleton-loading 1.5s infinite;
                animation-delay: {i * 0.1}s;
            "></div>
        """, unsafe_allow_html=True)


def show_empty_state(
    icon: str = "🔍",
    title: str = "No Data Available",
    description: str = "There's nothing to display here yet.",
    action_label: Optional[str] = None,
    action_callback: Optional[callable] = None
):
    """
    Display an empty state when there's no data.

    Args:
        icon: Emoji or icon to display
        title: Main heading
        description: Explanatory text
        action_label: Optional button label
        action_callback: Optional button callback function
    """
    st.markdown(f"""
        <div class="empty-state" style="
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, rgba(118, 75, 162, 0.03) 100%);
            border-radius: 16px;
            margin: 2rem 0;
        ">
            <div class="empty-state-icon" style="
                font-size: 64px;
                margin-bottom: 1.5rem;
                opacity: 0.6;
            ">{icon}</div>
            <div class="empty-state-title" style="
                font-size: 24px;
                font-weight: 600;
                color: #1e293b;
                margin-bottom: 0.5rem;
            ">{title}</div>
            <div class="empty-state-description" style="
                font-size: 16px;
                color: #64748b;
                margin-bottom: 1.5rem;
            ">{description}</div>
        </div>
    """, unsafe_allow_html=True)

    if action_label and action_callback:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_label, key=f"empty_action_{datetime.now().timestamp()}", use_container_width=True):
                action_callback()


def show_toast_notification(
    message: str,
    notification_type: str = "info",
    duration: int = 3,
    position: str = "top-right"
):
    """
    Display a toast notification.

    Args:
        message: Notification message
        notification_type: Type of notification ('success', 'error', 'warning', 'info')
        duration: Display duration in seconds
        position: Position on screen ('top-right', 'top-left', 'bottom-right', 'bottom-left')
    """
    colors = {
        "success": {"bg": "#10b981", "border": "#059669"},
        "error": {"bg": "#ef4444", "border": "#dc2626"},
        "warning": {"bg": "#f59e0b", "border": "#d97706"},
        "info": {"bg": "#3b82f6", "border": "#2563eb"}
    }

    icons = {
        "success": "✓",
        "error": "✕",
        "warning": "⚠",
        "info": "ℹ"
    }

    color_config = colors.get(notification_type, colors["info"])
    icon = icons.get(notification_type, icons["info"])

    # Position styling
    position_styles = {
        "top-right": "top: 20px; right: 20px;",
        "top-left": "top: 20px; left: 20px;",
        "bottom-right": "bottom: 20px; right: 20px;",
        "bottom-left": "bottom: 20px; left: 20px;"
    }

    position_style = position_styles.get(position, position_styles["top-right"])

    st.markdown(f"""
        <div class="toast-notification" style="
            position: fixed;
            {position_style}
            background: {color_config['bg']};
            color: white;
            padding: 16px 24px;
            border-radius: 12px;
            border-left: 4px solid {color_config['border']};
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
            z-index: 10000;
            animation: slideIn 0.3s ease-out;
            min-width: 300px;
            max-width: 500px;
        ">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="
                    font-size: 20px;
                    font-weight: 700;
                    width: 28px;
                    height: 28px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">{icon}</span>
                <span style="font-weight: 500; flex: 1;">{message}</span>
            </div>
        </div>
        <style>
            @keyframes slideIn {{
                from {{
                    transform: translateX(100%);
                    opacity: 0;
                }}
                to {{
                    transform: translateX(0);
                    opacity: 1;
                }}
            }}
        </style>
    """, unsafe_allow_html=True)


def show_progress_indicator(
    current: int,
    total: int,
    label: str = "Progress",
    show_percentage: bool = True
):
    """
    Display a progress indicator with label and percentage.

    Args:
        current: Current progress value
        total: Total value
        label: Progress label
        show_percentage: Whether to show percentage
    """
    percentage = (current / total * 100) if total > 0 else 0

    st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; color: #64748b; font-size: 14px;">{label}</span>
                {f'<span style="font-weight: 600; color: #667eea; font-size: 14px;">{percentage:.1f}%</span>' if show_percentage else ''}
            </div>
            <div class="progress-container" style="
                width: 100%;
                height: 8px;
                background: #e2e8f0;
                border-radius: 8px;
                overflow: hidden;
            ">
                <div class="progress-bar" style="
                    width: {percentage}%;
                    height: 100%;
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    border-radius: 8px;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                <span style="font-size: 12px; color: #94a3b8;">{current} / {total}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def show_info_card(
    title: str,
    value: Any,
    icon: str = "📊",
    trend: Optional[float] = None,
    trend_label: Optional[str] = None,
    color: str = "#667eea"
):
    """
    Display an information card with optional trend indicator.

    Args:
        title: Card title
        value: Main value to display
        icon: Icon or emoji
        trend: Trend percentage (positive or negative)
        trend_label: Label for trend
        color: Accent color
    """
    trend_html = ""
    if trend is not None:
        trend_color = "#10b981" if trend >= 0 else "#ef4444"
        trend_icon = "↑" if trend >= 0 else "↓"
        trend_html = f"""
            <div style="display: flex; align-items: center; gap: 4px; margin-top: 8px;">
                <span style="color: {trend_color}; font-size: 14px; font-weight: 600;">
                    {trend_icon} {abs(trend):.1f}%
                </span>
                {f'<span style="color: #94a3b8; font-size: 12px;">{trend_label}</span>' if trend_label else ''}
            </div>
        """

    st.markdown(f"""
        <div class="metric-card" style="
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            border-left: 4px solid {color};
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <div style="color: #64748b; font-size: 14px; margin-bottom: 8px; font-weight: 500;">
                        {title}
                    </div>
                    <div style="color: #1e293b; font-size: 32px; font-weight: 700;">
                        {value}
                    </div>
                    {trend_html}
                </div>
                <div style="
                    font-size: 32px;
                    opacity: 0.6;
                    background: linear-gradient(135deg, {color}22, {color}11);
                    width: 56px;
                    height: 56px;
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    {icon}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def show_status_badge(status: str, size: str = "medium"):
    """
    Display a status badge with appropriate styling.

    Args:
        status: Status text (success, warning, error, info, pending, running)
        size: Badge size ('small', 'medium', 'large')
    """
    status_configs = {
        "success": {"bg": "#10b981", "text": "Success"},
        "completed": {"bg": "#10b981", "text": "Completed"},
        "warning": {"bg": "#f59e0b", "text": "Warning"},
        "error": {"bg": "#ef4444", "text": "Error"},
        "failed": {"bg": "#ef4444", "text": "Failed"},
        "info": {"bg": "#3b82f6", "text": "Info"},
        "pending": {"bg": "#94a3b8", "text": "Pending"},
        "running": {"bg": "#667eea", "text": "Running"},
        "active": {"bg": "#10b981", "text": "Active"},
        "inactive": {"bg": "#94a3b8", "text": "Inactive"}
    }

    sizes = {
        "small": {"padding": "4px 8px", "font-size": "11px"},
        "medium": {"padding": "6px 12px", "font-size": "13px"},
        "large": {"padding": "8px 16px", "font-size": "15px"}
    }

    status_lower = status.lower()
    config = status_configs.get(status_lower, {"bg": "#64748b", "text": status})
    size_config = sizes.get(size, sizes["medium"])

    st.markdown(f"""
        <span class="status-badge" style="
            background: {config['bg']};
            color: white;
            padding: {size_config['padding']};
            border-radius: 6px;
            font-weight: 600;
            font-size: {size_config['font-size']};
            display: inline-block;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        ">
            {config['text']}
        </span>
    """, unsafe_allow_html=True)

"""AURA Platform Web Dashboard - Enhanced UI"""

import json
import os
from datetime import datetime
from io import BytesIO

import httpx
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go
import streamlit as st
import sys
import os

# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

# Enhanced UI Components
try:
    from src.dashboard.components.styles import get_enhanced_css, COLORS
    from src.dashboard.components.ui_components import (
        show_loading_state, show_loading_skeleton, show_empty_state,
        show_toast_notification, show_progress_indicator, show_info_card,
        show_status_badge
    )
    from src.dashboard.components.audit_wizard import render_audit_wizard
    from src.dashboard.components.charts import render_all_visualizations
    from src.dashboard.components.comparison_view import render_audit_comparison
    from src.dashboard.components.reports import render_reports
except ImportError:
    # Fallback if components not available
    from components.styles import get_enhanced_css, COLORS
    from components.ui_components import (
        show_loading_state, show_loading_skeleton, show_empty_state,
        show_toast_notification, show_progress_indicator, show_info_card,
        show_status_badge
    )
    from components.audit_wizard import render_audit_wizard
    from components.charts import render_all_visualizations
    from components.comparison_view import render_audit_comparison
    from components.reports import render_reports

# Configuration
from src.dashboard.api_client import api_request, API_URL

# Page configuration
st.set_page_config(
    page_title="AURA Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "audit_history" not in st.session_state:
    st.session_state.audit_history = []
if "reports" not in st.session_state:
    st.session_state.reports = []
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False
if "active_audit_tab" not in st.session_state:
    st.session_state.active_audit_tab = 0


def load_audit_history_from_api():
    """Load audit history from the API on startup"""
    if not st.session_state.history_loaded:
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{API_URL}/audits", params={"limit": 100})
                if response.status_code == 200:
                    data = response.json()
                    audits = data.get("audits", [])
                    # Update session state with loaded audits
                    st.session_state.audit_history = audits
                    st.session_state.reports = audits.copy()  # Reports are the same as audit history
                    st.session_state.history_loaded = True
        except Exception as e:
            # Silently fail - audit history will just be empty
            pass


# Load audit history on startup
load_audit_history_from_api()

# Apply Enhanced CSS
st.markdown(get_enhanced_css(), unsafe_allow_html=True)





# Removed: get_score_class, generate_report_* functions (moved to components)


# Removed: generate_report_json, generate_report_markdown, generate_report_pdf (moved to components/reports.py)


def render_workspace_suggestions():
    """Render workspace suggestions in sidebar."""
    # Fetch suggestions
    suggestions_res = api_request("/workspaces/suggestions")
    suggestions = suggestions_res.get("suggestions", []) if "error" not in suggestions_res else []

    if suggestions:
        st.markdown("### ✨ Suggestions")
        for s in suggestions:
            with st.expander(f"📌 {s['title']}", expanded=True):
                st.markdown(s['description'])
                st.markdown("**Resources:**")
                for r in s['resources']:
                    st.markdown(f"- `{r['type']}`: {r['id']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Accept", key=f"accept_{s['id']}", use_container_width=True):
                        res = api_request(f"/workspaces/suggestions/{s['id']}/accept", "POST")
                        if "error" not in res:
                            st.success("Accepted!")
                            st.rerun()
                with col2:
                    if st.button("Dismiss", key=f"dismiss_{s['id']}", use_container_width=True):
                        res = api_request(f"/workspaces/suggestions/{s['id']}/dismiss", "POST")
                        if "error" not in res:
                            st.info("Dismissed")
                            st.rerun()
        st.markdown("---")


def render_sidebar():
    """Render application sidebar with enhanced navigation."""
    with st.sidebar:
        st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="
                    width: 60px;
                    height: 60px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto;
                    color: white;
                    font-size: 30px;
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
                ">
                    🛡️
                </div>
                <h1 style="color: #1e293b; font-size: 20px; margin: 10px 0 0 0;">AURA Platform</h1>
                <p style="color: #64748b; font-size: 12px; margin: 0;">Autonomous Governance System</p>
            </div>
        """, unsafe_allow_html=True)

        # ACTION-FIRST BUTTON
        st.markdown("""
            <style>
                div[data-testid="stBaseButton-secondary"] button {
                    width: 100%;
                }
            </style>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start New Audit", type="primary", use_container_width=True, key="action_first_start_audit"):
            st.session_state.navigate_to = "Audits"
            st.session_state.wizard_step = 0 # specific to wizard
            
            # Reset wizard data
            st.session_state.wizard_data = {
                'model_id': None,
                'frameworks': [],
                'policies': [],
                'test_count': 10
            }
            # Force wizard tab active if on Audits page
            st.session_state.active_audit_tab = 0 # 0=Wizard, 1=History
            st.rerun()

        st.markdown("---")

        # Navigation
        pages = {
            "🏠 Home": "Dashboard",
            "🔍 Audits": "Audits",
            "🤖 Models": "Models",
            "🧠 Agents": "Agents",
            "📜 Policies": "Policies",
            "📊 Analytics": "Analytics",
            "📑 Reports": "Reports",
            "⚙️ Settings": "Settings"
        }

        # Check for navigation override (e.g. from "Start New Audit" button)
        if "navigate_to" in st.session_state and st.session_state.navigate_to:
             for key, value in pages.items():
                 if value == st.session_state.navigate_to:
                     st.session_state.navigation_radio = key
                     break
             del st.session_state.navigate_to

        # Initialize session state if needed
        if "navigation_radio" not in st.session_state:
            st.session_state.navigation_radio = list(pages.keys())[0]

        selection = st.radio(
            "Navigation",
            list(pages.keys()),
            label_visibility="collapsed",
            key="navigation_radio"
        )
        
        # If selection changed manually, update everything
        # This part is handled by the `navigate_to` state and `default_index` logic above.
        # No explicit `if selection != current_page_key` needed here as `navigate_to` is cleared.
        
        page = pages.get(selection, "Dashboard")

        st.markdown("---")

        # Enhanced health check with animation
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

        # Enhanced quick stats
        st.markdown("---")
        st.markdown('<p style="font-weight: 600; font-size: 14px; color: #64748b; margin-bottom: 8px;">Quick Stats</p>', unsafe_allow_html=True)

        audit_count = len(st.session_state.audit_history)
        report_count = len(st.session_state.reports)

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

        # Workspace Suggestions
        render_workspace_suggestions()

        # Footer (Moved to bottom)
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; font-size: 12px; color: #64748b; margin-top: 20px;">
                <p style="margin: 4px 0;">Version: 1.2.0 | Build: 20251212</p>
                <p style="margin: 4px 0;">© 2025 Agentic Platform</p>
            </div>
        """, unsafe_allow_html=True)

        return page


def render_home_command_center():
    """Render 'Action-First' Home Command Center."""
    st.markdown('<p class="main-header">🛡️ Command Center</p>', unsafe_allow_html=True)

    # Status & Quick Actions Row
    col1, col2 = st.columns([2, 1])

    with col1:
        # Platform Status Card
        status = api_request("/status")
        agents = status.get("agents", {}) if "error" not in status else {}
        active_agents = sum(1 for a in agents.values() if a.get("status") == "active")
        
        st.markdown(f"""
            <div style="
                background: white; border-radius: 12px; padding: 1.5rem;
                display: flex; justify-content: space-between; align-items: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;
            ">
                <div>
                    <h3 style="margin: 0; color: #1e293b;">System Status</h3>
                    <p style="margin: 0; color: #64748b; font-size: 14px;">Running v1.2.0-beta</p>
                </div>
                <div style="text-align: right;">
                    <span style="
                        background: #dcfce7; color: #166534; padding: 4px 12px;
                        border-radius: 20px; font-weight: 600; font-size: 14px;
                    ">● Online</span>
                    <p style="margin: 4px 0 0 0; font-size: 12px; color: #64748b;">{active_agents}/10 Agents Active</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚡ Quick Actions")
        qa_col1, qa_col2, qa_col3 = st.columns(3)
        with qa_col1:
            if st.button("🔍 Check GPT-4", use_container_width=True):
                st.session_state.navigate_to = "Audits" 
                st.session_state.active_audit_tab = 0 # Wizard
                # Pre-fill wizard TODO
                st.rerun()
        with qa_col2:
            if st.button("📜 New Policy", use_container_width=True):
                 st.session_state.navigate_to = "Policies"
                 st.rerun()
        with qa_col3:
            if st.button("📊 View Reports", use_container_width=True):
                 st.session_state.navigate_to = "Reports"
                 st.rerun()

    with col2:
         # Recent Activity / Stats
         audit_count = len(st.session_state.audit_history)
         pass_rate = 0 # TODO calculate
         
         st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 1.5rem; border: 1px solid #e2e8f0; height: 100%;">
                <h4 style="margin-top: 0; color: #475569;">At a Glance</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <span>Total Audits</span>
                    <b>{audit_count}</b>
                </div>
                 <div style="display: flex; justify-content: space-between;">
                    <span>Reports</span>
                    <b>{len(st.session_state.reports)}</b>
                </div>
            </div>
         """, unsafe_allow_html=True)
    
    st.markdown("### 🕒 Recent Audits")
    # Reuse rendering logic from Audit History, but simplified
    if st.session_state.audit_history:
        for audit in st.session_state.audit_history[:3]: # Show top 3
             with st.expander(f"{audit.get('timestamp')} - Score: {audit.get('compliance_score')}%"):
                 st.json(audit) # Simplified view
    else:
        st.info("No recent audits found. Start one now!")

def render_dashboard():
    """Render main dashboard page."""
    st.markdown('<p class="main-header">🛡️ AURA Dashboard</p>', unsafe_allow_html=True)

    # Get platform status
    status = api_request("/status")

    if "error" in status:
        st.error(f"Failed to fetch status: {status['error']}")
        return

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    agents = status.get("agents", {})
    with col1:
        st.metric("Total Agents", len(agents))

    with col2:
        active = sum(1 for a in agents.values() if a.get("status") == "idle")
        st.metric("Active Agents", active)

    with col3:
        kb = status.get("knowledge_base", {})
        st.metric("Knowledge Items", kb.get("total_items", 0))

    with col4:
        events = status.get("event_stream", {})
        st.metric("Total Events", events.get("total_events", 0))

    st.markdown("---")

    # Agent status overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Agent Status")

        agent_data = []
        for name, info in agents.items():
            metrics = info.get("metrics", {})
            agent_data.append({
                "Agent": name.capitalize(),
                "Status": info.get("status", "unknown").upper(),
                "Tasks Completed": metrics.get("tasks_completed", 0),
                "Tasks Failed": metrics.get("tasks_failed", 0),
                "Avg Response (ms)": round(metrics.get("avg_response_time_ms", 0), 2),
            })

        if agent_data:
            st.dataframe(agent_data, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Message Queue")
        queue_data = status.get("message_bus", {}).get("queue_depths", {})

        if queue_data:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(queue_data.keys()),
                    y=list(queue_data.values()),
                    marker_color='#1f77b4'
                )
            ])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Agent",
                yaxis_title="Queue Depth"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Scheduler status
    st.markdown("---")
    st.subheader("Scheduler Status")

    scheduler = status.get("scheduler", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Schedules", scheduler.get("total_schedules", 0))
    with col2:
        st.metric("Enabled", scheduler.get("enabled_schedules", 0))
    with col3:
        st.metric("Total Tasks", scheduler.get("total_tasks", 0))
    with col4:
        st.metric("Queue Depth", scheduler.get("queue_depth", 0))


def render_agents():
    """Render agents management page."""
    st.markdown('<p class="main-header">🤖 Agents</p>', unsafe_allow_html=True)

    agents_response = api_request("/agents")

    if "error" in agents_response:
        st.error(f"Failed to fetch agents: {agents_response['error']}")
        return

    agents = agents_response.get("agents", [])

    # Agent cards
    cols = st.columns(3)

    for i, agent in enumerate(agents):
        with cols[i % 3]:
            with st.container():
                st.markdown(f"""
                <div class="agent-card">
                    <h4>{agent['name'].capitalize()} Agent</h4>
                    <p><strong>ID:</strong> {agent['id']}</p>
                    <p><strong>Status:</strong> <span class="status-healthy">{agent['status'].upper()}</span></p>
                </div>
                """, unsafe_allow_html=True)

                # Get agent metrics
                if st.button(f"View Metrics", key=f"metrics_{agent['name']}"):
                    metrics = api_request(f"/agents/{agent['name']}/metrics")
                    if "error" not in metrics:
                        st.json(metrics)


def render_models():
    """Render models management page."""
    st.markdown('<p class="main-header">🧠 Models</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Registered Models", "Register New Model", "Model Stats"])

    with tab1:
        st.subheader("Registered Models")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.selectbox("Filter by Type", ["All", "api", "ollama", "huggingface", "uploaded"])
        with col2:
            filter_provider = st.selectbox("Filter by Provider", ["All", "openai", "anthropic", "ollama", "huggingface", "custom"])
        with col3:
            filter_status = st.selectbox("Filter by Status", ["All", "active", "inactive", "error"])

        # Build query params
        params = []
        if filter_type != "All":
            params.append(f"model_type={filter_type}")
        if filter_provider != "All":
            params.append(f"provider={filter_provider}")
        if filter_status != "All":
            params.append(f"status={filter_status}")

        query = "?" + "&".join(params) if params else ""
        models_response = api_request(f"/models{query}")

        if "error" in models_response:
            st.error(f"Failed to fetch models: {models_response['error']}")
        else:
            models = models_response.get("models", [])

            if not models:
                show_empty_state(
                    icon="🧠",
                    title="No Models Found",
                    description="No models match your current filters. Try adjusting your search criteria."
                )
            else:
                for model in models:
                    status_color = "status-healthy" if model.get("status") == "active" else "status-unhealthy"

                    with st.expander(f"🧠 {model.get('name', 'Unknown')} ({model.get('id', 'Unknown')})"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f"**Type:** {model.get('model_type', 'Unknown')}")
                            st.markdown(f"**Provider:** {model.get('provider', 'Unknown')}")
                            st.markdown(f"**Model Name:** {model.get('model_name', 'N/A')}")
                            st.markdown(f"**Status:** <span class='{status_color}'>{model.get('status', 'Unknown').upper()}</span>", unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"**Temperature:** {model.get('temperature', 0.7)}")
                            st.markdown(f"**Max Tokens:** {model.get('max_tokens', 4096)}")
                            st.markdown(f"**Total Requests:** {model.get('total_requests', 0)}")
                            st.markdown(f"**Avg Latency:** {model.get('avg_latency_ms', 0):.2f} ms")

                        if model.get("description"):
                            st.markdown(f"**Description:** {model.get('description')}")

                        if model.get("tags"):
                            st.markdown(f"**Tags:** {', '.join(model.get('tags', []))}")

                        # API Key Status Section
                        st.markdown("---")
                        st.markdown("**API Key Configuration**")

                        api_key = model.get("api_key", "")
                        has_api_key = bool(api_key and api_key.strip())

                        if has_api_key:
                            # Mask the API key for security (show first 4 and last 4 chars)
                            if len(api_key) > 8:
                                masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"
                            else:
                                masked_key = "*" * len(api_key)
                            st.success(f"API Key: `{masked_key}` (Configured)")
                        else:
                            st.warning("API Key: Not configured")

                        # API Key Update Form
                        with st.form(key=f"api_key_form_{model['id']}"):
                            new_api_key = st.text_input(
                                "Update API Key",
                                type="password",
                                placeholder="Enter new API key...",
                                help="Leave empty to keep current key, or enter a new key to update"
                            )

                            col_update, col_clear = st.columns(2)
                            with col_update:
                                update_key = st.form_submit_button("Update Key", type="primary")
                            with col_clear:
                                clear_key = st.form_submit_button("Clear Key")

                            if update_key and new_api_key:
                                result = api_request(f"/models/{model['id']}", "PUT", {"api_key": new_api_key})
                                if "error" in result:
                                    st.error(f"Failed to update API key: {result.get('error')}")
                                else:
                                    st.success("API key updated successfully!")
                                    st.rerun()
                            elif clear_key:
                                result = api_request(f"/models/{model['id']}", "PUT", {"api_key": ""})
                                if "error" in result:
                                    st.error(f"Failed to clear API key: {result.get('error')}")
                                else:
                                    st.success("API key cleared!")
                                    st.rerun()

                        st.markdown("---")

                        # Action buttons
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if st.button("🧪 Test", key=f"test_{model['id']}"):
                                with st.spinner("Testing model..."):
                                    result = api_request(f"/models/{model['id']}/test", "POST")
                                    if "error" in result:
                                        st.error(f"Test failed: {result.get('error')}")
                                    elif result.get("success"):
                                        st.success(f"Test passed! Latency: {result.get('latency_ms', 0):.0f}ms")
                                        if result.get("response"):
                                            st.code(result.get("response"))
                                    else:
                                        st.error(f"Test failed: {result.get('error', 'Unknown error')}")

                        with col2:
                            if model.get("status") == "active":
                                if st.button("⏸️ Deactivate", key=f"deactivate_{model['id']}"):
                                    api_request(f"/models/{model['id']}", "PUT", {"status": "inactive"})
                                    st.rerun()
                            else:
                                if st.button("▶️ Activate", key=f"activate_{model['id']}"):
                                    api_request(f"/models/{model['id']}", "PUT", {"status": "active"})
                                    st.rerun()

                        with col3:
                            if st.button("🗑️ Delete", key=f"delete_{model['id']}"):
                                api_request(f"/models/{model['id']}", "DELETE")
                                st.rerun()

    with tab2:
        st.subheader("Register New Model")

        with st.form("model_form"):
            model_name = st.text_input("Display Name", help="Friendly name for the model")
            model_description = st.text_area("Description", help="Brief description of the model")

            col1, col2 = st.columns(2)

            with col1:
                model_type = st.selectbox(
                    "Model Type",
                    ["api", "ollama", "huggingface", "uploaded"],
                    help="Type of model connection"
                )

            with col2:
                provider = st.selectbox(
                    "Provider",
                    ["openai", "anthropic", "ollama", "huggingface", "custom"],
                    help="Model provider"
                )

            # Conditional fields based on type
            if model_type == "api":
                model_id_name = st.text_input(
                    "Model ID/Name",
                    value="gpt-3.5-turbo",
                    help="e.g., gpt-4, claude-3-sonnet-20240229"
                )
                endpoint_url = st.text_input(
                    "Custom Endpoint URL (optional)",
                    help="Leave empty for default provider endpoint"
                )
                api_key = st.text_input(
                    "API Key (optional)",
                    type="password",
                    help="Leave empty to use environment variable"
                )

            elif model_type == "ollama":
                model_id_name = st.text_input(
                    "Ollama Model Name",
                    value="llama2",
                    help="e.g., llama2, mistral, codellama"
                )
                endpoint_url = st.text_input(
                    "Ollama Endpoint",
                    value="http://localhost:11434",
                    help="Ollama API endpoint"
                )
                api_key = ""

            elif model_type == "huggingface":
                model_id_name = st.text_input(
                    "HuggingFace Model ID",
                    value="meta-llama/Llama-2-7b-chat-hf",
                    help="HuggingFace model repository ID"
                )
                endpoint_url = st.text_input(
                    "Inference Endpoint (optional)",
                    help="Custom inference endpoint URL"
                )
                api_key = st.text_input(
                    "HuggingFace API Token",
                    type="password",
                    help="Your HuggingFace API token"
                )

            else:  # uploaded
                model_id_name = st.text_input("Model Name", help="Name for the uploaded model")
                endpoint_url = ""
                api_key = ""
                st.info("After registration, use the API to upload model files.")

            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
            with col2:
                max_tokens = st.number_input("Max Tokens", 100, 32000, 4096)

            tags = st.text_input("Tags (comma-separated)", help="e.g., production, testing, fast")

            submitted = st.form_submit_button("Register Model", type="primary")

            if submitted:
                if not model_name:
                    st.error("Model name is required")
                else:
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

                    result = api_request("/models", "POST", {
                        "name": model_name,
                        "description": model_description,
                        "model_type": model_type,
                        "provider": provider,
                        "model_name": model_id_name,
                        "endpoint_url": endpoint_url,
                        "api_key": api_key,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "tags": tag_list
                    })

                    if "error" in result:
                        st.error(f"Failed to register model: {result.get('error')}")
                    else:
                        st.success(f"Model registered successfully! ID: {result.get('model_id')}")
                        st.balloons()

    with tab3:
        st.subheader("Model Registry Statistics")

        stats = api_request("/models/stats")

        if "error" in stats:
            st.error(f"Failed to fetch stats: {stats.get('error')}")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Models", stats.get("total_models", 0))

                st.markdown("**By Type:**")
                for model_type, count in stats.get("by_type", {}).items():
                    st.markdown(f"- {model_type}: {count}")

            with col2:
                st.markdown("**By Status:**")
                for status, count in stats.get("by_status", {}).items():
                    st.markdown(f"- {status}: {count}")

                st.markdown("**By Provider:**")
                for provider, count in stats.get("by_provider", {}).items():
                    st.markdown(f"- {provider}: {count}")

            # Visualizations
            if stats.get("by_provider"):
                st.markdown("---")
                providers = list(stats["by_provider"].keys())
                counts = list(stats["by_provider"].values())

                fig = go.Figure(data=[go.Pie(labels=providers, values=counts, hole=0.4)])
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), title="Models by Provider")
                st.plotly_chart(fig, use_container_width=True)


def render_audits():
    """Render audits page."""
    # Header is now in the wizard
    tab1, tab2 = st.tabs(["🧙 Wizard Mode", "Audit History"])

    with tab1:
        # Use new wizard interface
        render_audit_wizard()

    with tab2:
        st.subheader("Audit History")

        if not st.session_state.audit_history:
            show_empty_state(
                icon="📋",
                title="No Audits Yet",
                description="You haven't run any audits yet. Start your first audit to see the results here."
            )
        else:
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_model = st.text_input("Filter by Model ID", "")
            with col2:
                filter_status = st.selectbox("Filter by Status", ["All", "completed", "failed", "partial"])
            with col3:
                sort_by = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Score (High)", "Score (Low)"])

            # Filter audits
            filtered_audits = st.session_state.audit_history.copy()

            if filter_model:
                filtered_audits = [a for a in filtered_audits if filter_model.lower() in a.get("model_id", "").lower()]

            if filter_status != "All":
                filtered_audits = [a for a in filtered_audits if a.get("status") == filter_status]

            # Sort audits
            if sort_by == "Date (Newest)":
                filtered_audits.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            elif sort_by == "Date (Oldest)":
                filtered_audits.sort(key=lambda x: x.get("timestamp", ""))
            elif sort_by == "Score (High)":
                filtered_audits.sort(key=lambda x: x.get("compliance_score", 0), reverse=True)
            elif sort_by == "Score (Low)":
                filtered_audits.sort(key=lambda x: x.get("compliance_score", 0))

            # Display audits
            for idx, audit in enumerate(filtered_audits):
                score = audit.get("compliance_score", 0)
                score_pct = score * 100 if score <= 1 else score

                with st.expander(f"📋 {audit.get('audit_id', 'Unknown')} - {audit.get('model_id', 'Unknown')} ({score_pct:.1f}%)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Model:** {audit.get('model_id', 'Unknown')}")
                        st.markdown(f"**Date:** {audit.get('timestamp', 'Unknown')[:19]}")
                    with col2:
                        st.markdown(f"**Status:** {audit.get('status', 'Unknown')}")
                        st.markdown(f"**Policies:** {len(audit.get('policies', []))}")
                    with col3:
                        score_class = get_score_class(score)
                        st.markdown(f'<p class="{score_class}">{score_pct:.1f}%</p>', unsafe_allow_html=True)

                    if st.button("View Full Report", key=f"view_audit_history_{idx}"):
                        st.session_state.selected_report = audit
                        st.session_state.navigate_to = "Reports"
                        st.rerun()


def render_policies():
    """Render policies page."""
    st.markdown('<p class="main-header">📜 Policies</p>', unsafe_allow_html=True)

    # tab1, tab2, tab3, tab4 = st.tabs(["Generate from Regulation", "Add Policy Manually", "Upload from Excel", "Existing Policies"])
    tab1, tab2, tab3 = st.tabs(["Generate from Regulation", "Add Policy Manually", "Existing Policies"])

    with tab1:
        st.subheader("Generate Policy from Regulatory Text")
        st.markdown("Enter regulatory text to automatically generate compliance policies using AI.")

        with st.form("policy_form"):
            regulation_name = st.text_input(
                "Regulation Name",
                value="EU AI Act",
                help="Name of the regulation or standard"
            )

            regulation_text = st.text_area(
                "Regulatory Text",
                value="""AI systems must be transparent about their capabilities and limitations.
Users must be informed when they are interacting with an AI system.
AI systems must not manipulate users through subliminal techniques.
High-risk AI systems must undergo conformity assessments.
AI systems must maintain accurate records of their operations.""",
                height=200,
                help="Paste the regulatory text you want to convert to policies"
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                submitted = st.form_submit_button("Generate Policies", type="primary")

            if submitted:
                with st.spinner("Analyzing regulation and generating policies with AI..."):
                    result = api_request("/policies/generate", "POST", {
                        "regulation_text": regulation_text,
                        "regulation_name": regulation_name
                    })

                    if "error" in result:
                        st.error(f"Failed to generate policies: {result.get('error', 'Unknown error')}")
                    else:
                        st.success(f"Generated {result.get('policies_created', 0)} policies successfully!")

                        # Display generated policies
                        for policy in result.get("policies", []):
                            with st.expander(f"📜 {policy.get('name', 'Unknown Policy')}"):
                                st.markdown(f"**ID:** {policy.get('id', 'Unknown')}")
                                st.markdown(f"**Category:** {policy.get('category', 'Unknown')}")
                                st.markdown(f"**Severity:** {policy.get('severity', 'Unknown')}")
                                st.markdown(f"**Description:** {policy.get('description', 'No description')}")

                                if policy.get("rules"):
                                    st.markdown("**Rules:**")
                                    for rule in policy.get("rules", []):
                                        st.markdown(f"- {rule.get('text', rule)}")

    with tab2:
        st.subheader("Add Policy Manually")
        st.markdown("Create a custom compliance policy with your own rules.")

        with st.form("manual_policy_form"):
            col1, col2 = st.columns(2)
            with col1:
                policy_id = st.text_input(
                    "Policy ID",
                    placeholder="e.g., custom-001",
                    help="Unique identifier for the policy"
                )
                policy_name = st.text_input(
                    "Policy Name",
                    placeholder="e.g., Custom Compliance Policy",
                    help="Display name for the policy"
                )
                policy_category = st.selectbox(
                    "Category",
                    options=["Illegal Activities", "Violence", "Financial Fraud", "Self Harm", "Extremism", "Medical", "Legal", "Hate Speech", "Sexual Content", "Privacy", "Hallucination", "Overconfidence"],
                    help="Policy category"
                )

            with col2:
                policy_severity = st.selectbox(
                    "Severity",
                    options=['critical', 'high', 'medium', 'low'],
                    help="Policy severity"
                )
                policy_version = st.text_input(
                    "Version",
                    value="1.0.0",
                    help="Policy version"
                )
                policy_active = st.checkbox("Active", value=True, help="Enable this policy for audits")
                

            policy_description = st.text_area(
                "Description",
                placeholder="Describe what this policy enforces...",
                height=100,
                help="Detailed description of the policy"
            )

            st.markdown("### Rules")
            st.markdown("Add compliance rules (one per line). Format: `rule text | severity` where severity is critical, high, medium, or low.")
            rules_text = st.text_area(
                "Rules (one per line)",
                placeholder="Must not generate harmful content | critical\nMust be transparent about AI nature | high\nMust protect user privacy | medium",
                height=150,
                help="Enter each rule on a new line with optional severity after '|'"
            )

            regulatory_refs = st.text_input(
                "Regulatory References (comma-separated)",
                placeholder="EU AI Act, NIST AI RMF",
                help="Related regulations or standards"
            )

            submitted = st.form_submit_button("Create Policy", type="primary")

            if submitted:
                if not policy_id or not policy_name or not policy_description:
                    st.error("Please fill in all required fields (ID, Name, Description)")
                else:
                    # Parse rules
                    rules = []
                    if rules_text.strip():
                        for i, line in enumerate(rules_text.strip().split('\n'), 1):
                            if '|' in line:
                                parts = line.split('|')
                                rule_text = parts[0].strip()
                                severity = parts[1].strip().lower() if len(parts) > 1 else "medium"
                            else:
                                rule_text = line.strip()
                                severity = "medium"

                            if rule_text:
                                rules.append({
                                    "id": f"{policy_id}-rule-{i}",
                                    "text": rule_text,
                                    "severity": severity
                                })

                    # Parse regulatory references
                    refs = [r.strip() for r in regulatory_refs.split(',') if r.strip()] if regulatory_refs else []

                    # Create policy
                    with st.spinner("Creating policy..."):
                        result = api_request("/policies", "POST", {
                            "id": policy_id,
                            "name": policy_name,
                            "description": policy_description,
                            "category": policy_category,
                            "severity": policy_severity,
                            "version": policy_version,
                            "active": policy_active,
                            "rules": rules,
                            "regulatory_references": refs
                        })

                        if "error" in result:
                            st.error(f"Failed to create policy: {result.get('error', 'Unknown error')}")
                        else:
                            st.success(f"Policy '{policy_name}' created successfully!")
                            st.rerun()

    # with tab3:
    #     st.subheader("Upload Policies from Excel")
    #     st.markdown("Upload an Excel file (.xlsx or .xls) containing policy data.")

    #     st.markdown("""
    #     **Excel Format - Two Options:**

    #     **Option 1 - Simple Format (Descriptions Only):**
    #     - Just one column with policy descriptions (one per row)
    #     - System will auto-generate policy IDs, names, and extract rules

    #     **Option 2 - Full Format (All Fields):**
    #     - Column 1: `policy_id` - Unique identifier for the policy
    #     - Column 2: `policy_name` - Name of the policy
    #     - Column 3: `description` - Policy description
    #     - Column 4: `category` - Category (safety, fairness, privacy, transparency, etc.)
    #     - Column 5: `severity` - Severity (low, high, medium, critical,)
    #     - Column 5: `version` - Version number (e.g., 1.0.0)
    #     - Column 6: `active` - TRUE or FALSE
    #     - Column 7: `rules` - Rules separated by semicolon (;)
    #     - Column 8: `rule_severities` - Severities for each rule, separated by semicolon (;)
    #     """)

    #     uploaded_file = st.file_uploader(
    #         "Choose an Excel file",
    #         type=['xlsx', 'xls'],
    #         help="Upload an Excel file with policy data"
    #     )

    #     if uploaded_file is not None:
    #         try:
    #             import pandas as pd

    #             # Read Excel file
    #             df = pd.read_excel(uploaded_file)

    #             st.success(f"File uploaded successfully! Found {len(df)} rows.")

    #             # Detect format: Simple (1 column) or Full (multiple columns)
    #             is_simple_format = len(df.columns) == 1 or 'description' not in df.columns.str.lower()

    #             if is_simple_format:
    #                 st.info("Detected: **Simple format** (descriptions only) - will auto-generate policy IDs and names")
    #                 # Use first column as description regardless of header name
    #                 df_descriptions = df.iloc[:, 0]
    #             else:
    #                 st.info("Detected: **Full format** (all fields provided)")

    #             # Preview the data
    #             st.markdown("### Preview")
    #             st.dataframe(df.head(), use_container_width=True)

    #             # Import button
    #             if st.button("Import Policies", type="primary"):
    #                 policies_created = 0
    #                 policies_failed = 0

    #                 with st.spinner("Importing policies..."):
    #                     if is_simple_format:
    #                         # Simple format: Auto-generate everything from descriptions
    #                         for idx, description in enumerate(df_descriptions):
    #                             try:
    #                                 description_text = str(description).strip()
    #                                 if not description_text or description_text == 'nan':
    #                                     continue

    #                                 # Auto-generate policy ID
    #                                 policy_id = f"policy-{str(idx + 1).zfill(3)}"

    #                                 # Auto-generate policy name from first 50 chars of description
    #                                 policy_name = description_text[:50]
    #                                 if len(description_text) > 50:
    #                                     policy_name += "..."

    #                                 # Extract rules from description (split by sentences)
    #                                 import re
    #                                 sentences = re.split(r'[.!?]+', description_text)
    #                                 rules = []
    #                                 for i, sentence in enumerate(sentences):
    #                                     sentence = sentence.strip()
    #                                     if sentence and len(sentence) > 5:  # Skip very short fragments
    #                                         rules.append({
    #                                             "id": f"{policy_id}-rule-{i+1}",
    #                                             "text": sentence,
    #                                             "severity": "medium"
    #                                         })

    #                                 # Create policy with auto-generated values
    #                                 policy_data = {
    #                                     "id": policy_id,
    #                                     "name": policy_name,
    #                                     "description": description_text,
    #                                     "category": "other",
    #                                     "severity": "other",
    #                                     "version": "1.0.0",
    #                                     "active": True,
    #                                     "rules": rules,
    #                                     "regulatory_references": []
    #                                 }

    #                                 result = api_request("/policies", "POST", policy_data)

    #                                 if "error" in result:
    #                                     st.warning(f"Row {idx + 1}: Failed to create policy '{policy_id}' - {result.get('error')}")
    #                                     policies_failed += 1
    #                                 else:
    #                                     policies_created += 1

    #                             except Exception as e:
    #                                 st.warning(f"Row {idx + 1}: Error processing row - {str(e)}")
    #                                 policies_failed += 1

    #                     else:
    #                         # Full format: Use provided values
    #                         for idx, row in df.iterrows():
    #                             try:
    #                                 # Parse rules
    #                                 rules_text = str(row.get('rules', '')).strip()
    #                                 severities_text = str(row.get('rule_severities', '')).strip()

    #                                 rules = []
    #                                 if rules_text and rules_text != 'nan':
    #                                     rule_list = [r.strip() for r in rules_text.split(';') if r.strip()]
    #                                     severity_list = [s.strip().lower() for s in severities_text.split(';')] if severities_text != 'nan' else []

    #                                     for i, rule_text in enumerate(rule_list):
    #                                         severity = severity_list[i] if i < len(severity_list) else "medium"
    #                                         rules.append({
    #                                             "id": f"{row.get('policy_id', 'policy')}-rule-{i+1}",
    #                                             "text": rule_text,
    #                                             "severity": severity
    #                                         })

    #                                 # Create policy
    #                                 policy_data = {
    #                                     "id": str(row.get('policy_id', '')).strip(),
    #                                     "name": str(row.get('policy_name', '')).strip(),
    #                                     "description": str(row.get('description', '')).strip(),
    #                                     "category": str(row.get('category', 'other')).strip().lower(),
    #                                     "severity": str(row.get('severity', 'other')).strip().lower(),
    #                                     "version": str(row.get('version', '1.0.0')).strip(),
    #                                     "active": bool(str(row.get('active', 'TRUE')).strip().upper() == 'TRUE'),
    #                                     "rules": rules,
    #                                     "regulatory_references": []
    #                                 }

    #                                 result = api_request("/policies", "POST", policy_data)

    #                                 if "error" in result:
    #                                     st.warning(f"Row {idx + 1}: Failed to create policy '{policy_data['id']}' - {result.get('error')}")
    #                                     policies_failed += 1
    #                                 else:
    #                                     policies_created += 1

    #                             except Exception as e:
    #                                 st.warning(f"Row {idx + 1}: Error processing row - {str(e)}")
    #                                 policies_failed += 1

    #                 # Summary
    #                 st.success(f"Import complete! Created {policies_created} policies.")
    #                 if policies_failed > 0:
    #                     st.warning(f"{policies_failed} policies failed to import.")

    #                 if policies_created > 0:
    #                     st.balloons()
    #                     st.rerun()

    #         except Exception as e:
    #             st.error(f"Failed to read Excel file: {str(e)}")
    #             st.info("Please ensure your file follows the required format.")

    #     # Download template
    #     st.markdown("---")
    #     st.markdown("### Download Template")
    #     st.markdown("Download a sample Excel template to see the required format:")

    #     col1, col2 = st.columns(2)

    #     with col1:
    #         if st.button("Download Simple Template", type="secondary"):
    #             import pandas as pd
    #             from io import BytesIO

    #             # Create simple template with just descriptions
    #             template_data = {
    #                 'Description': [
    #                     'AI system must not generate harmful or dangerous content. It should refuse requests that could lead to harm.',
    #                     'User privacy must be protected at all times. Personal information should never be shared without consent.',
    #                     'System outputs must be fair and unbiased. Discrimination based on protected characteristics is prohibited.'
    #                 ]
    #             }

    #             template_df = pd.DataFrame(template_data)

    #             # Create Excel file in memory
    #             output = BytesIO()
    #             with pd.ExcelWriter(output, engine='openpyxl') as writer:
    #                 template_df.to_excel(writer, index=False, sheet_name='Policies')

    #             output.seek(0)

    #             st.download_button(
    #                 label="📥 Download Simple Template.xlsx",
    #                 data=output.getvalue(),
    #                 file_name="policy_template_simple.xlsx",
    #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #                 key="download_simple"
    #             )

    #     with col2:
    #         if st.button("Download Full Template", type="secondary"):
    #             import pandas as pd
    #             from io import BytesIO

    #             # Create full template with all fields
    #             template_data = {
    #                 'policy_id': ['example-001', 'example-002'],
    #                 'policy_name': ['Example Safety Policy', 'Example Privacy Policy'],
    #                 'description': ['Ensures AI safety compliance', 'Protects user privacy'],
    #                 'category': ['meidcal', 'privacy'],
    #                 'severity': ['critical', 'high'],
    #                 'version': ['1.0.0', '1.0.0'],
    #                 'active': [True, True],
    #                 'rules': [
    #                     'Must not generate harmful content; Must refuse dangerous requests',
    #                     'Must protect user data; Must not share personal information'
    #                 ],
    #                 'rule_severities': ['critical; high', 'high; critical']
    #             }

    #             template_df = pd.DataFrame(template_data)

    #             # Create Excel file in memory
    #             output = BytesIO()
    #             with pd.ExcelWriter(output, engine='openpyxl') as writer:
    #                 template_df.to_excel(writer, index=False, sheet_name='Policies')

    #             output.seek(0)

    #             st.download_button(
    #                 label="📥 Download Full Template.xlsx",
    #                 data=output.getvalue(),
    #                 file_name="policy_template_full.xlsx",
    #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #                 key="download_full"
    #             )

    with tab3:
        st.subheader("Existing Policies")

        if st.button("Refresh Policies"):
            st.rerun()

        policies = api_request("/policies")

        if "error" in policies:
            st.error(f"Failed to fetch policies: {policies['error']}")
        elif not policies.get("policies"):
            st.info("No policies created yet. Generate policies from regulatory text to get started.")
        else:
            # Policy statistics
            policy_list = policies.get("policies", [])
            categories = {}
            for p in policy_list:
                cat = p.get("category", "uncategorized")
                categories[cat] = categories.get(cat, 0) + 1

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Total Policies", len(policy_list))
            with col2:
                st.markdown("**By Category:** " + ", ".join([f"{k}: {v}" for k, v in categories.items()]))

            st.markdown("---")

            for policy in policy_list:
                policy_id = policy.get('id', 'Unknown')
                with st.expander(f"📜 {policy.get('name', policy_id)}"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.markdown(f"**ID:** {policy_id}")
                        st.markdown(f"**Category:** {policy.get('category', 'Unknown')}")
                        st.markdown(f"**Version:** {policy.get('version', '1.0.0')}")
                    with col2:
                        st.markdown(f"**Severity:** {policy.get('severity', 'Unknown')}")
                        st.markdown(f"**Active:** {'Yes' if policy.get('active', True) else 'No'}")
                        st.markdown(f"**Rules:** {len(policy.get('rules', []))}")
                    with col3:
                        # Delete button
                        if st.button("🗑️ Delete", key=f"delete_{policy_id}"):
                            result = api_request(f"/policies/{policy_id}", "DELETE")
                            if "error" in result:
                                st.error(f"Failed to delete: {result.get('error')}")
                            else:
                                st.success(f"Policy '{policy_id}' deleted!")
                                st.rerun()

                    st.markdown(f"**Description:** {policy.get('description', 'No description')}")

                    if policy.get("rules"):
                        st.markdown("**Rules:**")
                        for i, rule in enumerate(policy.get("rules", []), 1):
                            rule_text = rule.get("text", rule) if isinstance(rule, dict) else rule
                            severity = rule.get("severity", "medium") if isinstance(rule, dict) else "medium"
                            st.markdown(f"{i}. {rule_text} ({severity})")

                    # Edit form
                    st.markdown("---")
                    st.markdown("#### Edit Policy")

                    with st.form(f"edit_form_{policy_id}"):
                        edit_col1, edit_col2 = st.columns(2)
                        with edit_col1:
                            edit_name = st.text_input(
                                "Name",
                                value=policy.get('name', ''),
                                key=f"edit_name_{policy_id}"
                            )
                            edit_category = st.selectbox(
                                "Category",
                                options=["Illegal Activities", "Violence", "Financial Fraud", "Self Harm", "Extremism", "Medical", "Legal", "Hate Speech", "Sexual Content", "Privacy", "Hallucination", "Overconfidence"],
                                index=["Illegal Activities", "Violence", "Financial Fraud", "Self Harm", "Extremism", "Medical", "Legal", "Hate Speech", "Sexual Content", "Privacy", "Hallucination", "Overconfidence"].index(policy.get('category', 'other')) if policy.get('category', 'other') in ["Illegal Activities", "Violence", "Financial Fraud", "Self Harm", "Extremism", "Medical", "Legal", "Hate Speech", "Sexual Content", "Privacy", "Hallucination", "Overconfidence"] else 11,
                                key=f"edit_category_{policy_id}"
                            )
                        with edit_col2:
                            edit_severity = st.selectbox(
                                "Severity",
                                options=['critical', 'high', 'medium', 'low'],
                                index=['critical', 'high', 'medium','low'].index(policy.get('severity', 'other')) if policy.get('severity', 'other') in ['critical', 'high', 'medium', 'low'] else 3,
                                key=f"edit_severity_{policy_id}"
                            ) 
                            edit_version = st.text_input(
                                "Version",
                                value=policy.get('version', '1.0.0'),
                                key=f"edit_version_{policy_id}"
                            )
                            edit_active = st.checkbox(
                                "Active",
                                value=policy.get('active', True),
                                key=f"edit_active_{policy_id}"
                            )

                        edit_description = st.text_area(
                            "Description",
                            value=policy.get('description', ''),
                            height=80,
                            key=f"edit_desc_{policy_id}"
                        )

                        # Convert rules to editable format
                        current_rules = policy.get('rules', [])
                        rules_str = '\n'.join([
                            f"{r.get('text', r) if isinstance(r, dict) else r} | {r.get('severity', 'medium') if isinstance(r, dict) else 'medium'}"
                            for r in current_rules
                        ])

                        edit_rules_text = st.text_area(
                            "Rules (one per line, format: rule text | severity)",
                            value=rules_str,
                            height=120,
                            key=f"edit_rules_{policy_id}"
                        )

                        if st.form_submit_button("Update Policy"):
                            # Parse rules
                            updated_rules = []
                            if edit_rules_text.strip():
                                for i, line in enumerate(edit_rules_text.strip().split('\n'), 1):
                                    if '|' in line:
                                        parts = line.split('|')
                                        rule_text = parts[0].strip()
                                        severity = parts[1].strip().lower() if len(parts) > 1 else "medium"
                                    else:
                                        rule_text = line.strip()
                                        severity = "medium"

                                    if rule_text:
                                        updated_rules.append({
                                            "id": f"{policy_id}-rule-{i}",
                                            "text": rule_text,
                                            "severity": severity
                                        })

                            # Update policy
                            result = api_request(f"/policies/{policy_id}", "PUT", {
                                "name": edit_name,
                                "description": edit_description,
                                "category": edit_category,
                                "severity": edit_severity,
                                "version": edit_version,
                                "active": edit_active,
                                "rules": updated_rules
                            })

                            if "error" in result:
                                st.error(f"Failed to update policy: {result.get('error')}")
                            else:
                                st.success(f"Policy '{edit_name}' updated successfully!")
                                st.rerun()


# Removed: render_reports (moved to components/reports.py)


def render_analytics():
    """Render analytics and visualizations page."""
    st.markdown('<p class="main-header">📈 Analytics & Visualizations</p>', unsafe_allow_html=True)
    st.markdown("Comprehensive data visualizations and insights from audit results")
    st.markdown("---")

    # Render all visualization components
    render_all_visualizations()


def render_metrics():
    """Render metrics page."""
    st.markdown('<p class="main-header">📊 Metrics</p>', unsafe_allow_html=True)

    status = api_request("/status")

    if "error" in status:
        st.error(f"Failed to fetch metrics: {status['error']}")
        return

    # Agent performance metrics
    st.subheader("Agent Performance")

    agents = status.get("agents", {})

    if agents:
        # Tasks completed chart
        agent_names = [name.capitalize() for name in agents.keys()]
        tasks_completed = [info.get("metrics", {}).get("tasks_completed", 0) for info in agents.values()]
        tasks_failed = [info.get("metrics", {}).get("tasks_failed", 0) for info in agents.values()]

        fig = go.Figure(data=[
            go.Bar(name='Completed', x=agent_names, y=tasks_completed, marker_color='#28a745'),
            go.Bar(name='Failed', x=agent_names, y=tasks_failed, marker_color='#dc3545')
        ])
        fig.update_layout(
            barmode='group',
            height=400,
            xaxis_title="Agent",
            yaxis_title="Tasks"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Response time chart
        st.subheader("Average Response Time (ms)")
        response_times = [info.get("metrics", {}).get("avg_response_time_ms", 0) for info in agents.values()]

        fig2 = go.Figure(data=[
            go.Bar(x=agent_names, y=response_times, marker_color='#1f77b4')
        ])
        fig2.update_layout(
            height=300,
            xaxis_title="Agent",
            yaxis_title="Response Time (ms)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Event stream metrics
    st.markdown("---")
    st.subheader("Event Stream")

    events = status.get("event_stream", {})
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Events", events.get("total_events", 0))
    with col2:
        st.metric("Subscribers", events.get("subscriber_count", 0))
    with col3:
        st.metric("Queue Size", events.get("queue_size", 0))


def render_settings():
    """Render settings page."""
    st.markdown('<p class="main-header">⚙️ Settings</p>', unsafe_allow_html=True)

    st.subheader("Platform Information")

    info = api_request("/")

    if "error" not in info:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Platform:** {info.get('name', 'Unknown')}")
            st.info(f"**Version:** {info.get('version', 'Unknown')}")
        with col2:
            st.info(f"**Status:** {info.get('status', 'Unknown')}")
            st.info(f"**API URL:** {API_URL}")

    st.markdown("---")
    st.subheader("Session Management")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Audit History"):
            st.session_state.audit_history = []
            st.session_state.reports = []
            st.success("Audit history cleared!")
            st.rerun()

    with col2:
        st.metric("Stored Audits", len(st.session_state.audit_history))

    st.markdown("---")
    st.subheader("API Configuration")

    new_url = st.text_input("API URL", value=API_URL)
    if st.button("Update API URL"):
        st.session_state['api_url'] = new_url
        st.success(f"API URL updated to: {new_url}")
        st.rerun()

    st.markdown("---")
    st.subheader("About AURA Platform")
    st.markdown("""
    **AURA (Autonomous Universal Regulatory Auditor)** is an AI-powered platform for:

    - 🔍 Automated compliance auditing
    - 📜 Policy generation from regulations
    - 🧪 Adversarial testing
    - 📊 Real-time monitoring
    - 📑 Comprehensive reporting
    - 🔧 Auto-remediation

    Built with LangChain, LangGraph, and FastAPI.
    """)


def main():
    """Main application entry point."""
    page = render_sidebar()

    if page == "Dashboard":
        render_home_command_center()
    elif page == "Audits":
        if st.session_state.active_audit_tab == 0:
            render_audit_wizard()
        else:
            render_audit_history()
            
    elif page == "Models":
        render_models()

    elif page == "Agents":
        render_agents()
    elif page == "Policies":
        render_policies()
    elif page == "Reports":
        render_reports()
    elif page == "Analytics":
        render_analytics()
    elif page == "Comparison":
        render_audit_comparison()
    elif page == "Metrics":
        render_metrics()
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()

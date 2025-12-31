"""Real-Time Progress Monitoring Component for AURA Dashboard"""

import streamlit as st
import textwrap
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from .styles import COLORS


def render_progress_monitor(
    audit_id: str,
    total_tests: int = 100,
    current_stage: str = "Initializing",
    progress_percent: float = 0.0,
    tests_completed: int = 0,
    tests_passed: int = 0,
    tests_failed: int = 0,
    start_time: Optional[datetime] = None,
    current_test: Optional[Dict[str, Any]] = None,
    logs: Optional[List[str]] = None,
    show_logs: bool = False
):
    """
    Render real-time progress monitoring interface.

    Args:
        audit_id: Unique identifier for the audit
        total_tests: Total number of tests to execute
        current_stage: Current execution stage name
        progress_percent: Progress percentage (0-100)
        tests_completed: Number of tests completed
        tests_passed: Number of tests that passed
        tests_failed: Number of tests that failed
        start_time: Audit start time for ETA calculation
        current_test: Currently executing test details
        logs: List of log messages
        show_logs: Whether to show log streaming section
    """

    # Calculate estimated time remaining
    eta_str = "Calculating..."
    if start_time and tests_completed > 0:
        elapsed = datetime.now() - start_time
        avg_time_per_test = elapsed.total_seconds() / tests_completed
        remaining_tests = total_tests - tests_completed
        eta_seconds = avg_time_per_test * remaining_tests
        eta = timedelta(seconds=int(eta_seconds))
        eta_str = str(eta).split('.')[0]  # Remove microseconds

    # Header
    st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
">
    <h3 style="margin: 0; font-size: 1.5rem;">🔄 Audit in Progress</h3>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Audit ID: {audit_id}</p>
</div>""", unsafe_allow_html=True)

    # Progress Bar with Percentage
    st.markdown(f"""
<div style="margin: 1.5rem 0;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
<span style="font-weight: 600; color: {COLORS['text_primary']};">Overall Progress</span>
<span style="font-weight: 700; color: {COLORS['primary']}; font-size: 1.25rem;">{progress_percent:.1f}%</span>
</div>
<div style="
background: {COLORS['border']};
border-radius: 9999px;
height: 12px;
overflow: hidden;
position: relative;
">
<div style="
background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
height: 100%;
width: {progress_percent}%;
border-radius: 9999px;
transition: width 0.5s ease;
box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
"></div>
</div>
</div>""", unsafe_allow_html=True)

    # Stage Indicator
    st.markdown(f"""
<div style="
background: rgba(102, 126, 234, 0.1);
border-left: 4px solid {COLORS['primary']};
padding: 1rem 1.25rem;
border-radius: 8px;
margin: 1.5rem 0;
">
<div style="display: flex; align-items: center; gap: 0.75rem;">
<div style="
width: 12px;
height: 12px;
border-radius: 50%;
background: {COLORS['primary']};
animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
"></div>
<div>
<div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Current Stage</div>
<div style="font-size: 1.1rem; font-weight: 600; color: {COLORS['text_primary']}; margin-top: 0.25rem;">{current_stage}</div>
</div>
</div>
</div>
<style>
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(1.2); }}
}}
</style>""", unsafe_allow_html=True)

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
<div style="
background: white;
border-radius: 12px;
padding: 1.25rem;
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
border: 1px solid {COLORS['border']};
">
<div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem;">Tests Completed</div>
<div style="font-size: 2rem; font-weight: 700; color: {COLORS['text_primary']};">{tests_completed}</div>
<div style="font-size: 0.875rem; color: {COLORS['text_secondary']};">of {total_tests}</div>
</div>""", unsafe_allow_html=True)

    with col2:
        pass_rate = (tests_passed / tests_completed * 100) if tests_completed > 0 else 0
        st.markdown(f"""
<div style="
background: white;
border-radius: 12px;
padding: 1.25rem;
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
border: 1px solid {COLORS['border']};
">
<div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem;">Passed</div>
<div style="font-size: 2rem; font-weight: 700; color: {COLORS['success']};">{tests_passed}</div>
<div style="font-size: 0.875rem; color: {COLORS['success']};">{pass_rate:.1f}% pass rate</div>
</div>""", unsafe_allow_html=True)

    with col3:
        fail_rate = (tests_failed / tests_completed * 100) if tests_completed > 0 else 0
        st.markdown(f"""
<div style="
background: white;
border-radius: 12px;
padding: 1.25rem;
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
border: 1px solid {COLORS['border']};
">
<div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem;">Failed</div>
<div style="font-size: 2rem; font-weight: 700; color: {COLORS['danger']};">{tests_failed}</div>
<div style="font-size: 0.875rem; color: {COLORS['danger']};">{fail_rate:.1f}% fail rate</div>
</div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
<div style="
background: white;
border-radius: 12px;
padding: 1.25rem;
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
border: 1px solid {COLORS['border']};
">
<div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem;">ETA</div>
<div style="font-size: 1.5rem; font-weight: 700; color: {COLORS['info']};">{eta_str}</div>
<div style="font-size: 0.875rem; color: {COLORS['text_secondary']};">Remaining</div>
</div>""", unsafe_allow_html=True)

    # Current Test Details (Expandable)
    if current_test:
        with st.expander("📝 Current Test Details", expanded=True):
            st.markdown(f"""
                <div style="background: rgba(59, 130, 246, 0.05); padding: 1rem; border-radius: 8px; border-left: 3px solid {COLORS['info']};">
                    <div style="margin-bottom: 0.75rem;">
                        <span style="font-weight: 600; color: {COLORS['text_primary']};">Test:</span>
                        <span style="color: {COLORS['text_secondary']}; margin-left: 0.5rem;">{current_test.get('name', 'Unknown Test')}</span>
                    </div>
                    <div style="margin-bottom: 0.75rem;">
                        <span style="font-weight: 600; color: {COLORS['text_primary']};">Policy:</span>
                        <span style="color: {COLORS['text_secondary']}; margin-left: 0.5rem;">{current_test.get('policy', 'N/A')}</span>
                    </div>
                    <div style="margin-bottom: 0.75rem;">
                        <span style="font-weight: 600; color: {COLORS['text_primary']};">Framework:</span>
                        <span style="color: {COLORS['text_secondary']}; margin-left: 0.5rem;">{current_test.get('framework', 'N/A')}</span>
                    </div>
                    <div>
                        <span style="font-weight: 600; color: {COLORS['text_primary']};">Status:</span>
                        <span style="
                            background: {COLORS['warning']}20;
                            color: {COLORS['warning']};
                            padding: 0.25rem 0.75rem;
                            border-radius: 12px;
                            font-size: 0.75rem;
                            font-weight: 600;
                            text-transform: uppercase;
                            margin-left: 0.5rem;
                        ">RUNNING</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Log Streaming (Optional)
    if show_logs and logs:
        with st.expander("📋 Live Logs", expanded=False):
            st.markdown(f"""
                <div style="
                    background: #1e293b;
                    color: #e2e8f0;
                    padding: 1rem;
                    border-radius: 8px;
                    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                    font-size: 0.875rem;
                    max-height: 400px;
                    overflow-y: auto;
                    line-height: 1.6;
                ">
                    {"<br>".join([f"<div>{log}</div>" for log in logs[-50:]])}
                </div>
            """, unsafe_allow_html=True)

    # Control Buttons (Only show when active)
    # Control Buttons (Only show when active)
    # Temporarily disabled to prevent UI duplication issues until functionality is fully implemented
    should_show_buttons = False  # Hard-disable for now
    
    # Original logic preserved for reference:
    # if progress_percent >= 100:
    #     should_show_buttons = False
    # if current_stage in ["Completed", "Generated Report", "Generating Report", "Analysis", "Analyzing"]:
    #     should_show_buttons = False

    if should_show_buttons:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("⏸️ Pause", use_container_width=True, disabled=True, key=f"pause_btn_{audit_id}_{current_stage}"):
                st.info("Pause functionality coming soon")

        with col2:
            if st.button("🛑 Cancel", use_container_width=True, type="secondary", key=f"cancel_btn_{audit_id}_{current_stage}"):
                if st.session_state.get(f"confirm_cancel_{audit_id}"):
                    st.warning("Canceling audit...")
                    # TODO: Implement actual cancel logic
                    st.session_state[f"confirm_cancel_{audit_id}"] = False
                else:
                    st.session_state[f"confirm_cancel_{audit_id}"] = True
                    st.warning("Click again to confirm cancellation")


def render_progress_stages(stages: List[Dict[str, Any]], current_stage_index: int):
    """
    Render visual progress through multiple stages.

    Args:
        stages: List of stage dictionaries with 'name', 'icon', 'status'
        current_stage_index: Index of the currently active stage
    """

    st.markdown("""
<div style="margin: 2rem 0;">
    <h4 style="margin-bottom: 1.5rem; color: #1e293b;">Audit Pipeline</h4>
</div>
""", unsafe_allow_html=True)

    # Build stages HTML
    stages_html = ""
    for idx, stage in enumerate(stages):
        icon = stage.get('icon', '⚙️')
        name = stage.get('name', f'Stage {idx + 1}')
        status = stage.get('status', 'pending')  # pending, active, completed, failed

        # Determine colors based on status
        if status == 'completed':
            bg_color = COLORS['success']
            border_color = COLORS['success']
            text_color = 'white'
            icon = '✓'
        elif status == 'active':
            bg_color = COLORS['primary']
            border_color = COLORS['primary']
            text_color = 'white'
        elif status == 'failed':
            bg_color = COLORS['danger']
            border_color = COLORS['danger']
            text_color = 'white'
            icon = '✗'
        else:  # pending
            bg_color = COLORS['border']
            border_color = COLORS['border']
            text_color = COLORS['text_secondary']

        # Connector line (except for last stage)
        connector = ""
        if idx < len(stages) - 1:
            connector_color = COLORS['success'] if status == 'completed' else COLORS['border']
            connector = f"""
<div style="
position: absolute;
top: 24px;
left: 50%;
width: 100%;
height: 4px;
background: {connector_color};
z-index: 0;
"></div>"""

        # Determine animation style
        animation_style = 'animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;' if status == 'active' else ''

        stages_html += f"""
<div style="flex: 1; position: relative; text-align: center;">
{connector}
<div style="width: 48px; height: 48px; border-radius: 50%; background: {bg_color}; color: {text_color}; display: inline-flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: 600; margin-bottom: 0.75rem; position: relative; z-index: 1; border: 3px solid {border_color}; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); {animation_style}">
{icon}
</div>
<div style="font-size: 0.875rem; font-weight: 600; color: {COLORS['text_primary'] if status in ['active', 'completed'] else COLORS['text_secondary']}; margin-top: 0.5rem;">
{name}
</div>
</div>"""

    st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: flex-start; position: relative; margin: 2rem 0;">
{stages_html}
</div>""", unsafe_allow_html=True)


def demo_progress_monitor():
    """Demo function to showcase the progress monitor component."""

    st.markdown("## Real-Time Progress Monitor Demo")

    # Simulated audit data
    if 'demo_progress' not in st.session_state:
        st.session_state.demo_progress = {
            'audit_id': 'audit_20251202_demo',
            'total_tests': 50,
            'current_stage': 'Running Tests',
            'progress_percent': 45.0,
            'tests_completed': 22,
            'tests_passed': 18,
            'tests_failed': 4,
            'start_time': datetime.now() - timedelta(minutes=5),
            'current_test': {
                'name': 'Bias Detection - Gender Equality',
                'policy': 'fairness-001',
                'framework': 'AURA Native'
            },
            'logs': [
                '[2025-12-02 10:15:23] Starting audit...',
                '[2025-12-02 10:15:24] Initializing frameworks...',
                '[2025-12-02 10:15:25] Loading policies...',
                '[2025-12-02 10:15:30] Running test 1/50...',
                '[2025-12-02 10:16:45] Test 1 PASSED',
                '[2025-12-02 10:16:46] Running test 2/50...',
            ]
        }

    data = st.session_state.demo_progress

    # Render the progress monitor
    render_progress_monitor(
        audit_id=data['audit_id'],
        total_tests=data['total_tests'],
        current_stage=data['current_stage'],
        progress_percent=data['progress_percent'],
        tests_completed=data['tests_completed'],
        tests_passed=data['tests_passed'],
        tests_failed=data['tests_failed'],
        start_time=data['start_time'],
        current_test=data['current_test'],
        logs=data['logs'],
        show_logs=True
    )

    # Demo stages
    st.markdown("---")
    stages = [
        {'name': 'Initialize', 'icon': '⚙️', 'status': 'completed'},
        {'name': 'Load Policies', 'icon': '📋', 'status': 'completed'},
        {'name': 'Run Tests', 'icon': '🧪', 'status': 'active'},
        {'name': 'Analyze Results', 'icon': '📊', 'status': 'pending'},
        {'name': 'Generate Report', 'icon': '📑', 'status': 'pending'},
    ]

    render_progress_stages(stages, 2)

    # Simulate progress button
    if st.button("▶️ Simulate Progress", use_container_width=True, type="primary", key="simulate_progress_btn"):
        # Advance progress
        data['tests_completed'] += 5
        data['tests_passed'] += 4
        data['tests_failed'] += 1
        data['progress_percent'] = (data['tests_completed'] / data['total_tests']) * 100
        data['logs'].append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running test {data['tests_completed']}/50...")
        st.rerun()


if __name__ == "__main__":
    demo_progress_monitor()

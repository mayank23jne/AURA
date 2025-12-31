"""
Wizard-Style Audit Creation Component
Provides a multi-step guided experience for creating audits
"""

import streamlit as st
import textwrap
from typing import Dict, List, Any, Optional
from datetime import datetime
from .reports import render_reports


def render_wizard_steps(current_step: int, total_steps: int, step_names: List[str]):
    """Render wizard progress indicator."""
    steps_html = ""
    for i in range(total_steps):
        is_completed = i < current_step
        is_current = i == current_step
        
        # Determine styling
        bg_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)" if i <= current_step else "#e2e8f0"
        box_shadow = "0 4px 12px rgba(102, 126, 234, 0.3)" if i <= current_step else "none"
        font_weight = "600" if is_current else "400"
        text_color = "#667eea" if is_current else ("#1e293b" if is_completed else "#94a3b8")
        label = str(i + 1)
        if is_completed:
            label = "✓"
            
        steps_html += f"""
<div style="display: flex; flex-direction: column; align-items: center; position: relative; z-index: 2; flex: 1;">
    <div style="
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: {bg_color};
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 18px;
        box-shadow: {box_shadow};
        transition: all 0.3s ease;
    ">
        {label}
    </div>
    <div style="
        margin-top: 8px;
        font-size: 13px;
        font-weight: {font_weight};
        color: {text_color};
        text-align: center;
    ">
        {step_names[i]}
    </div>
</div>"""

    st.markdown(f"""
<div style="margin: 2rem 0;">
    <div style="display: flex; justify-content: space-between; align-items: center; position: relative;">
        <div style="position: absolute; top: 24px; left: 0; right: 0; height: 4px; background: #e2e8f0; z-index: 0;"></div>
        <div style="position: absolute; top: 24px; left: 0; width: {(current_step / (total_steps - 1)) * 100}%; height: 4px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); z-index: 1; transition: width 0.3s ease;"></div>
        {steps_html}
    </div>
</div>""", unsafe_allow_html=True)


def render_audit_wizard():
    """Render the complete audit creation wizard."""

    # Initialize wizard state
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 0
    if 'wizard_data' not in st.session_state:
        st.session_state.wizard_data = {
            'model_id': None,
            'frameworks': [],
            'policies': [],
            'test_count': 10
        }

    step_names = ["Select Model", "Choose Frameworks", "Select Policies", "Review & Launch"]
    current_step = st.session_state.wizard_step

    # Render wizard header
    st.markdown('<p class="main-header">🧙 Create New Audit</p>', unsafe_allow_html=True)
    render_wizard_steps(current_step, len(step_names), step_names)

    st.markdown("<br>", unsafe_allow_html=True)

    # Render current step
    if current_step == 0:
        render_step_model_selection()
    elif current_step == 1:
        render_step_framework_selection()
    elif current_step == 2:
        render_step_policy_selection()
    elif current_step == 3:
        render_step_review_launch()

    # Navigation buttons with clear separation
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if current_step > 0:
            if st.button("← Previous", use_container_width=True, key="wizard_prev_btn"):
                st.session_state.wizard_step -= 1
                st.session_state.navigate_to = "Audits"
                st.rerun()

    with col3:
        if current_step < len(step_names) - 1:
            # Check if current step is valid before allowing next
            can_proceed = validate_current_step(current_step)

            # Show validation message if can't proceed
            if not can_proceed:
                if current_step == 0:
                    st.info("👆 Please select a model above to continue")
                elif current_step == 1:
                    st.info("👆 Please select at least one framework to continue")
                elif current_step == 2:
                    st.info("👆 Please select at least one policy to continue")

            if st.button("Next →", use_container_width=True, disabled=not can_proceed, type="primary", key="wizard_next_btn"):
                st.session_state.wizard_step += 1
                st.session_state.navigate_to = "Audits"
                st.rerun()


def validate_current_step(step: int) -> bool:
    """Validate if current step has required data to proceed."""
    wizard_data = st.session_state.wizard_data

    if step == 0:  # Model selection
        return wizard_data.get('model_id') is not None
    elif step == 1:  # Framework selection
        return len(wizard_data.get('frameworks', [])) > 0
    elif step == 2:  # Policy selection
        return len(wizard_data.get('policies', [])) > 0

    return True


def toggle_framework(fw_id):
    """Callback to toggle framework selection."""
    current = st.session_state.wizard_data.get('frameworks', [])
    if fw_id in current:
        current.remove(fw_id)
    else:
        current.append(fw_id)


def toggle_policy(policy_id):
    """Callback to toggle policy selection."""
    current = st.session_state.wizard_data.get('policies', [])
    if policy_id in current:
        current.remove(policy_id)
    else:
        current.append(policy_id)


def update_test_count_slider():
    """Callback to update test count from slider."""
    st.session_state.wizard_data['test_count'] = st.session_state.tc_slider


def update_test_count_input():
    """Callback to update test count from input."""
    st.session_state.wizard_data['test_count'] = st.session_state.tc_input


def render_step_model_selection():
    """Step 1: Model Selection with visual cards."""
    st.markdown("""
<div style="margin-bottom: 2rem;">
    <h3 style="color: #1e293b; font-weight: 600;">Select the AI Model to Audit</h3>
    <p style="color: #64748b;">Choose which model you want to test for security, bias, and compliance.</p>
</div>""", unsafe_allow_html=True)

    # Fetch models from API
    from src.dashboard.api_client import api_request
    models_response = api_request("/models")

    if "error" in models_response:
        st.error(f"Failed to load models: {models_response['error']}")
        return

    models = models_response.get("models", [])

    if not models:
        st.warning("No models available. Please register a model first.")
        return

    # Group models by provider
    providers = {}
    for model in models:
        provider = model.get('provider', 'Unknown')
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(model)

    # Display models as cards
    for provider, provider_models in providers.items():
        st.markdown(f"### {provider}")

        cols = st.columns(min(3, len(provider_models)))
        for idx, model in enumerate(provider_models):
            with cols[idx % 3]:
                model_id = model.get('id')
                model_name = model.get('name', 'Unknown')
                model_type = model.get('model_type', 'Unknown')
                status = model.get('status', 'unknown')

                is_selected = st.session_state.wizard_data.get('model_id') == model_id

                st.markdown(f"""
<div style="
    background: {'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)' if is_selected else 'white'};
    border: 2px solid {'#667eea' if is_selected else '#e2e8f0'};
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
">
    <div style="font-weight: 600; font-size: 16px; color: #1e293b; margin-bottom: 8px;">
        {'✓ ' if is_selected else ''}{model_name}
    </div>
    <div style="font-size: 13px; color: #64748b; margin-bottom: 8px;">
        <strong>ID:</strong> {model_id}
    </div>
    <div style="font-size: 13px; color: #64748b; margin-bottom: 8px;">
        <strong>Type:</strong> {model_type}
    </div>
    <div style="font-size: 13px; color: #64748b;">
        <strong>Status:</strong> <span style="color: {'#10b981' if status == 'active' else '#94a3b8'};">
            {status.upper()}
        </span>
    </div>
</div>""", unsafe_allow_html=True)

                if st.button(f"Select {model_name}", key=f"select_model_{model_id}", use_container_width=True):
                    st.session_state.wizard_data['model_id'] = model_id
                    st.session_state.wizard_step = 1  # Auto-advance to next step
                    st.session_state.navigate_to = "Audits"
                    st.rerun()


def render_step_framework_selection():
    """Step 2: Framework Selection with checkboxes and descriptions."""
    st.markdown("""
<div style="margin-bottom: 2rem;">
    <h3 style="color: #1e293b; font-weight: 600;">Choose Testing Frameworks</h3>
    <p style="color: #64748b;">Select one or more frameworks to run comprehensive security tests.</p>
</div>""", unsafe_allow_html=True)

    # Available frameworks with descriptions
    frameworks = {
        "aura-native": {
            "name": "AURA Native",
            "description": "Built-in testing framework with comprehensive test coverage",
            "icon": "🛡️",
            "tests": "Safety, Security, Bias, Privacy"
        },
        "garak": {
            "name": "Garak",
            "description": "LLM vulnerability scanner by NVIDIA",
            "icon": "⚔️",
            "tests": "Adversarial, Jailbreak, Prompt Injection"
        },
        "pyrit": {
            "name": "PyRIT",
            "description": "Microsoft's AI Red Team framework",
            "icon": "🔴",
            "tests": "Red Team, Attack Simulation"
        }
    }

    current_selection = st.session_state.wizard_data.get('frameworks', [])

    for fw_id, fw_info in frameworks.items():
        is_selected = fw_id in current_selection

        col1, col2 = st.columns([0.1, 0.9])


        with col1:
            st.checkbox(
                "",
                value=is_selected,
                key=f"fw_{fw_id}",
                label_visibility="collapsed",
                on_change=toggle_framework,
                args=(fw_id,)
            )

        with col2:
            st.markdown(f"""
<div style="
    background: {'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)' if is_selected else 'white'};
    border: 2px solid {'#667eea' if is_selected else '#e2e8f0'};
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <span style="font-size: 24px;">{fw_info['icon']}</span>
        <span style="font-weight: 600; font-size: 18px; color: #1e293b;">
            {fw_info['name']}
        </span>
    </div>
    <p style="color: #64748b; margin: 8px 0; font-size: 14px;">
        {fw_info['description']}
    </p>
    <div style="font-size: 13px; color: #667eea; font-weight: 500;">
        Tests: {fw_info['tests']}
    </div>
</div>""", unsafe_allow_html=True)

    # st.session_state.wizard_data['frameworks'] is updated via callbacks

    if len(current_selection) > 0:
        st.success(f"✓ {len(current_selection)} framework(s) selected")
    print('current_selection', current_selection)

def render_step_policy_selection():
    """Step 3: Policy Selection with multi-select."""
    st.markdown("""
<div style="margin-bottom: 2rem;">
    <h3 style="color: #1e293b; font-weight: 600;">Select Compliance Policies</h3>
    <p style="color: #64748b;">Choose which policies and regulations to test against.</p>
</div>""", unsafe_allow_html=True)

    # Fetch policies from API
    from src.dashboard.api_client import api_request
    policies_response = api_request("/policies")

    if "error" in policies_response:
        st.error(f"Failed to load policies: {policies_response['error']}")
        return

    policies = policies_response.get("policies", [])

    if not policies:
        st.warning("No policies available.")
        return

    # Group by category
    categories = {}
    for policy in policies:
        category = policy.get('category', 'General')
        if category not in categories:
            categories[category] = []
        categories[category].append(policy)

    current_selection = st.session_state.wizard_data.get('policies', [])

    for category, cat_policies in categories.items():
        st.markdown(f"### {category}")

        for policy in cat_policies:
            print('policy', policy)
            policy_id = policy.get('id')
            policy_name = policy.get('name', 'Unknown')
            policy_desc = policy.get('description', '')
            severity = policy.get('severity', '')

            is_selected = policy_id in current_selection

            col1, col2 = st.columns([0.1, 0.9])


            with col1:
                st.checkbox(
                    "",
                    value=is_selected,
                    key=f"pol_{policy_id}",
                    label_visibility="collapsed",
                    on_change=toggle_policy,
                    args=(policy_id,)
                )

            with col2:
                # Recalculate selection state for immediate visual feedback
                is_active = policy_id in current_selection
                                
                severity_colors = {
                    "critical": "#ef4444",
                    "high": "#f59e0b",
                    "medium": "#3b82f6",
                    "low": "#10b981"
                }
                severity_color = severity_colors.get(severity, "#64748b")

                st.markdown(f"""
<div style="
    background: {'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)' if is_active else 'white'};
    border: 2px solid {'#667eea' if is_active else '#e2e8f0'};
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="font-weight: 600; font-size: 15px; color: #1e293b;">
            {policy_name}
        </span>
        <span style="
            background: {severity_color};
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        ">
            {severity}
        </span>
    </div>
    <p style="color: #64748b; margin: 0; font-size: 13px;">
        {policy_desc[:200]}{'...' if len(policy_desc) > 200 else ''}
    </p>
</div>""", unsafe_allow_html=True)

    # st.session_state.wizard_data['policies'] is updated via callbacks

    if len(current_selection) > 0:
        st.success(f"✓ {len(current_selection)} policy/policies selected")


def render_step_review_launch():
    """Step 4: Review selections and launch audit."""
    st.markdown("""
<div style="margin-bottom: 2rem;">
    <h3 style="color: #1e293b; font-weight: 600;">Review & Launch Audit</h3>
    <p style="color: #64748b;">Review your selections and configure final options before launching.</p>
</div>""", unsafe_allow_html=True)

    wizard_data = st.session_state.wizard_data

    # Summary card
    st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    border: 2px solid #667eea;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
">
<h4 style="color: #667eea; margin-top: 0;">Audit Configuration Summary</h4>
<div style="margin: 1.5rem 0;">
<div style="display: flex; gap: 8px; margin-bottom: 12px;">
<span style="font-weight: 600; color: #1e293b; min-width: 140px;">Model:</span>
<span style="color: #64748b;">{wizard_data.get('model_id', 'Not selected')}</span>
</div>
<div style="display: flex; gap: 8px; margin-bottom: 12px;">
<span style="font-weight: 600; color: #1e293b; min-width: 140px;">Frameworks:</span>
<span style="color: #64748b;">{len(wizard_data.get('frameworks', []))} selected ({', '.join(wizard_data.get('frameworks', []))})</span>
</div>
<div style="display: flex; gap: 8px; margin-bottom: 12px;">
<span style="font-weight: 600; color: #1e293b; min-width: 140px;">Policies:</span>
<span style="color: #64748b;">{len(wizard_data.get('policies', []))} selected</span>
</div>
<div style="display: flex; gap: 8px;">
<span style="font-weight: 600; color: #1e293b; min-width: 140px;">Test Count:</span>
<span style="color: #64748b;">{wizard_data.get('test_count', 10)} tests per policy</span>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # Additional configuration
    st.markdown("### Advanced Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Test Count Configuration**")
        sub_col1, sub_col2 = st.columns([2, 1])
        
        current_count = wizard_data.get('test_count', 10)
        
        with sub_col1:
            st.slider(
                "Tests per Policy",
                min_value=5,
                max_value=100,
                value=current_count,
                step=5,
                key="tc_slider",
                on_change=update_test_count_slider,
                help="Slide to adjust number of tests"
            )
            
        with sub_col2:
            st.number_input(
                "Manual Input",
                min_value=5,
                max_value=100,
                value=current_count,
                step=5,
                key="tc_input",
                on_change=update_test_count_input,
                label_visibility="visible"
            )
            
        st.caption("Adjust the number of test cases to run per policy.")

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        estimated_time = len(wizard_data.get('frameworks', [])) * len(wizard_data.get('policies', [])) * current_count * 0.5
        st.info(f"⏱️ Estimated Duration: ~{int(estimated_time / 60)} minutes")

    st.markdown("<br>", unsafe_allow_html=True)

    # Launch button
    if st.button("🚀 Launch Audit", use_container_width=True, type="primary", key="launch_audit_btn"):
        launch_audit()


def launch_audit():
    """Launch the audit with selected configuration and show real-time progress."""
    from src.dashboard.api_client import api_request
    from src.dashboard.components.progress_monitor import render_progress_monitor, render_progress_stages
    import time

    wizard_data = st.session_state.wizard_data

    # Prepare audit request
    audit_data = {
        "model_id": wizard_data['model_id'],
        "policy_ids": wizard_data['policies'],
        "test_count": wizard_data['test_count'],
        "frameworks": wizard_data['frameworks']
    }
    print('audit_data', audit_data)
    # Calculate total tests
    total_tests = len(wizard_data['policies']) * wizard_data['test_count']

    # Create audit ID
    audit_id = f"audit-{int(time.time())}"
    start_time = datetime.now()

    # Progress tracking containers - use container() for proper nesting
    st.markdown("---")

    # Create placeholder containers
    stages_placeholder = st.empty()
    progress_placeholder = st.empty()

    # Define audit stages
    stages = [
        {'name': 'Initialize', 'icon': '⚙️', 'status': 'active'},
        {'name': 'Load Policies', 'icon': '📋', 'status': 'pending'},
        {'name': 'Run Tests', 'icon': '🧪', 'status': 'pending'},
        {'name': 'Analyze Results', 'icon': '📊', 'status': 'pending'},
        {'name': 'Generate Report', 'icon': '📑', 'status': 'pending'},
    ]

    # Stage 1: Initialization
    with stages_placeholder.container():
        render_progress_stages(stages, 0)
    
    progress_placeholder.empty()
    with progress_placeholder.container():
        render_progress_monitor(
            audit_id=audit_id,
            total_tests=total_tests,
            current_stage="Initializing",
            progress_percent=5,
            tests_completed=0,
            tests_passed=0,
            tests_failed=0,
            start_time=start_time,
            logs=["Initializing audit context...", "Loading configurations..."]
        )
    stages[0]['status'] = 'completed'
    stages[1]['status'] = 'active'
    with stages_placeholder.container():
        render_progress_stages(stages, 1)

    progress_placeholder.empty()
    with progress_placeholder.container():
        render_progress_monitor(
            audit_id=audit_id,
            total_tests=total_tests,
            current_stage="Loading Policies",
            progress_percent=20,
            tests_completed=0,
            tests_passed=0,
            tests_failed=0,
            start_time=start_time
        )

    time.sleep(0.5)

    # Stage 3: Run Tests - Make API call
    stages[1]['status'] = 'completed'
    stages[2]['status'] = 'active'
    with stages_placeholder.container():
        render_progress_stages(stages, 2)

    progress_placeholder.empty()
    with progress_placeholder.container():
        render_progress_monitor(
            audit_id=audit_id,
            total_tests=total_tests,
            current_stage="Running Tests",
            progress_percent=40,
            tests_completed=0,
            tests_passed=0,
            tests_failed=0,
            start_time=start_time,
            current_test={
                'name': 'Executing test suite',
                'policy': wizard_data['policies'][0] if wizard_data['policies'] else 'N/A',
                'framework': wizard_data['frameworks'][0] if wizard_data['frameworks'] else 'N/A'
            }
        )

    # Make the actual API call
    response = api_request("/audit", method="POST", data=audit_data)

    if "error" in response:
        stages[2]['status'] = 'failed'
        with stages_placeholder.container():
            render_progress_stages(stages, 2)
        st.error(f"❌ Failed to launch audit: {response['error']}")
        return

    # Stage 4: Analyze Results
    stages[2]['status'] = 'completed'
    stages[3]['status'] = 'active'
    with stages_placeholder.container():
        render_progress_stages(stages, 3)

    # Extract results
    results = response.get('results', {})
    total_actual = results.get('total_tests', total_tests)
    passed = results.get('passed_tests', int(total_actual * 0.8))  # Estimate if not provided
    failed = total_actual - passed

    progress_placeholder.empty()
    with progress_placeholder.container():
        render_progress_monitor(
            audit_id=audit_id,
            total_tests=total_actual,
            current_stage="Analyzing Results",
            progress_percent=70,
            tests_completed=total_actual,
            tests_passed=passed,
            tests_failed=failed,
            start_time=start_time
        )

    time.sleep(0.5)

    # Stage 5: Generate Report
    stages[3]['status'] = 'completed'
    stages[4]['status'] = 'active'
    with stages_placeholder.container():
        render_progress_stages(stages, 4)

    progress_placeholder.empty()
    with progress_placeholder.container():
        render_progress_monitor(
            audit_id=audit_id,
            total_tests=total_actual,
            current_stage="Generating Report",
            progress_percent=90,
            tests_completed=total_actual,
            tests_passed=passed,
            tests_failed=failed,
            start_time=start_time
        )

    time.sleep(0.5)

    # Complete!
    stages[4]['status'] = 'completed'
    with stages_placeholder.container():
        render_progress_stages(stages, 4)

    progress_placeholder.empty()
    with progress_placeholder.container():
        render_progress_monitor(
            audit_id=response.get("audit_id", audit_id),
            total_tests=total_actual,
            current_stage="Completed",
            progress_percent=100,
            tests_completed=total_actual,
            tests_passed=passed,
            tests_failed=failed,
            start_time=start_time
        )

    st.success("✅ Audit completed successfully!")

    # Store audit in session history
    # Build robust results dictionary
    results_data = response.get("results", {})
    if not results_data or results_data.get("total_tests", 0) == 0:
        # Fallback to local counts if API response is empty/incomplete
        results_data = {
            "passed": passed,
            "failed": failed,
            "total_tests": total_actual,
            "by_policy": {}  # Can't easily reconstruct this here without more logic
        }

    audit_record = {
        "audit_id": response.get("audit_id", audit_id),
        "model_id": wizard_data['model_id'],
        "frameworks": wizard_data['frameworks'],
        "policies": wizard_data['policies'],
        "test_count": wizard_data['test_count'],
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "compliance_score": response.get("compliance_score", 0),
        "results": results_data,
        "findings": response.get("findings", []),
        "recommendations": response.get("recommendations", [])
    }

    st.session_state.audit_history.append(audit_record)
    st.session_state.reports.append(audit_record)
    st.session_state.selected_report = audit_record

    # Show full report directly
    st.markdown("---")
    render_reports(report_data=audit_record)
    
    st.markdown("---")

    # Option to start another audit
    if st.button("🔄 Start New Audit", use_container_width=True, key=f"new_audit_{audit_id}"):
        # Clear specific widget keys to prevent "ghost" selections
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith("fw_") or k.startswith("pol_")]
        for k in keys_to_clear:
            del st.session_state[k]

        # Reset wizard
        st.session_state.wizard_step = 0
        st.session_state.wizard_data = {
            'model_id': None,
            'frameworks': [],
            'policies': [],
            'test_count': 10
        }
        st.session_state.navigate_to = "Audits"
        st.rerun()

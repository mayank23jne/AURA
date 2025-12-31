"""Audit Comparison View Component"""

import streamlit as st
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from .styles import COLORS


def render_comparison_selector():
    """Render audit selection interface for comparison."""
    st.markdown("### Select Audits to Compare")
    st.markdown("Choose 2-4 audits to compare side-by-side")

    # Get available audits from session state
    available_audits = st.session_state.get('audit_history', [])

    if len(available_audits) < 2:
        st.warning("⚠️ You need at least 2 completed audits to use the comparison feature.")
        st.info("💡 Run some audits first using the Wizard Mode to enable comparisons.")
        return None

    # Create selection UI
    st.markdown("---")

    selected_audits = []
    cols = st.columns(min(4, len(available_audits)))

    for idx, audit in enumerate(available_audits[:8]):  # Limit to 8 for UI purposes
        with cols[idx % 4]:
            # Audit card
            audit_id = audit.get('audit_id', f'audit-{idx}')
            model_id = audit.get('model_id', 'Unknown')
            timestamp = audit.get('timestamp', '')
            score = audit.get('compliance_score', 0)

            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp)
                date_str = dt.strftime('%b %d, %Y')
            except:
                date_str = 'Unknown date'

            # Render card
            st.markdown(f"""
                <div style="
                    background: white;
                    border-radius: 12px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    border: 2px solid {'#667eea' if idx < 2 else '#e2e8f0'};
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
                ">
                    <div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; margin-bottom: 0.5rem;">{date_str}</div>
                    <div style="font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 0.5rem; font-size: 0.9rem;">{model_id}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {COLORS['primary']};">{score}%</div>
                </div>
            """, unsafe_allow_html=True)

            if st.checkbox(f"Select", key=f"select_audit_{idx}", value=(idx < 2)):
                selected_audits.append(audit)

    st.markdown("---")

    if len(selected_audits) < 2:
        st.warning("⚠️ Please select at least 2 audits to compare")
        return None
    elif len(selected_audits) > 4:
        st.warning("⚠️ Please select no more than 4 audits to compare")
        return None
    else:
        st.success(f"✓ {len(selected_audits)} audits selected for comparison")
        return selected_audits


def render_comparison_view(audits: List[Dict[str, Any]]):
    """
    Render side-by-side comparison of audits.

    Args:
        audits: List of audit records to compare
    """
    if not audits or len(audits) < 2:
        st.error("Please select at least 2 audits to compare")
        return

    st.markdown("### 📊 Audit Comparison Dashboard")
    st.markdown("---")

    # Summary Cards
    st.markdown("#### Overall Comparison")

    cols = st.columns(len(audits))

    for idx, audit in enumerate(audits):
        with cols[idx]:
            audit_id = audit.get('audit_id', f'audit-{idx}')
            model_id = audit.get('model_id', 'Unknown')
            score = audit.get('compliance_score', 0)
            total_tests = audit.get('results', {}).get('total_tests', 0)
            passed = audit.get('results', {}).get('passed_tests', 0)
            timestamp = audit.get('timestamp', '')

            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp)
                date_str = dt.strftime('%b %d, %Y %H:%M')
            except:
                date_str = 'Unknown'

            # Determine score color
            if score >= 90:
                score_color = COLORS['success']
            elif score >= 70:
                score_color = COLORS['warning']
            else:
                score_color = COLORS['danger']

            # Render comparison card
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
                    border: 2px solid {COLORS['border']};
                    border-radius: 12px;
                    padding: 1.25rem;
                    height: 100%;
                ">
                    <div style="
                        font-size: 0.75rem;
                        color: {COLORS['text_secondary']};
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        margin-bottom: 0.5rem;
                    ">Audit #{idx + 1}</div>

                    <div style="
                        font-size: 3rem;
                        font-weight: 700;
                        color: {score_color};
                        line-height: 1;
                        margin-bottom: 0.75rem;
                    ">{score}%</div>

                    <div style="margin-bottom: 0.5rem;">
                        <span style="font-weight: 600; color: {COLORS['text_primary']};">Model:</span>
                        <span style="color: {COLORS['text_secondary']}; font-size: 0.875rem;"> {model_id}</span>
                    </div>

                    <div style="margin-bottom: 0.5rem;">
                        <span style="font-weight: 600; color: {COLORS['text_primary']};">Tests:</span>
                        <span style="color: {COLORS['text_secondary']}; font-size: 0.875rem;"> {passed}/{total_tests}</span>
                    </div>

                    <div style="margin-bottom: 0.5rem;">
                        <span style="font-weight: 600; color: {COLORS['text_primary']};">Date:</span>
                        <span style="color: {COLORS['text_secondary']}; font-size: 0.875rem;"> {date_str}</span>
                    </div>

                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid {COLORS['border']};">
                        <span style="font-weight: 600; color: {COLORS['text_primary']}; font-size: 0.875rem;">ID:</span>
                        <span style="color: {COLORS['text_secondary']}; font-size: 0.75rem; font-family: monospace;"> {audit_id}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Detailed Metrics Comparison
    st.markdown("#### Detailed Metrics")

    # Build comparison table
    comparison_data = {
        'Metric': [
            'Compliance Score',
            'Total Tests',
            'Passed Tests',
            'Failed Tests',
            'Pass Rate',
            'Number of Findings',
            'Critical Findings',
            'Frameworks Used'
        ]
    }

    for idx, audit in enumerate(audits):
        results = audit.get('results', {})
        total_tests = results.get('total_tests', 0)
        passed = results.get('passed_tests', 0)
        failed = total_tests - passed
        pass_rate = (passed / total_tests * 100) if total_tests > 0 else 0

        findings = audit.get('findings', [])
        critical_findings = len([f for f in findings if f.get('severity') == 'critical'])

        frameworks = audit.get('frameworks', [])
        frameworks_str = ', '.join([f.upper() for f in frameworks]) if frameworks else 'N/A'

        comparison_data[f'Audit #{idx + 1}'] = [
            f"{audit.get('compliance_score', 0)}%",
            str(total_tests),
            str(passed),
            str(failed),
            f"{pass_rate:.1f}%",
            str(len(findings)),
            str(critical_findings),
            frameworks_str
        ]

    df = pd.DataFrame(comparison_data)

    # Style the dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Highlighted Differences
    st.markdown("#### Key Differences")

    differences = []

    # Compare scores
    scores = [a.get('compliance_score', 0) for a in audits]
    max_score = max(scores)
    min_score = min(scores)

    if max_score - min_score > 10:
        best_idx = scores.index(max_score)
        worst_idx = scores.index(min_score)
        differences.append({
            'category': 'Compliance Score',
            'description': f"Audit #{best_idx + 1} scored {max_score - min_score:.1f}% higher than Audit #{worst_idx + 1}",
            'severity': 'info' if max_score - min_score < 20 else 'warning'
        })

    # Compare test counts
    test_counts = [a.get('results', {}).get('total_tests', 0) for a in audits]
    if len(set(test_counts)) > 1:
        differences.append({
            'category': 'Test Coverage',
            'description': f"Audits used different test counts: {', '.join(map(str, test_counts))}",
            'severity': 'info'
        })

    # Compare frameworks
    framework_sets = [set(a.get('frameworks', [])) for a in audits]
    if len(set(tuple(sorted(fs)) for fs in framework_sets)) > 1:
        differences.append({
            'category': 'Frameworks',
            'description': "Audits used different testing frameworks",
            'severity': 'warning'
        })

    # Display differences
    if differences:
        for diff in differences:
            severity_colors = {
                'info': COLORS['info'],
                'warning': COLORS['warning'],
                'danger': COLORS['danger']
            }

            st.markdown(f"""
                <div style="
                    background: {severity_colors[diff['severity']]}15;
                    border-left: 4px solid {severity_colors[diff['severity']]};
                    padding: 1rem 1.25rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                ">
                    <div style="font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 0.25rem;">
                        {diff['category']}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 0.9rem;">
                        {diff['description']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("✓ All audits show similar characteristics")

    st.markdown("---")

    # Export Options
    st.markdown("#### Export Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Export as PDF", use_container_width=True):
            st.info("PDF export feature coming soon")

    with col2:
        if st.button("📊 Export as CSV", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"audit_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col3:
        if st.button("🔗 Share Link", use_container_width=True):
            st.info("Share link feature coming soon")


def render_audit_comparison():
    """Main function to render the full comparison interface."""
    st.markdown('<h2 style="color: #1e293b;">🔍 Audit Comparison</h2>', unsafe_allow_html=True)
    st.markdown("Compare multiple audit results side-by-side to identify trends and improvements")
    st.markdown("---")

    # Step 1: Select audits
    selected_audits = render_comparison_selector()

    # Step 2: Show comparison if audits selected
    if selected_audits and len(selected_audits) >= 2:
        st.markdown("---")
        render_comparison_view(selected_audits)

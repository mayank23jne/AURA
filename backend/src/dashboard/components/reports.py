"""
Report rendering component for AURA dashboard.
"""
import json
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO

# PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

from .styles import get_score_class
from .ui_components import show_empty_state


def generate_report_json(report: dict) -> str:
    """Generate JSON export of report."""
    return json.dumps(report, indent=2, default=str)


def generate_report_markdown(report: dict) -> str:
    """Generate Markdown export of report."""
    md = f"""# Audit Report: {report.get('audit_id', 'Unknown')}

## Summary
- **Model ID**: {report.get('model_id', 'Unknown')}
- **Date**: {report.get('timestamp', 'Unknown')}
- **Compliance Score**: {report.get('compliance_score', 0) * 100:.1f}%
- **Status**: {report.get('status', 'Unknown')}

## Results
- **Total Tests**: {report.get('results', {}).get('total_tests', 0)}
- **Passed**: {report.get('results', {}).get('passed', 0)}
- **Failed**: {report.get('results', {}).get('failed', 0)}

## Findings
"""
    for finding in report.get('findings', []):
        md += f"\n### {finding.get('severity', 'Unknown').upper()}: {finding.get('policy', 'Unknown')}\n"
        md += f"{finding.get('description', 'No description')}\n"

    md += "\n## Recommendations\n"
    for rec in report.get('recommendations', []):
        md += f"- {rec}\n"

    return md


def generate_report_pdf(report: dict) -> bytes:
    """Generate comprehensive PDF export of audit report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=20,
        textColor=colors.HexColor('#1f77b4')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#333333')
    )
    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.HexColor('#555555')
    )
    normal_style = styles['Normal']
    small_style = ParagraphStyle(
        'Small',
        parent=normal_style,
        fontSize=8,
        textColor=colors.HexColor('#666666')
    )

    # Build content
    content = []

    # Title
    content.append(Paragraph("AURA Compliance Audit Report", title_style))
    content.append(Spacer(1, 10))

    # Executive Summary Box
    content.append(Paragraph("Executive Summary", heading_style))
    content.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1f77b4')))
    content.append(Spacer(1, 10))

    # Calculate metrics
    score = report.get('compliance_score', 0)
    score_pct = score * 100 if score <= 1 else score
    results = report.get('results', {})
    total_tests = results.get('total_tests', 0)
    passed = results.get('passed', 0)
    failed = results.get('failed', 0)
    findings = report.get('findings', [])

    # Count findings by severity
    critical_count = sum(1 for f in findings if f.get('severity', '').lower() == 'critical')
    high_count = sum(1 for f in findings if f.get('severity', '').lower() == 'high')
    medium_count = sum(1 for f in findings if f.get('severity', '').lower() == 'medium')
    low_count = sum(1 for f in findings if f.get('severity', '').lower() == 'low')

    # Executive summary table
    exec_data = [
        ['Audit ID:', report.get('audit_id', 'Unknown'), 'Model ID:', report.get('model_id', 'Unknown')],
        ['Date:', report.get('timestamp', 'Unknown')[:19] if report.get('timestamp') else 'Unknown', 'Status:', report.get('status', 'Unknown').upper()],
        ['Compliance Score:', f"{score_pct:.1f}%", 'Pass Rate:', f"{(passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A"],
        ['Total Tests:', str(total_tests), 'Total Findings:', str(len(findings))],
    ]

    exec_table = Table(exec_data, colWidths=[1.3*inch, 1.7*inch, 1.3*inch, 1.7*inch])
    exec_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
    ]))
    content.append(exec_table)
    content.append(Spacer(1, 15))

    # Findings Summary by Severity
    if findings:
        severity_data = [
            ['Severity', 'Count', 'Percentage'],
            ['Critical', str(critical_count), f"{(critical_count/len(findings)*100):.0f}%" if findings else "0%"],
            ['High', str(high_count), f"{(high_count/len(findings)*100):.0f}%" if findings else "0%"],
            ['Medium', str(medium_count), f"{(medium_count/len(findings)*100):.0f}%" if findings else "0%"],
            ['Low', str(low_count), f"{(low_count/len(findings)*100):.0f}%" if findings else "0%"],
        ]

        severity_table = Table(severity_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        severity_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ffe6e6')),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#fff3e6')),
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#fffde6')),
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#e6ffe6')),
        ]))
        content.append(severity_table)
    content.append(Spacer(1, 20))

    # Testing Frameworks Section
    content.append(Paragraph("Testing Frameworks Used", heading_style))
    content.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1f77b4')))
    content.append(Spacer(1, 10))

    framework_results = results.get('framework_results', {})
    if framework_results:
        # Framework descriptions
        framework_info = {
            'pyrit': 'Microsoft PyRIT - Python Risk Identification Tool for AI red-teaming and adversarial testing',
            'aura-native': 'AURA Native Framework - Built-in compliance and safety testing suite',
            'aiverify': 'AI Verify - Comprehensive AI governance and fairness testing framework'
        }

        # Create framework summary table
        fw_data = [['Framework', 'Tests Run', 'Passed', 'Failed', 'Pass Rate']]

        for fw_name, fw_result in framework_results.items():
            fw_tests = fw_result.get('tests', [])
            fw_total = len(fw_tests)
            fw_passed = sum(1 for t in fw_tests if t.get('passed', False))
            fw_failed = fw_total - fw_passed
            fw_pass_rate = f"{(fw_passed/fw_total*100):.1f}%" if fw_total > 0 else "N/A"

            fw_display_name = fw_name.upper().replace('-', ' ')
            fw_data.append([fw_display_name, str(fw_total), str(fw_passed), str(fw_failed), fw_pass_rate])

        fw_table = Table(fw_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        fw_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]))
        content.append(fw_table)
        content.append(Spacer(1, 10))

        # Framework descriptions
        content.append(Paragraph("Framework Capabilities:", subheading_style))
        content.append(Spacer(1, 5))

        for fw_name in framework_results.keys():
            fw_desc = framework_info.get(fw_name, f'{fw_name.upper()} - Advanced AI testing framework')
            desc_style = ParagraphStyle('FWDesc', parent=normal_style, fontSize=9, leftIndent=15, spaceAfter=5)
            content.append(Paragraph(f"<b>{fw_name.upper().replace('-', ' ')}:</b> {fw_desc}", desc_style))

    else:
        content.append(Paragraph("No framework information available for this audit.", normal_style))

    content.append(Spacer(1, 20))

    # Detailed Test Results Section
    content.append(Paragraph("Detailed Test Results", heading_style))
    content.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1f77b4')))
    content.append(Spacer(1, 10))

    test_results = results.get('test_results', [])
    if test_results:
        # Create table header
        test_header = ['Test Name', 'Policy', 'Severity', 'Score', 'Result']
        test_data = [test_header]

        for test in test_results:
            test_name = test.get('test_name', test.get('test_id', 'Unknown'))[:25]
            policy = test.get('policy_name', test.get('policy_id', 'Unknown'))[:20]
            severity = test.get('severity', 'medium').upper()
            score = test.get('score', 0)
            passed_status = 'PASS' if test.get('passed', False) else 'FAIL'
            test_data.append([test_name, policy, severity, f"{score:.2f}", passed_status])

        test_table = Table(test_data, colWidths=[1.5*inch, 1.3*inch, 1*inch, 0.8*inch, 0.8*inch])
        test_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ALIGN', (3, 0), (4, -1), 'CENTER'),
        ]))

        # Color code results
        for i, test in enumerate(test_results, 1):
            if not test.get('passed', False):
                test_table.setStyle(TableStyle([
                    ('BACKGROUND', (4, i), (4, i), colors.HexColor('#ffcccc')),
                ]))
            else:
                test_table.setStyle(TableStyle([
                    ('BACKGROUND', (4, i), (4, i), colors.HexColor('#ccffcc')),
                ]))

        content.append(test_table)
        content.append(Spacer(1, 10))

        # Add detailed test information for failed tests
        failed_tests = [t for t in test_results if not t.get('passed', False)]
        if failed_tests:
            content.append(Paragraph("Failed Test Details", subheading_style))
            content.append(Spacer(1, 5))

            for test in failed_tests[:5]:  # Limit to first 5 for readability
                test_name = test.get('test_name', test.get('test_id', 'Unknown'))
                content.append(Paragraph(f"<b>{test_name}</b>", normal_style))

                # Test details
                detail_style = ParagraphStyle('Detail', parent=small_style, leftIndent=15)

                if test.get('rule_text'):
                    content.append(Paragraph(f"Rule: {test.get('rule_text', '')[:100]}", detail_style))
                if test.get('prompt'):
                    prompt_text = test.get('prompt', '')[:150]
                    content.append(Paragraph(f"Prompt: {prompt_text}...", detail_style))
                if test.get('response'):
                    response_text = test.get('response', '')[:150]
                    content.append(Paragraph(f"Response: {response_text}...", detail_style))
                if test.get('evaluation'):
                    content.append(Paragraph(f"Evaluation: {test.get('evaluation', '')[:100]}", detail_style))

                content.append(Spacer(1, 8))
    else:
        content.append(Paragraph("No test results available.", normal_style))

    content.append(Spacer(1, 15))

    # Detailed Findings Section
    content.append(Paragraph("Detailed Findings", heading_style))
    content.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1f77b4')))
    content.append(Spacer(1, 10))

    if not findings:
        content.append(Paragraph("No compliance findings detected. The model appears to meet all tested policy requirements.", normal_style))
    else:
        severity_colors = {
            'critical': colors.HexColor('#dc3545'),
            'high': colors.HexColor('#fd7e14'),
            'medium': colors.HexColor('#ffc107'),
            'low': colors.HexColor('#28a745')
        }

        for i, finding in enumerate(findings, 1):
            severity = finding.get('severity', 'medium').lower()
            sev_color = severity_colors.get(severity, colors.HexColor('#6c757d'))

            # Finding header with number and severity
            sev_style = ParagraphStyle(
                f'Severity_{i}',
                parent=styles['Heading4'],
                textColor=sev_color,
                fontSize=11,
                spaceBefore=10,
                spaceAfter=5
            )

            finding_title = finding.get('title', finding.get('policy', f'Finding {i}'))
            content.append(Paragraph(f"Finding {i}: [{severity.upper()}] {finding_title}", sev_style))

            # Finding details in a box
            detail_style = ParagraphStyle('FindingDetail', parent=normal_style, fontSize=9, leftIndent=15, spaceAfter=3)

            # Description
            description = finding.get('description', 'No description provided')
            content.append(Paragraph(f"<b>Description:</b> {description}", detail_style))

            # Policy information
            if finding.get('policy_name') or finding.get('policy_id'):
                policy_info = finding.get('policy_name', finding.get('policy_id', 'Unknown'))
                content.append(Paragraph(f"<b>Policy:</b> {policy_info}", detail_style))

            # Impact assessment
            if finding.get('impact'):
                content.append(Paragraph(f"<b>Impact:</b> {finding.get('impact')}", detail_style))

            # Confidence level
            if finding.get('confidence'):
                confidence = finding.get('confidence', 0)
                conf_pct = confidence * 100 if confidence <= 1 else confidence
                content.append(Paragraph(f"<b>Confidence:</b> {conf_pct:.0f}%", detail_style))

            # Affected tests
            affected_tests = finding.get('affected_tests', [])
            if affected_tests:
                tests_str = ', '.join(affected_tests[:5])
                if len(affected_tests) > 5:
                    tests_str += f" (+{len(affected_tests)-5} more)"
                content.append(Paragraph(f"<b>Affected Tests:</b> {tests_str}", detail_style))

            # Failed rules
            failed_rules = finding.get('failed_rules', [])
            if failed_rules:
                rules_str = '; '.join(failed_rules[:3])
                if len(failed_rules) > 3:
                    rules_str += f" (+{len(failed_rules)-3} more)"
                content.append(Paragraph(f"<b>Failed Rules:</b> {rules_str}", detail_style))

            content.append(Spacer(1, 8))

    content.append(Spacer(1, 15))

    # Recommendations Section
    content.append(Paragraph("Recommendations", heading_style))
    content.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1f77b4')))
    content.append(Spacer(1, 10))

    recommendations = report.get('recommendations', [])
    if not recommendations:
        content.append(Paragraph("No specific recommendations at this time. Continue monitoring for compliance.", normal_style))
    else:
        for i, rec in enumerate(recommendations, 1):
            # Handle both string and dict recommendations
            if isinstance(rec, dict):
                priority = rec.get('priority', 'medium').upper()
                title = rec.get('title', f'Recommendation {i}')
                description = rec.get('description', '')

                # Priority colors
                priority_colors = {
                    'HIGH': colors.HexColor('#dc3545'),
                    'MEDIUM': colors.HexColor('#ffc107'),
                    'LOW': colors.HexColor('#28a745')
                }
                p_color = priority_colors.get(priority, colors.HexColor('#6c757d'))

                rec_header_style = ParagraphStyle(
                    f'RecHeader_{i}',
                    parent=normal_style,
                    fontSize=10,
                    textColor=p_color,
                    spaceBefore=8
                )
                rec_detail_style = ParagraphStyle(
                    f'RecDetail_{i}',
                    parent=normal_style,
                    fontSize=9,
                    leftIndent=15,
                    spaceAfter=5
                )

                content.append(Paragraph(f"<b>{i}. [{priority}] {title}</b>", rec_header_style))
                if description:
                    content.append(Paragraph(description, rec_detail_style))
            else:
                rec_style = ParagraphStyle(
                    f'Rec_{i}',
                    parent=normal_style,
                    fontSize=9,
                    leftIndent=15,
                    spaceBefore=5,
                    spaceAfter=5
                )
                content.append(Paragraph(f"<b>{i}.</b> {rec}", rec_style))

    content.append(Spacer(1, 15))

    # Remediation Suggestions Section
    remediation = report.get('remediation_suggestions', [])
    if remediation:
        content.append(Paragraph("Remediation Suggestions", heading_style))
        content.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1f77b4')))
        content.append(Spacer(1, 10))

        for i, rem in enumerate(remediation, 1):
            if isinstance(rem, dict):
                suggestion = rem.get('suggestion', '')
                effort = rem.get('effort', 'medium').upper()
                priority = rem.get('priority', 'medium').upper()

                rem_style = ParagraphStyle(
                    f'Rem_{i}',
                    parent=normal_style,
                    fontSize=9,
                    leftIndent=15,
                    spaceBefore=5,
                    spaceAfter=3
                )

                content.append(Paragraph(f"<b>{i}. {suggestion}</b>", rem_style))
                content.append(Paragraph(f"   Effort: {effort} | Priority: {priority}", small_style))
            else:
                content.append(Paragraph(f"{i}. {rem}", normal_style))

        content.append(Spacer(1, 15))

    # Audit Trail / Stages Completed
    stages = report.get('stages_completed', [])
    if stages:
        content.append(Paragraph("Audit Trail", heading_style))
        content.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1f77b4')))
        content.append(Spacer(1, 10))

        for stage in stages:
            stage_style = ParagraphStyle('Stage', parent=small_style, leftIndent=10)
            content.append(Paragraph(f"• {stage}", stage_style))

        content.append(Spacer(1, 15))

    # Errors Section (if any)
    errors = report.get('errors', [])
    if errors:
        content.append(Paragraph("Errors Encountered", heading_style))
        content.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#dc3545')))
        content.append(Spacer(1, 10))

        for error in errors:
            error_style = ParagraphStyle('Error', parent=normal_style, fontSize=9, textColor=colors.HexColor('#dc3545'))
            content.append(Paragraph(f"• {error}", error_style))

        content.append(Spacer(1, 15))

    # Footer
    content.append(Spacer(1, 20))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#dee2e6')))
    footer_style = ParagraphStyle(
        'Footer',
        parent=normal_style,
        fontSize=8,
        textColor=colors.grey,
        alignment=1  # Center
    )
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Generated by AURA Platform on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
    content.append(Paragraph("This report is confidential and intended for authorized personnel only.", footer_style))

    # Build PDF
    doc.build(content)
    buffer.seek(0)
    return buffer.getvalue()


def render_reports(report_data: dict = None):
    """Render reports page or a specific report if detailed."""
    if not report_data:
        st.markdown('<p class="main-header">📑 Reports</p>', unsafe_allow_html=True)

        if not st.session_state.reports:
            show_empty_state(
                icon="📑",
                title="No Reports Available",
                description="Complete an audit to generate your first report. Reports will appear here once audits are completed."
            )
            return

        # Report selector
        # Sort so newest is first
        sorted_reports = sorted(st.session_state.reports, key=lambda r: r.get('timestamp', ''), reverse=True)
        report_options = {f"{r.get('audit_id', 'Unknown')} - {r.get('model_id', 'Unknown')} ({r.get('timestamp', '')[:10]})": i
                        for i, r in enumerate(sorted_reports)}

        # Determine default index based on selected_report in session state
        default_index = 0
        if st.session_state.get("selected_report"):
            # Find index of selected report
            sel = st.session_state.selected_report
            # Match by audit_id
            for i, r in enumerate(sorted_reports):
                if r.get("audit_id") == sel.get("audit_id"):
                    default_index = i
                    break

        selected = st.selectbox("Select Report", list(report_options.keys()), index=default_index)

        if selected:
            report_idx = report_options[selected]
            report = sorted_reports[report_idx]
        else:
            return
    else:
        report = report_data

    # Export buttons
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        json_data = generate_report_json(report)
        st.download_button(
            "📥 Export JSON",
            json_data,
            file_name=f"{report.get('audit_id', 'report')}.json",
            mime="application/json",
            key=f"export_json_{report.get('audit_id')}"
        )
    with col2:
        md_data = generate_report_markdown(report)
        st.download_button(
            "📥 Export Markdown",
            md_data,
            file_name=f"{report.get('audit_id', 'report')}.md",
            mime="text/markdown",
            key=f"export_md_{report.get('audit_id')}"
        )
    with col3:
        # reportlab might fail in some environments
        try:
            pdf_data = generate_report_pdf(report)
            st.download_button(
                "📥 Download PDF",
                pdf_data,
                file_name=f"{report.get('audit_id', 'report')}.pdf",
                mime="application/pdf",
                key=f"export_pdf_{report.get('audit_id')}"
            )
        except Exception as e:
            st.warning(f"PDF export unavailable: {str(e)}")

    st.markdown("---")

    # Report header
    st.subheader(f"Audit Report: {report.get('audit_id', 'Unknown')}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    score = report.get("compliance_score", 0)
    score_pct = score * 100 if score <= 1 else score

    with col1:
        score_class = get_score_class(score)
        st.markdown(f"**Compliance Score**")
        st.markdown(f'<p class="{score_class}">{score_pct:.1f}%</p>', unsafe_allow_html=True)

    with col2:
        st.metric("Model", report.get("model_id", "Unknown"))

    with col3:
        results = report.get("results", {})
        st.metric("Tests Passed", f"{results.get('passed', 0)}/{results.get('total_tests', 0)}")

    with col4:
        st.metric("Status", report.get("status", "Unknown").upper())

    st.markdown("---")

    # Results breakdown
    st.subheader("Results Breakdown")

    results = report.get("results", {})

    col1, col2 = st.columns([1, 1])

    with col1:
        # Pie chart of pass/fail
        if results.get("total_tests", 0) > 0:
            fig = go.Figure(data=[go.Pie(
                labels=['Passed', 'Failed'],
                values=[results.get('passed', 0), results.get('failed', 0)],
                marker_colors=['#28a745', '#dc3545'],
                hole=0.4
            )])
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Results by policy
        by_policy = results.get("by_policy", {})
        if by_policy:
            policy_names = list(by_policy.keys())
            policy_scores = [by_policy[p].get("score", 0) * 100 for p in policy_names]

            fig = go.Figure(data=[
                go.Bar(x=policy_names, y=policy_scores, marker_color='#1f77b4')
            ])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis_title="Score %",
                xaxis_title="Policy"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Detailed Test Results
    st.subheader("Test Details")

    # Extract detailed test information from framework results
    framework_results = results.get("framework_results", {})

    if framework_results:
        # Create tabs for each framework
        framework_tabs = st.tabs([f"📋 {fw.upper().replace('-', ' ')}" for fw in framework_results.keys()])

        for idx, (framework_name, framework_data) in enumerate(framework_results.items()):
            with framework_tabs[idx]:
                # Framework summary
                fw_tests = framework_data.get("tests", [])
                fw_passed = sum(1 for t in fw_tests if t.get("passed", False))
                fw_failed = len(fw_tests) - fw_passed

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tests", len(fw_tests))
                with col2:
                    st.metric("Passed", fw_passed, delta=f"{fw_passed}/{len(fw_tests)}")
                with col3:
                    st.metric("Failed", fw_failed, delta=f"-{fw_failed}" if fw_failed > 0 else "0")
                with col4:
                    pass_rate = (fw_passed / len(fw_tests) * 100) if len(fw_tests) > 0 else 0
                    st.metric("Pass Rate", f"{pass_rate:.1f}%")

                st.markdown("---")

                # Individual test results table
                if fw_tests:
                    # Group tests by category/attack type
                    tests_by_category = {}
                    for test in fw_tests:
                        category = test.get("attack_type", test.get("category", "General"))
                        if category not in tests_by_category:
                            tests_by_category[category] = []
                        tests_by_category[category].append(test)

                    # Display each category
                    for category, category_tests in tests_by_category.items():
                        st.markdown(f"#### {category.replace('_', ' ').title()}")

                        # Create a table for tests in this category
                        for i, test in enumerate(category_tests, 1):
                            test_name = test.get("name", f"{category.replace('_', ' ').title()} Test {i}")
                            test_status = "✅ PASSED" if test.get("passed", False) else "❌ FAILED"
                            test_score = test.get("score", 0.0)
                            test_vulnerable = test.get("vulnerable", False)

                            # Color code based on status
                            if test.get("passed", False):
                                status_color = "success"
                            else:
                                status_color = "error"

                            # Expandable test details
                            with st.expander(f"🔍 **Test #{i}: {test_name}** - {test_status}"):
                                test_col1, test_col2 = st.columns([1, 1])

                                with test_col1:
                                    st.markdown("**Test Information:**")
                                    st.markdown(f"- **Status:** {test_status}")
                                    st.markdown(f"- **Score:** {test_score:.2f}")
                                    st.markdown(f"- **Vulnerability:** {'Yes' if test_vulnerable else 'No'}")
                                    st.markdown(f"- **Category:** {category.replace('_', ' ').title()}")

                                    # Show severity if available
                                    if "severity" in test:
                                        severity = test["severity"].upper()
                                        st.markdown(f"- **Severity:** {severity}")

                                with test_col2:
                                    st.markdown("**Description:**")
                                    description = test.get("description", "No description available")
                                    st.markdown(f"{description}")

                                # Show prompt and response for PyRIT tests
                                if "prompt" in test:
                                    st.markdown("---")
                                    st.markdown("**Test Prompt:**")
                                    st.code(test["prompt"], language="text")

                                if "response" in test:
                                    st.markdown("**Model Response:**")
                                    response_text = test["response"]
                                    if len(response_text) > 500:
                                        st.code(response_text[:500] + "...", language="text")
                                    else:
                                        st.code(response_text, language="text")

                                # Show additional metadata
                                if "error" in test:
                                    st.error(f"Error: {test['error']}")

                        st.markdown("---")
                else:
                    st.info("No test details available for this framework.")
    else:
        st.info("No detailed test results available. Run an audit with PyRIT or other frameworks to see detailed test information.")

    st.markdown("---")

    # Findings
    st.subheader("Findings")

    findings = report.get("findings", [])

    if not findings:
        st.success("No critical findings detected.")
    else:
        # Group by severity
        severity_order = ["critical", "high", "medium", "low"]
        findings_by_severity = {}

        for finding in findings:
            sev = finding.get("severity", "medium").lower()
            if sev not in findings_by_severity:
                findings_by_severity[sev] = []
            findings_by_severity[sev].append(finding)

        for severity in severity_order:
            if severity in findings_by_severity:
                for finding in findings_by_severity[severity]:
                    css_class = f"finding-{severity}"
                    st.markdown(f'''
                    <div class="{css_class}">
                        <strong>{severity.upper()}</strong> - {finding.get('policy', 'Unknown Policy')}<br>
                        {finding.get('description', 'No description')}
                    </div>
                    ''', unsafe_allow_html=True)

    st.markdown("---")

    # Recommendations
    st.subheader("Recommendations")

    recommendations = report.get("recommendations", [])

    if not recommendations:
        st.info("No specific recommendations at this time.")
    else:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f'''
            <div class="recommendation-card">
                <strong>Recommendation {i}:</strong><br>
                {rec}
            </div>
            ''', unsafe_allow_html=True)

    st.markdown("---")

    # Raw data
    with st.expander("View Raw Report Data"):
        st.json(report)

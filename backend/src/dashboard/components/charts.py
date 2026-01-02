"""Enhanced Interactive Visualizations for AURA Dashboard"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import random

from .styles import COLORS





def hex_to_rgba(hex_color: str, opacity: float = 1.0) -> str:
    """Convert hex color to rgba string."""
    hex_color = hex_color.lstrip('#')
    return f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {opacity})"

def render_compliance_heatmap(audit_results: List[Dict[str, Any]]):
    """
    Render compliance heatmap showing Framework × Policy matrix.

    Args:
        audit_results: List of audit result dictionaries
    """
    st.markdown("### 🔥 Compliance Heat Map")
    st.markdown("Framework performance across all policies")

    # Prepare data for heatmap
    frameworks = ['AURA Native', 'Garak', 'PyRIT']
    # policies = ['Safety', 'Fairness', 'Transparency', 'Privacy', 'Accuracy']
    policies = ['Critical', 'High', 'Medium', 'Low']


    # Generate sample data (replace with real data from audit_results)
    data = []
    for framework in frameworks:
        row = []
        for policy in policies:
            # In real implementation, calculate from audit_results
            score = random.randint(60, 100)
            row.append(score)
        data.append(row)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=policies,
        y=frameworks,
        colorscale=[
            [0, COLORS['danger']],
            [0.5, COLORS['warning']],
            [0.7, COLORS['info']],
            [1, COLORS['success']]
        ],
        text=[[f"{val}%" for val in row] for row in data],
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>%{x}<br>Score: %{z}%<extra></extra>',
        colorbar=dict(
            title=dict(text="Compliance<br>Score (%)", side="right"),
            tickmode="linear",
            tick0=0,
            dtick=20
        )
    ))

    fig.update_layout(
        title=dict(
            text="Framework Performance by Policy Category",
            font=dict(size=16, color=COLORS['text_primary'])
        ),
        xaxis=dict(
            title=dict(
                text="Policy Category",
                font=dict(size=14, color=COLORS['text_primary'])
            ),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(
                text="Testing Framework",
                font=dict(size=14, color=COLORS['text_primary'])
            ),
            tickfont=dict(size=12)
        ),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig, use_container_width=True)


def render_trend_charts(audit_history: List[Dict[str, Any]]):
    """
    Render trend analysis charts.

    Args:
        audit_history: List of historical audit records
    """
    st.markdown("### 📈 Compliance Trends")

    col1, col2 = st.columns(2)

    with col1:
        # Pass Rate Over Time (Line Chart)
        st.markdown("#### Pass Rate Over Time")

        # Generate sample time series data
        dates = [datetime.now() - timedelta(days=30-i) for i in range(30)]
        pass_rates = [random.randint(70, 95) for _ in range(30)]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=pass_rates,
            mode='lines+markers',
            name='Pass Rate',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=6, color=COLORS['primary']),
            fill='tozeroy',
            fillcolor=f"rgba(102, 126, 234, 0.2)",
            hovertemplate='<b>%{x|%b %d}</b><br>Pass Rate: %{y}%<extra></extra>'
        ))

        # Add trend line
        avg_pass_rate = sum(pass_rates) / len(pass_rates)
        fig.add_hline(
            y=avg_pass_rate,
            line_dash="dash",
            line_color=COLORS['text_secondary'],
            annotation_text=f"Average: {avg_pass_rate:.1f}%",
            annotation_position="right"
        )

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)'
            ),
            yaxis=dict(
                title="Pass Rate (%)",
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
                range=[0, 100]
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,1)',
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Framework Comparison (Radar Chart)
        st.markdown("#### Framework Comparison")

        # categories = ['Safety', 'Fairness', 'Transparency', 'Privacy', 'Accuracy']
        categories = ['Critical', 'High', 'Medium', 'Low']

        fig = go.Figure()

        # AURA Native
        fig.add_trace(go.Scatterpolar(
            r=[92, 88, 85, 90, 87],
            theta=categories,
            fill='toself',
            name='AURA Native',
            line_color=COLORS['primary'],
            fillcolor=f"rgba(102, 126, 234, 0.3)"
        ))

        # Garak
        fig.add_trace(go.Scatterpolar(
            r=[85, 82, 88, 83, 86],
            theta=categories,
            fill='toself',
            name='Garak',
            line_color=COLORS['info'],
            fillcolor=f"rgba(59, 130, 246, 0.3)"
        ))

        # PyRIT
        fig.add_trace(go.Scatterpolar(
            r=[88, 85, 82, 87, 84],
            theta=categories,
            fill='toself',
            name='PyRIT',
            line_color=COLORS['success'],
            fillcolor=f"rgba(16, 185, 129, 0.3)"
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)'
                )
            ),
            showlegend=True,
            height=300,
            margin=dict(l=60, r=60, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig, use_container_width=True)


def render_risk_dashboard(findings: List[Dict[str, Any]]):
    """
    Render risk assessment dashboard.

    Args:
        findings: List of audit findings
    """
    st.markdown("### ⚠️ Risk Assessment Dashboard")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Overall Risk Score Gauge
        st.markdown("#### Overall Risk Score")

        # Calculate risk score (lower is better)
        risk_score = random.randint(15, 45)
        risk_level = "Low" if risk_score < 30 else "Medium" if risk_score < 60 else "High"
        risk_color = COLORS['success'] if risk_score < 30 else COLORS['warning'] if risk_score < 60 else COLORS['danger']

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{risk_level} Risk</b>", 'font': {'size': 16}},
            delta={'reference': 30, 'increasing': {'color': COLORS['danger']}, 'decreasing': {'color': COLORS['success']}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': risk_color, 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': hex_to_rgba(COLORS['success'], 0.2)},
                    {'range': [30, 60], 'color': hex_to_rgba(COLORS['warning'], 0.2)},
                    {'range': [60, 100], 'color': hex_to_rgba(COLORS['danger'], 0.2)}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text_primary'], 'family': "Inter"}
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk by Category (Bar Chart)
        st.markdown("#### Risk by Category")

        categories = ['Critical', 'High', 'Medium', 'Low']
        # categories = ['Safety', 'Fairness', 'Transparency', 'Privacy', 'Accuracy', 'Robustness']
        risk_values = [random.randint(10, 50) for _ in categories]

        # Color bars based on risk level
        colors = [
            COLORS['success'] if v < 30 else COLORS['warning'] if v < 60 else COLORS['danger']
            for v in risk_values
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=categories,
            y=risk_values,
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.2)', width=1)
            ),
            text=[f"{v}%" for v in risk_values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Risk Level: %{y}%<extra></extra>'
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=60),
            xaxis=dict(
                title="Category",
                showgrid=False,
                tickangle=-45
            ),
            yaxis=dict(
                title="Risk Level (%)",
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
                range=[0, 100]
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,1)',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # Critical Findings Timeline
    st.markdown("#### Critical Findings Timeline")

    # Generate sample timeline data
    timeline_data = []
    for i in range(10):
        timeline_data.append({
            'date': datetime.now() - timedelta(days=random.randint(0, 30)),
            'severity': random.choice(['Critical', 'High', 'Medium']),
            'category': random.choice(categories),
            'count': random.randint(1, 5)
        })

    df = pd.DataFrame(timeline_data).sort_values('date')

    # Create severity colors
    severity_colors = {
        'Critical': COLORS['danger'],
        'High': COLORS['warning'],
        'Medium': COLORS['info']
    }

    fig = go.Figure()

    for severity in ['Critical', 'High', 'Medium']:
        severity_data = df[df['severity'] == severity]
        fig.add_trace(go.Scatter(
            x=severity_data['date'],
            y=severity_data['count'],
            mode='markers+lines',
            name=severity,
            marker=dict(
                size=10,
                color=severity_colors[severity],
                line=dict(color='white', width=2)
            ),
            line=dict(color=severity_colors[severity], width=2),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%b %d}<br>Findings: %{y}<extra></extra>'
        ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)'
        ),
        yaxis=dict(
            title="Number of Findings",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,1)',
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def render_vulnerability_distribution(findings: List[Dict[str, Any]]):
    """
    Render vulnerability distribution as sunburst chart.

    Args:
        findings: List of audit findings
    """
    st.markdown("### 🎯 Vulnerability Distribution")

    # Prepare hierarchical data
    data = {
        'labels': [
            'Total',
            'Safety', 'Fairness', 'Privacy', 'Transparency',
            'Safety-Critical', 'Safety-High', 'Safety-Medium',
            'Fairness-Critical', 'Fairness-High', 'Fairness-Medium',
            'Privacy-Critical', 'Privacy-High', 'Privacy-Medium',
            'Transparency-Critical', 'Transparency-High', 'Transparency-Medium'
        ],
        'parents': [
            '',
            'Total', 'Total', 'Total', 'Total',
            'Safety', 'Safety', 'Safety',
            'Fairness', 'Fairness', 'Fairness',
            'Privacy', 'Privacy', 'Privacy',
            'Transparency', 'Transparency', 'Transparency'
        ],
        'values': [
            100,
            25, 30, 25, 20,
            8, 10, 7,
            12, 10, 8,
            10, 8, 7,
            8, 7, 5
        ]
    }

    # Define colors for severity
    colors_map = {
        'Critical': COLORS['danger'],
        'High': COLORS['warning'],
        'Medium': COLORS['info'],
        'Category': COLORS['primary']
    }

    marker_colors = [
        COLORS['text_primary'],  # Total
        COLORS['primary'], COLORS['primary'], COLORS['primary'], COLORS['primary'],  # Categories
        COLORS['danger'], COLORS['warning'], COLORS['info'],  # Safety
        COLORS['danger'], COLORS['warning'], COLORS['info'],  # Fairness
        COLORS['danger'], COLORS['warning'], COLORS['info'],  # Privacy
        COLORS['danger'], COLORS['warning'], COLORS['info']   # Transparency
    ]

    fig = go.Figure(go.Sunburst(
        labels=data['labels'],
        parents=data['parents'],
        values=data['values'],
        marker=dict(
            colors=marker_colors,
            line=dict(color='white', width=2)
        ),
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percentParent}<extra></extra>'
    ))

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)


def render_test_flow_sankey(audit_data: Dict[str, Any]):
    """
    Render test flow as Sankey diagram.

    Args:
        audit_data: Audit result data
    """
    st.markdown("### 🌊 Test Flow Analysis")
    st.markdown("Visualize how tests flow through frameworks to results")

    # Define nodes
    node_labels = [
        'Tests', 'AURA Native', 'Garak', 'PyRIT', 'Passed', 'Failed'
    ]

    # Define flows (source, target, value)
    flows = [
        # Tests to Frameworks
        (0, 1, 40),  # Tests → AURA Native
        (0, 2, 35),  # Tests → Garak
        (0, 3, 25),  # Tests → PyRIT
        # Frameworks to Results
        (1, 4, 35),  # AURA Native → Passed
        (1, 5, 5),   # AURA Native → Failed
        (2, 4, 28),  # Garak → Passed
        (2, 5, 7),   # Garak → Failed
        (3, 4, 20),  # PyRIT → Passed
        (3, 5, 5),   # PyRIT → Failed
    ]

    # Define node colors
    node_colors = [
        COLORS['text_primary'],  # Tests
        COLORS['primary'],        # AURA Native
        COLORS['info'],          # Garak
        COLORS['success'],       # PyRIT
        COLORS['success'],       # Passed
        COLORS['danger']         # Failed
    ]

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=2),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=[f[0] for f in flows],
            target=[f[1] for f in flows],
            value=[f[2] for f in flows],
            color=[f"rgba(102, 126, 234, 0.3)" for _ in flows]
        )
    )])

    fig.update_layout(
        title=dict(
            text="Test Execution Flow",
            font=dict(size=16, color=COLORS['text_primary'])
        ),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color=COLORS['text_primary'])
    )

    st.plotly_chart(fig, use_container_width=True)


def render_all_visualizations(
    audit_results: Optional[List[Dict[str, Any]]] = None,
    audit_history: Optional[List[Dict[str, Any]]] = None,
    findings: Optional[List[Dict[str, Any]]] = None,
    audit_data: Optional[Dict[str, Any]] = None
):
    """
    Render all visualization components.

    Args:
        audit_results: List of audit results for heatmap
        audit_history: Historical audit data for trends
        findings: Audit findings for risk dashboard
        audit_data: Current audit data for Sankey diagram
    """
    # Use session state data if not provided
    if audit_results is None:
        audit_results = st.session_state.get('audit_history', [])

    if audit_history is None:
        audit_history = st.session_state.get('audit_history', [])

    if findings is None:
        findings = st.session_state.get('findings', [])

    if audit_data is None and st.session_state.get('selected_report'):
        audit_data = st.session_state.selected_report

    # Render all charts
    render_compliance_heatmap(audit_results)
    st.markdown("---")

    render_trend_charts(audit_history)
    st.markdown("---")

    render_risk_dashboard(findings)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        render_vulnerability_distribution(findings)

    with col2:
        render_test_flow_sankey(audit_data or {})

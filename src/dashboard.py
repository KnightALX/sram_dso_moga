"""
SRAM DSO-MOGA: Professional EDA-Style Dashboard.

Professional EDA tool aesthetic - inspired by Cadence Virtuoso, Synopsys Design Compiler.
Layout: Left sidebar for controls, right panel for visualizations.
Theme: Light/dark mode toggle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import (
    Dash, dcc, html, dash_table,
    State, Input, Output,
)


# =============================================================================
# Dashboard Class
# =============================================================================

class Dashboard:
    """Professional EDA-style dashboard for MOGA optimization results."""

    def __init__(self, results: dict, config: dict, output_dir: Path):
        self.results = results
        self.config = config
        self.output_dir = output_dir
        self.current_theme = 'dark'

        # Build data structures
        self.pareto_df = self._build_pareto_df()
        self.convergence_data = self._build_convergence_data()

        self.app = self._create_app()

    def _build_pareto_df(self) -> pd.DataFrame:
        """Build DataFrame of Pareto solutions."""
        solutions = self.results.get('pareto_solutions', [])
        objectives = self.results.get('pareto_objectives', [])

        df = pd.DataFrame(solutions)
        obj_df = pd.DataFrame(objectives, columns=['Area', 'Power', 'Delay'])
        df = pd.concat([df, obj_df], axis=1)
        df['ID'] = range(1, len(df) + 1)
        return df

    def _build_convergence_data(self) -> pd.DataFrame:
        """Build convergence history DataFrame."""
        history = self.results.get('history', [])
        return pd.DataFrame(history)

    def _create_app(self) -> Dash:
        """Create and configure Dash application."""
        app = Dash(__name__)
        app.title = "SRAM DSO-MOGA Dashboard"
        app.config.suppress_callback_exceptions = True

        self._register_callbacks(app)
        app.layout = self._create_layout()

        return app

    def _create_layout(self) -> html.Div:
        """Create main dashboard layout with EDA-style panels."""
        top_name = self.config.get('top_name', 'SRAM DSO-MOGA')
        n_pareto = len(self.pareto_df)
        algo_name = self.config.get('algorithm', {}).get('name', 'NSGA-II')

        objectives = self.pareto_df[['Area', 'Power', 'Delay']]
        stats = {
            'area_min': objectives['Area'].min() if len(objectives) else 0,
            'area_mean': objectives['Area'].mean() if len(objectives) else 0,
            'power_min': objectives['Power'].min() if len(objectives) else 0,
            'power_mean': objectives['Power'].mean() if len(objectives) else 0,
            'delay_min': objectives['Delay'].min() if len(objectives) else 0,
            'delay_mean': objectives['Delay'].mean() if len(objectives) else 0,
        }

        return html.Div([
            # Header bar
            html.Div([
                html.Div([
                    html.Span('SRAM DSO-MOGA', className='header-logo'),
                    html.Span('Multi-Objective Genetic Algorithm', className='header-subtitle'),
                ], className='header-left'),

                html.Div([
                    html.Span(f'Design: {top_name}', className='header-badge'),
                    html.Span(f'{n_pareto} Pareto', className='header-badge accent'),
                    html.Span(f'Algo: {algo_name}', className='header-badge'),

                    # Theme toggle
                    html.Div([
                        html.Button('☀', id='btn-light', className='theme-btn', title='Light Mode'),
                        html.Button('☾', id='btn-dark', className='theme-btn', title='Dark Mode'),
                    ], className='theme-toggle'),

                    html.Div([
                        html.Span(className='status-dot'),
                        html.Span('Ready', className='header-badge'),
                    ]),
                ], className='header-right'),
            ], className='header-bar'),

            # Main content area
            html.Div([
                # Left sidebar - controls
                html.Div([
                    # Algorithm Settings
                    html.Div([
                        html.Span('Algorithm Settings', className='sidebar-title'),
                    ], className='sidebar-section'),

                    html.Div([
                        html.Div([
                            html.Label('Population Size', className='form-label'),
                            dcc.Input(id='input-pop-size', type='number',
                                      value=self.config.get('algorithm', {}).get('pop_size', 80),
                                      className='form-input'),
                        ], className='form-group'),
                        html.Div([
                            html.Label('Generations', className='form-label'),
                            dcc.Input(id='input-n-gen', type='number',
                                      value=self.config.get('algorithm', {}).get('n_gen', 60),
                                      className='form-input'),
                        ], className='form-group'),
                        html.Div([
                            html.Label('Algorithm', className='form-label'),
                            dcc.Dropdown(id='input-algo', options=[
                                {'label': 'NSGA-II', 'value': 'NSGA-II'},
                                {'label': 'NSGA-III', 'value': 'NSGA-III'},
                            ], value=algo_name, className='form-dropdown'),
                        ], className='form-group'),
                    ], className='sidebar-section'),

                    # Objective Filters
                    html.Div([
                        html.Span('Objective Filters', className='sidebar-title'),
                    ], className='sidebar-section'),

                    html.Div([
                        html.Div([
                            html.Label('Area Range (um²)', className='form-label'),
                            dcc.RangeSlider(id='slider-area', min=0, max=100, step=1, value=[0, 100],
                                          className='form-slider'),
                            html.Div(id='slider-area-labels', className='slider-labels'),
                        ], className='form-group'),
                        html.Div([
                            html.Label('Power Range (uW)', className='form-label'),
                            dcc.RangeSlider(id='slider-power', min=0, max=1000, step=10, value=[0, 1000],
                                          className='form-slider'),
                            html.Div(id='slider-power-labels', className='slider-labels'),
                        ], className='form-group'),
                        html.Div([
                            html.Label('Delay Range (ps)', className='form-label'),
                            dcc.RangeSlider(id='slider-delay', min=0, max=1000, step=10, value=[0, 1000],
                                          className='form-slider'),
                            html.Div(id='slider-delay-labels', className='slider-labels'),
                        ], className='form-group'),
                    ], className='sidebar-section'),

                    # Visualization
                    html.Div([
                        html.Span('Visualization', className='sidebar-title'),
                    ], className='sidebar-section'),

                    html.Div([
                        html.Div([
                            html.Label('Chart Type', className='form-label'),
                            dcc.Dropdown(id='input-chart-type', options=[
                                {'label': '2D Scatter', 'value': '2d'},
                                {'label': '3D Scatter', 'value': '3d'},
                            ], value='2d', className='form-dropdown'),
                        ], className='form-group'),
                    ], className='sidebar-section'),
                ], className='sidebar'),

                # Right panel
                html.Div([
                    # Metrics row
                    html.Div([
                        html.Div([
                            html.Div('Best Area', className='metric-label'),
                            html.Div([
                                html.Span(f"{stats['area_min']:.3f}", className='metric-value best'),
                                html.Span('um²', className='metric-unit'),
                            ]),
                            html.Div(f'Mean: {stats["area_mean"]:.3f}', className='metric-subtext'),
                        ], className='metric-card'),
                        html.Div([
                            html.Div('Best Power', className='metric-label'),
                            html.Div([
                                html.Span(f"{stats['power_min']:.2f}", className='metric-value best'),
                                html.Span('uW', className='metric-unit'),
                            ]),
                            html.Div(f'Mean: {stats["power_mean"]:.2f}', className='metric-subtext'),
                        ], className='metric-card'),
                        html.Div([
                            html.Div('Best Delay', className='metric-label'),
                            html.Div([
                                html.Span(f"{stats['delay_min']:.2f}", className='metric-value best'),
                                html.Span('ps', className='metric-unit'),
                            ]),
                            html.Div(f'Mean: {stats["delay_mean"]:.2f}', className='metric-subtext'),
                        ], className='metric-card'),
                    ], className='metrics-grid'),

                    # Tabs
                    dcc.Tabs(id='tabs', value='tab-overview', children=[
                        dcc.Tab(label='Overview', value='tab-overview', children=[
                            html.Div(self._create_overview_chart(), className='chart-area'),
                        ]),
                        dcc.Tab(label='3D View', value='tab-3d', children=[
                            html.Div(self._create_3d_chart(), className='chart-area'),
                        ]),
                        dcc.Tab(label='Convergence', value='tab-conv', children=[
                            html.Div(self._create_convergence_chart(), className='chart-area'),
                        ]),
                        dcc.Tab(label='Solutions', value='tab-solutions', children=[
                            html.Div(self._create_solutions_table(), className='table-area'),
                        ]),
                    ], className='main-tabs'),
                ], className='right-panel'),
            ], className='main-content'),
        ], className='app-container')

    def _create_overview_chart(self) -> dcc.Graph:
        """Create 2D scatter overview."""
        obj = self.pareto_df[['Area', 'Power', 'Delay']].values

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Area vs Power', 'Area vs Delay', 'Power vs Delay'],
            horizontal_spacing=0.1,
        )

        colors = obj[:, 2]

        fig.add_trace(go.Scatter(
            x=obj[:, 0], y=obj[:, 1], mode='markers',
            marker=dict(size=8, color=colors, colorscale='Viridis'),
            hovertemplate='Area: %{x:.3f}<br>Power: %{y:.2f}<extra></extra>',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=obj[:, 0], y=obj[:, 2], mode='markers',
            marker=dict(size=8, color=colors, colorscale='Viridis'),
            hovertemplate='Area: %{x:.3f}<br>Delay: %{y:.2f}<extra></extra>',
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=obj[:, 1], y=obj[:, 2], mode='markers',
            marker=dict(size=8, color=obj[:, 0], colorscale='Plasma'),
            hovertemplate='Power: %{x:.2f}<br>Delay: %{y:.2f}<extra></extra>',
        ), row=1, col=3)

        fig.update_layout(
            height=380, showlegend=False, title_font_size=13, title_x=0.5,
            paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
            font=dict(color='#8b949e'),
            margin=dict(l=40, r=40, t=50, b=40),
        )

        for i in range(1, 4):
            fig.update_xaxes(gridcolor='#30363d', row=1, col=i)
            fig.update_yaxes(gridcolor='#30363d', row=1, col=i)

        return dcc.Graph(figure=fig, className='main-graph')

    def _create_3d_chart(self) -> dcc.Graph:
        """Create 3D scatter chart."""
        obj = self.pareto_df[['Area', 'Power', 'Delay']].values

        fig = go.Figure(data=[go.Scatter3d(
            x=obj[:, 0], y=obj[:, 1], z=obj[:, 2],
            mode='markers',
            marker=dict(size=6, color=obj[:, 2], colorscale='Viridis', opacity=0.85),
            text=[f'Sol {i+1}<br>Area: {obj[i,0]:.3f}<br>Power: {obj[i,1]:.2f}<br>Delay: {obj[i,2]:.2f}'
                  for i in range(len(obj))],
            hoverinfo='text',
        )])

        fig.update_layout(
            height=500,
            scene=dict(
                xaxis_title='Area (um²)', yaxis_title='Power (uW)', zaxis_title='Delay (ps)',
                xaxis=dict(backgroundcolor='#0d1117', gridcolor='#30363d'),
                yaxis=dict(backgroundcolor='#0d1117', gridcolor='#30363d'),
                zaxis=dict(backgroundcolor='#0d1117', gridcolor='#30363d'),
            ),
            paper_bgcolor='#161b22',
            margin=dict(l=0, r=0, t=50, b=0),
            title_font_size=13, title_x=0.5,
        )

        return dcc.Graph(figure=fig, className='main-graph')

    def _create_convergence_chart(self) -> dcc.Graph:
        """Create convergence chart."""
        conv_df = self.convergence_data

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Pareto Front Size', 'Average Fitness'],
            vertical_spacing=0.15,
        )

        if len(conv_df) > 0:
            fig.add_trace(go.Scatter(
                x=conv_df['gen'],
                y=conv_df.get('front0_size', [0]*len(conv_df)),
                mode='lines+markers', name='Front 0',
                line=dict(color='#3fb950', width=2), marker=dict(size=5),
            ), row=1, col=1)

            if 'avg_fitness' in conv_df.columns:
                avg_fit = conv_df['avg_fitness'].tolist()
                if avg_fit and isinstance(avg_fit[0], list):
                    for i, (label, color) in enumerate(zip(['Area', 'Power', 'Delay'], ['#58a6ff', '#3fb950', '#f0883e'])):
                        fig.add_trace(go.Scatter(
                            x=conv_df['gen'],
                            y=[f[i] if len(f) > i else 0 for f in avg_fit],
                            mode='lines', name=f'Avg {label}',
                            line=dict(color=color, width=1.5, dash='dash'),
                        ), row=2, col=1)

        fig.update_layout(
            height=450, showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
            font=dict(color='#8b949e'),
            margin=dict(l=50, r=50, t=50, b=40),
        )

        return dcc.Graph(figure=fig, className='main-graph')

    def _create_solutions_table(self) -> dash_table.DataTable:
        """Create solutions data table."""
        display_df = self.pareto_df.copy()
        cols = ['ID'] + [c for c in display_df.columns if c != 'ID']
        display_df = display_df[cols]

        return dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in display_df.columns],
            page_size=20,
            sort_action='native',
            filter_action='native',
            style_cell=dict(
                textAlign='left', fontFamily="'JetBrains Mono', monospace",
                fontSize='12px', padding='8px 12px',
                backgroundColor='#161b22', color='#e6edf3',
            ),
            style_header=dict(
                backgroundColor='#21262d', fontWeight='600',
                textTransform='uppercase', fontSize='10px', letterSpacing='0.05em',
            ),
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#0d1117'},
            ],
            style_table=dict(overflowX='auto'),
        )

    def _register_callbacks(self, app: Dash) -> None:
        """Register Dash callbacks."""

        @app.callback(
            [Output('slider-area-labels', 'children'),
             Output('slider-power-labels', 'children'),
             Output('slider-delay-labels', 'children')],
            [Input('slider-area', 'value'),
             Input('slider-power', 'value'),
             Input('slider-delay', 'value')]
        )
        def update_slider_labels(area_val, power_val, delay_val):
            return [
                f'{area_val[0]} - {area_val[1]}',
                f'{power_val[0]} - {power_val[1]}',
                f'{delay_val[0]} - {delay_val[1]}',
            ]

    def run(self, port: int = 8050, debug: bool = False, host: str = '0.0.0.0') -> None:
        """Run the dashboard server."""
        self.app.run(debug=debug, port=port, host=host)

    def save_html(self, path: Optional[Path] = None) -> Path:
        """Export dashboard as standalone HTML."""
        path = path or self.output_dir / 'dashboard.html'
        obj = self.pareto_df[['Area', 'Power', 'Delay']].values

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            subplot_titles=['3D Pareto', 'Area vs Power', 'Area vs Delay', 'Power vs Delay'],
            vertical_spacing=0.12, horizontal_spacing=0.1,
        )

        fig.add_trace(go.Scatter3d(
            x=obj[:, 0], y=obj[:, 1], z=obj[:, 2],
            mode='markers',
            marker=dict(size=5, color=obj[:, 2], colorscale='Viridis'),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=obj[:, 0], y=obj[:, 1], mode='markers'), row=1, col=2)
        fig.add_trace(go.Scatter(x=obj[:, 0], y=obj[:, 2], mode='markers'), row=2, col=1)
        fig.add_trace(go.Scatter(x=obj[:, 1], y=obj[:, 2], mode='markers'), row=2, col=2)

        fig.update_layout(
            title='SRAM DSO-MOGA Results',
            height=700, showlegend=False,
        )

        fig.write_html(str(path), include_plotlyjs='cdn', full_html=True)
        return path


def create_dashboard(results: dict, config: dict, output_dir: Path) -> Dashboard:
    """Factory function to create dashboard."""
    return Dashboard(results, config, output_dir)

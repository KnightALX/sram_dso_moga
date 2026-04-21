"""
SRAM DSO-MOGA: Professional Dash Dashboard.

Provides interactive visualization and analysis of optimization results
using Plotly Dash with EDA-style dark theme.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import (
    ALL, Dash, State, dcc, html,
    Input, Output, callback_context as cb_context
)


# EDA-style color theme
THEME = {
    'bg_primary': '#0a0a14',
    'bg_secondary': '#16213e',
    'bg_tertiary': '#1a1a2e',
    'bg_hover': '#1f4068',
    'border': '#2d4a6f',
    'text_primary': '#e2e8f0',
    'text_secondary': '#94a3b8',
    'text_muted': '#64748b',
    'accent': '#3b82f6',
    'status_pass': '#22c55e',
    'status_fail': '#ef4444',
    'status_warning': '#f59e0b',
}


class Dashboard:
    """Interactive dashboard for MOGA results."""

    def __init__(self, results: dict, config: dict, output_dir: Path):
        self.results = results
        self.config = config
        self.output_dir = output_dir

        self.pareto_df = self._build_pareto_df()
        self.convergence_data = self._build_convergence_data()
        self.param_df = self._build_param_df()

        self.app = self._create_app()

    def _build_pareto_df(self) -> pd.DataFrame:
        """Build DataFrame of Pareto solutions."""
        solutions = self.results.get('pareto_solutions', [])
        objectives = self.results.get('pareto_objectives', [])

        df = pd.DataFrame(solutions)
        obj_df = pd.DataFrame(objectives, columns=['Area', 'Power', 'Delay'])
        df = pd.concat([df, obj_df], axis=1)

        # Add rank column for visualization
        df['Solution ID'] = range(1, len(df) + 1)

        return df

    def _build_convergence_data(self) -> pd.DataFrame:
        """Build convergence history DataFrame."""
        history = self.results.get('history', [])
        return pd.DataFrame(history)

    def _build_param_df(self) -> pd.DataFrame:
        """Build parameter distribution DataFrame."""
        return self.pareto_df.drop(columns=['Area', 'Power', 'Delay', 'Solution ID'],
                                    errors='ignore')

    def _create_app(self) -> Dash:
        """Create and configure Dash application."""
        app = Dash(__name__)
        app.title = "SRAM DSO-MOGA Dashboard"

        app.layout = self._create_layout()

        self._register_callbacks(app)

        return app

    def _create_layout(self) -> html.Div:
        """Create main dashboard layout."""
        return html.Div([
            # Header
            self._create_header(),

            # Main content
            html.Div([
                # Left sidebar - controls
                self._create_sidebar(),

                # Main content area
                html.Div([
                    dcc.Tabs(id='tabs', value='tab-overview', children=[
                        dcc.Tab(label='Overview', value='tab-overview',
                                children=self._create_overview_tab()),
                        dcc.Tab(label='3D Pareto', value='tab-3d',
                                children=self._create_3d_tab()),
                        dcc.Tab(label='2D Projections', value='tab-2d',
                                children=self._create_2d_tab()),
                        dcc.Tab(label='Convergence', value='tab-conv',
                                children=self._create_convergence_tab()),
                        dcc.Tab(label='Parameters', value='tab-params',
                                children=self._create_params_tab()),
                        dcc.Tab(label='Solutions', value='tab-solutions',
                                children=self._create_solutions_tab()),
                    ]),
                ], className='main-content'),

                # Right panel - summary
                self._create_summary_panel(),
            ], className='main-layout'),

            # Hidden data stores
            dcc.Store(id='store-results', data=self._serialize_results()),
        ], className='dashboard-container')

    def _create_header(self) -> html.Div:
        """Create header bar."""
        top_name = self.config.get('top_name', 'SRAM DSO-MOGA')
        n_pareto = len(self.pareto_df)

        return html.Div([
            html.Div([
                html.Span('SRAM DSO-MOGA', className='logo'),
                html.Span('Multi-Objective Genetic Algorithm Optimizer', className='subtitle'),
            ], className='header-left'),
            html.Div([
                html.Span(f'Design: {top_name}', className='status-badge'),
                html.Span(f'Pareto Solutions: {n_pareto}', className='status-badge status-accent'),
                html.Span(className='status-dot'),
                html.Span('Ready', className='status-text'),
            ], className='header-right'),
        ], className='header-bar')

    def _create_sidebar(self) -> html.Div:
        """Create left sidebar with controls."""
        return html.Div([
            html.Div([
                html.Span('Configuration', className='section-title'),
                html.Div([
                    html.Label('Population Size', className='form-label'),
                    dcc.Input(id='input-pop-size', type='number', value=80, className='input-field'),
                ], className='form-group'),
                html.Div([
                    html.Label('Generations', className='form-label'),
                    dcc.Input(id='input-n-gen', type='number', value=60, className='input-field'),
                ], className='form-group'),
                html.Div([
                    html.Label('Algorithm', className='form-label'),
                    dcc.Dropdown(id='input-algo', options=[
                        {'label': 'NSGA-II', 'value': 'NSGA-II'},
                        {'label': 'NSGA-III', 'value': 'NSGA-III'},
                    ], value='NSGA-II', className='dropdown'),
                ], className='form-group'),
            ], className='sidebar-section'),

            html.Div([
                html.Span('Filters', className='section-title'),
                html.Div([
                    html.Label('Area Range', className='form-label'),
                    dcc.RangeSlider(id='slider-area', min=0, max=100, step=1,
                                    value=[0, 100], className='range-slider'),
                ], className='form-group'),
                html.Div([
                    html.Label('Power Range', className='form-label'),
                    dcc.RangeSlider(id='slider-power', min=0, max=1000, step=10,
                                    value=[0, 1000], className='range-slider'),
                ], className='form-group'),
                html.Div([
                    html.Label('Delay Range', className='form-label'),
                    dcc.RangeSlider(id='slider-delay', min=0, max=1000, step=10,
                                    value=[0, 1000], className='range-slider'),
                ], className='form-group'),
            ], className='sidebar-section'),

            html.Div([
                html.Button('Apply Filters', id='btn-apply', className='btn btn-primary btn-block'),
                html.Button('Reset', id='btn-reset', className='btn btn-secondary btn-block'),
            ], className='sidebar-section'),
        ], className='sidebar')

    def _create_summary_panel(self) -> html.Div:
        """Create right summary panel."""
        objectives = self.pareto_df[['Area', 'Power', 'Delay']]

        return html.Div([
            html.Div('Pareto Summary', className='panel-header'),
            html.Div([
                html.Div([
                    html.Span('Best Area', className='metric-label'),
                    html.Span(f"{objectives['Area'].min():.3f}", className='metric-value'),
                ], className='metric-row'),
                html.Div([
                    html.Span('Best Power', className='metric-label'),
                    html.Span(f"{objectives['Power'].min():.2f}", className='metric-value'),
                ], className='metric-row'),
                html.Div([
                    html.Span('Best Delay', className='metric-label'),
                    html.Span(f"{objectives['Delay'].min():.2f}", className='metric-value'),
                ], className='metric-row'),
                html.Hr(className='divider'),
                html.Div([
                    html.Span('Total Solutions', className='metric-label'),
                    html.Span(f"{len(self.pareto_df)}", className='metric-value'),
                ], className='metric-row'),
            ], className='panel-content'),
        ], className='right-panel')

    def _create_overview_tab(self) -> html.Div:
        """Create overview tab with summary charts."""
        fig = go.Figure()

        # 2D scatter matrix of top objectives
        obj_cols = ['Area', 'Power', 'Delay']
        colors = self.pareto_df['Area'].values if 'Area' in self.pareto_df else [1] * len(self.pareto_df)

        for i, (x, y) in enumerate([('Area', 'Power'), ('Area', 'Delay'), ('Power', 'Delay')]):
            fig.add_trace(go.Scatter(
                x=self.pareto_df[x], y=self.pareto_df[y],
                mode='markers', name=f'{x} vs {y}',
                marker=dict(size=8, color=colors, colorscale='Viridis'),
                xaxis=f'x{i+1}' if i > 0 else 'x',
                yaxis=f'y{i+1}' if i > 0 else 'y',
                showlegend=False,
            ))

        fig.update_layout(
            grid=dict(rows=1, columns=3, pattern='coupled'),
            height=400,
            title='Pareto Front Overview',
            paper_bgcolor=THEME['bg_secondary'],
            font=dict(color=THEME['text_primary']),
        )

        return html.Div([
            html.Div([
                html.H3('Optimization Results Summary'),
                html.P(f'Found {len(self.pareto_df)} Pareto-optimal solutions'),
            ], className='tab-header'),
            dcc.Graph(figure=fig, className='chart-container'),
            self._create_quick_stats(),
        ])

    def _create_3d_tab(self) -> html.Div:
        """Create 3D Pareto visualization tab."""
        obj = self.pareto_df[['Area', 'Power', 'Delay']].values

        fig = go.Figure(data=[go.Scatter3d(
            x=obj[:, 0], y=obj[:, 1], z=obj[:, 2],
            mode='markers',
            marker=dict(size=6, color=obj[:, 2], colorscale='Viridis'),
            text=[f'Sol {i}' for i in range(len(obj))],
            hoverinfo='text',
        )])

        fig.update_layout(
            title='3D Pareto Front',
            scene=dict(
                xaxis_title='Area (um²)',
                yaxis_title='Power (uW)',
                zaxis_title='Delay (ps)',
                bgcolor=THEME['bg_primary'],
            ),
            paper_bgcolor=THEME['bg_secondary'],
            font=dict(color=THEME['text_primary']),
            height=600,
        )

        return html.Div([
            dcc.Graph(figure=fig, className='chart-container-full'),
        ])

    def _create_2d_tab(self) -> html.Div:
        """Create 2D projection charts."""
        obj = self.pareto_df[['Area', 'Power', 'Delay']].values

        fig = make_subplots(rows=1, cols=3,
                           subplot_titles=['Area vs Power', 'Area vs Delay', 'Power vs Delay'])

        fig.add_trace(go.Scatter(x=obj[:, 0], y=obj[:, 1], mode='markers',
                                marker=dict(color=obj[:, 2], colorscale='Viridis')),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=obj[:, 0], y=obj[:, 2], mode='markers',
                                marker=dict(color=obj[:, 1], colorscale='Plasma')),
                     row=1, col=2)
        fig.add_trace(go.Scatter(x=obj[:, 1], y=obj[:, 2], mode='markers',
                                marker=dict(color=obj[:, 0], colorscale='Cividis')),
                     row=1, col=3)

        fig.update_layout(
            title='2D Objective Projections',
            height=450,
            paper_bgcolor=THEME['bg_secondary'],
            font=dict(color=THEME['text_primary']),
            showlegend=False,
        )

        return html.Div([dcc.Graph(figure=fig)])

    def _create_convergence_tab(self) -> html.Div:
        """Create convergence analysis tab."""
        conv_df = self.convergence_data

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=conv_df['gen'], y=conv_df['front0_size'],
            mode='lines+markers',
            name='Front 0 Size',
            line=dict(color='#3b82f6', width=2),
        ))

        fig.update_layout(
            title='NSGA-II Convergence',
            xaxis_title='Generation',
            yaxis_title='Pareto Front Size',
            height=400,
            paper_bgcolor=THEME['bg_secondary'],
            font=dict(color=THEME['text_primary']),
        )

        return html.Div([
            dcc.Graph(figure=fig),
            html.Div([
                html.H4('Convergence Analysis'),
                html.P('The front size should stabilize as the algorithm converges.'),
            ], className='analysis-text'),
        ])

    def _create_params_tab(self) -> html.Div:
        """Create parameter distribution tab."""
        param_cols = [c for c in self.param_df.columns if c]

        fig = make_subplots(rows=len(param_cols), cols=1,
                           subplot_titles=param_cols)

        for i, col in enumerate(param_cols):
            fig.add_trace(go.Histogram(x=self.param_df[col], name=col,
                                      nbinsx=10, marker_color='#3b82f6'),
                         row=i+1, col=1)

        fig.update_layout(
            title='Parameter Distributions',
            height=max(300, 100 * len(param_cols)),
            showlegend=False,
            paper_bgcolor=THEME['bg_secondary'],
            font=dict(color=THEME['text_primary']),
        )

        return html.Div([dcc.Graph(figure=fig)])

    def _create_solutions_tab(self) -> html.Div:
        """Create solutions table tab."""
        from dash import dash_table

        display_df = self.pareto_df.copy()
        display_df['ID'] = range(1, len(display_df) + 1)
        display_df = display_df[['ID'] + [c for c in display_df.columns if c != 'ID']]

        return html.Div([
            dash_table.DataTable(
                data=display_df.to_dict('records'),
                columns=[{'name': c, 'id': c} for c in display_df.columns],
                page_size=15,
                sort_action='native',
                filter_action='native',
                style_cell={
                    'textAlign': 'left',
                    'fontFamily': 'JetBrains Mono, monospace',
                    'fontSize': '12px',
                    'backgroundColor': THEME['bg_primary'],
                    'color': THEME['text_primary'],
                },
                style_header={
                    'backgroundColor': THEME['bg_tertiary'],
                    'fontWeight': '600',
                    'textTransform': 'uppercase',
                    'fontSize': '11px',
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': THEME['bg_secondary']},
                ],
            ),
        ])

    def _create_quick_stats(self) -> html.Div:
        """Create quick statistics section."""
        obj = self.pareto_df[['Area', 'Power', 'Delay']]

        return html.Div([
            html.Div('Quick Statistics', className='section-title'),
            html.Div([
                html.Div([
                    html.Span('Mean Area:', className='stat-label'),
                    html.Span(f"{obj['Area'].mean():.3f}", className='stat-value'),
                ], className='stat-row'),
                html.Div([
                    html.Span('Mean Power:', className='stat-label'),
                    html.Span(f"{obj['Power'].mean():.2f}", className='stat-value'),
                ], className='stat-row'),
                html.Div([
                    html.Span('Mean Delay:', className='stat-label'),
                    html.Span(f"{obj['Delay'].mean():.2f}", className='stat-value'),
                ], className='stat-row'),
            ], className='stats-grid'),
        ])

    def _serialize_results(self) -> dict:
        """Serialize results for Dash store."""
        return {
            'pareto_solutions': self.results.get('pareto_solutions', []),
            'pareto_objectives': self.results.get('pareto_objectives', []),
            'history': self.results.get('history', []),
        }

    def _register_callbacks(self, app: Dash) -> None:
        """Register Dash callbacks."""

        @app.callback(
            Output('store-results', 'data'),
            [Input('btn-apply', 'n_clicks'),
             Input('slider-area', 'value'),
             Input('slider-power', 'value'),
             Input('slider-delay', 'value')],
            [State('store-results', 'data')]
        )
        def update_filters(n_clicks, area_range, power_range, delay_range, data):
            ctx = cb_context
            if not ctx.triggered:
                return data

            # Re-filter results based on ranges
            # (Implementation would filter the DataFrame)
            return data

        @app.callback(
            Output('tabs', 'value'),
            [Input('btn-reset', 'n_clicks')]
        )
        def reset_filters(n_clicks):
            return 'tab-overview'

    def run(self, port: int = 8050, debug: bool = True) -> None:
        """Run the dashboard server."""
        self.app.run(debug=debug, port=port)

    def save_html(self, path: Optional[Path] = None) -> Path:
        """Export dashboard as standalone HTML."""
        path = path or self.output_dir / 'dashboard.html'

        # Generate static HTML version
        fig = self._create_static_figure()
        fig.write_html(str(path), include_plotlyjs='cdn')

        return path

    def _create_static_figure(self) -> go.Figure:
        """Create combined static figure for HTML export."""
        obj = self.pareto_df[['Area', 'Power', 'Delay']].values

        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'scatter3d'}, None, None],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
            subplot_titles=['3D Pareto', 'Area vs Power', 'Area vs Delay', '', 'Power vs Delay', '']
        )

        # 3D scatter
        fig.add_trace(go.Scatter3d(
            x=obj[:, 0], y=obj[:, 1], z=obj[:, 2],
            mode='markers', marker=dict(size=5, color=obj[:, 2], colorscale='Viridis')
        ), row=1, col=1)

        # 2D projections
        fig.add_trace(go.Scatter(x=obj[:, 0], y=obj[:, 1], mode='markers'), row=2, col=1)
        fig.add_trace(go.Scatter(x=obj[:, 0], y=obj[:, 2], mode='markers'), row=2, col=2)
        fig.add_trace(go.Scatter(x=obj[:, 1], y=obj[:, 2], mode='markers'), row=2, col=3)

        fig.update_layout(
            title='SRAM DSO-MOGA Results',
            height=800,
            showlegend=False,
        )

        return fig


def create_dashboard(results: dict, config: dict, output_dir: Path) -> Dashboard:
    """Factory function to create dashboard."""
    return Dashboard(results, config, output_dir)


# Need to import make_subplots
from plotly.subplots import make_subplots
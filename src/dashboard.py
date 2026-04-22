"""
SRAM DSO-MOGA: Professional EDA-Style Dashboard.

Provides interactive visualization and analysis of optimization results
using Plotly Dash with light/dark theme support.

Visual Language: Technical Precision
- Inspired by professional EDA tools (Cadence Virtuoso, Synopsys)
- Monospace fonts for data, clean sans-serif for UI
- Muted backgrounds with sharp accent colors
- Dense information displays with clear hierarchy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import (
    Dash, dcc, html, dash_table,
    State, Input, Output, callback_context as cb_context,
    DiskcacheManager, CeleryManager
)


# =============================================================================
# Theme System
# =============================================================================

class Theme:
    """Theme manager with light/dark mode support."""

    # Light theme - Clean technical white
    LIGHT = {
        'name': 'light',
        'bg_primary': '#ffffff',
        'bg_secondary': '#f8f9fa',
        'bg_tertiary': '#e9ecef',
        'bg_hover': '#dee2e6',
        'bg_card': '#ffffff',
        'border': '#ced4da',
        'border_light': '#e9ecef',
        'text_primary': '#212529',
        'text_secondary': '#495057',
        'text_muted': '#6c757d',
        'accent': '#2563eb',
        'accent_hover': '#1d4ed8',
        'accent_light': '#dbeafe',
        'status_pass': '#16a34a',
        'status_fail': '#dc2626',
        'status_warning': '#d97706',
        'chart_bg': '#ffffff',
        'chart_grid': '#f1f3f5',
        'shadow': 'rgba(0,0,0,0.08)',
        'shadow_hover': 'rgba(0,0,0,0.12)',
    }

    # Dark theme - Deep technical slate
    DARK = {
        'name': 'dark',
        'bg_primary': '#0f1419',
        'bg_secondary': '#1a1f2e',
        'bg_tertiary': '#242938',
        'bg_hover': '#2d3344',
        'bg_card': '#1a1f2e',
        'border': '#374151',
        'border_light': '#2d3344',
        'text_primary': '#f1f5f9',
        'text_secondary': '#94a3b8',
        'text_muted': '#64748b',
        'accent': '#3b82f6',
        'accent_hover': '#60a5fa',
        'accent_light': '#1e3a5f',
        'status_pass': '#22c55e',
        'status_fail': '#ef4444',
        'status_warning': '#f59e0b',
        'chart_bg': '#1a1f2e',
        'chart_grid': '#2d3344',
        'shadow': 'rgba(0,0,0,0.3)',
        'shadow_hover': 'rgba(0,0,0,0.4)',
    }

    @classmethod
    def get_theme(cls, mode: str) -> dict:
        return cls.LIGHT if mode == 'light' else cls.DARK


# =============================================================================
# CSS Styles
# =============================================================================

def get_theme_css(theme: dict) -> str:
    """Generate theme-aware CSS."""
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {{
        --bg-primary: {theme['bg_primary']};
        --bg-secondary: {theme['bg_secondary']};
        --bg-tertiary: {theme['bg_tertiary']};
        --bg-hover: {theme['bg_hover']};
        --bg-card: {theme['bg_card']};
        --border: {theme['border']};
        --border-light: {theme['border_light']};
        --text-primary: {theme['text_primary']};
        --text-secondary: {theme['text_secondary']};
        --text-muted: {theme['text_muted']};
        --accent: {theme['accent']};
        --accent-hover: {theme['accent_hover']};
        --accent-light: {theme['accent_light']};
        --status-pass: {theme['status_pass']};
        --status-fail: {theme['status_fail']};
        --status-warning: {theme['status_warning']};
        --chart-bg: {theme['chart_bg']};
        --chart-grid: {theme['chart_grid']};
        --shadow: {theme['shadow']};
        --shadow-hover: {theme['shadow_hover']};
    }}

    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }}

    html, body {{
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
        line-height: 1.5;
        background: var(--bg-primary);
        color: var(--text-primary);
        transition: background 0.3s ease, color 0.3s ease;
    }}

    .dashboard-container {{
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }}

    /* === Header Bar === */
    .header-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
        height: 56px;
        background: var(--bg-card);
        border-bottom: 1px solid var(--border);
        box-shadow: 0 1px 3px var(--shadow);
        position: sticky;
        top: 0;
        z-index: 100;
    }}

    .header-left {{
        display: flex;
        align-items: baseline;
        gap: 16px;
    }}

    .logo {{
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: var(--text-primary);
    }}

    .subtitle {{
        font-size: 12px;
        color: var(--text-muted);
        font-weight: 400;
    }}

    .header-right {{
        display: flex;
        align-items: center;
        gap: 16px;
    }}

    .header-badge {{
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        background: var(--bg-tertiary);
        border: 1px solid var(--border);
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        color: var(--text-secondary);
    }}

    .header-badge.accent {{
        background: var(--accent-light);
        border-color: var(--accent);
        color: var(--accent);
    }}

    .status-indicator {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--status-pass);
        animation: pulse 2s infinite;
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    .status-label {{
        font-size: 12px;
        color: var(--text-muted);
        font-weight: 500;
    }}

    /* === Theme Toggle === */
    .theme-toggle {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px;
        background: var(--bg-tertiary);
        border: 1px solid var(--border);
        border-radius: 6px;
    }}

    .theme-btn {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-muted);
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    .theme-btn:hover {{
        background: var(--bg-hover);
        color: var(--text-primary);
    }}

    .theme-btn.active {{
        background: var(--accent);
        color: white;
    }}

    .theme-btn svg {{
        width: 18px;
        height: 18px;
    }}

    /* === Main Layout === */
    .main-layout {{
        display: grid;
        grid-template-columns: 280px 1fr 260px;
        gap: 0;
        flex: 1;
        min-height: calc(100vh - 56px);
    }}

    /* === Sidebar === */
    .sidebar {{
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
        padding: 20px;
        overflow-y: auto;
    }}

    .sidebar-section {{
        margin-bottom: 24px;
    }}

    .section-title {{
        display: block;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-muted);
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-light);
    }}

    .form-group {{
        margin-bottom: 16px;
    }}

    .form-label {{
        display: block;
        font-size: 12px;
        font-weight: 500;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }}

    .input-field {{
        width: 100%;
        padding: 8px 12px;
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: var(--text-primary);
        transition: border-color 0.2s ease;
    }}

    .input-field:focus {{
        outline: none;
        border-color: var(--accent);
    }}

    .dropdown {{
        width: 100%;
        padding: 8px 12px;
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 4px;
        font-size: 13px;
        color: var(--text-primary);
    }}

    /* Range Slider */
    .range-slider {{
        margin: 8px 0;
    }}

    .slider-labels {{
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-muted);
        margin-top: 4px;
    }}

    /* Buttons */
    .btn-group {{
        display: flex;
        gap: 8px;
    }}

    .btn {{
        flex: 1;
        padding: 10px 16px;
        border: 1px solid transparent;
        border-radius: 4px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    .btn-primary {{
        background: var(--accent);
        color: white;
        border-color: var(--accent);
    }}

    .btn-primary:hover {{
        background: var(--accent-hover);
        border-color: var(--accent-hover);
    }}

    .btn-secondary {{
        background: var(--bg-tertiary);
        color: var(--text-secondary);
        border-color: var(--border);
    }}

    .btn-secondary:hover {{
        background: var(--bg-hover);
        color: var(--text-primary);
    }}

    /* === Main Content === */
    .main-content {{
        padding: 0;
        background: var(--bg-primary);
        overflow-y: auto;
    }}

    /* Tabs */
    .tab-header {{
        padding: 20px 24px 0;
    }}

    .tab-header h3 {{
        font-size: 16px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 4px;
    }}

    .tab-header p {{
        font-size: 13px;
        color: var(--text-muted);
    }}

    /* === Right Panel === */
    .right-panel {{
        background: var(--bg-secondary);
        border-left: 1px solid var(--border);
        padding: 20px;
        overflow-y: auto;
    }}

    .panel-header {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-muted);
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-light);
    }}

    .panel-content {{
        display: flex;
        flex-direction: column;
        gap: 12px;
    }}

    .metric-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px;
        transition: all 0.2s ease;
    }}

    .metric-card:hover {{
        border-color: var(--accent);
        box-shadow: 0 2px 8px var(--shadow);
    }}

    .metric-label {{
        font-size: 11px;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 4px;
    }}

    .metric-value {{
        font-size: 24px;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-primary);
        line-height: 1.2;
    }}

    .metric-unit {{
        font-size: 12px;
        font-weight: 400;
        color: var(--text-muted);
        margin-left: 4px;
    }}

    .metric-subtext {{
        font-size: 11px;
        color: var(--text-muted);
        margin-top: 4px;
    }}

    .divider {{
        border: none;
        border-top: 1px solid var(--border);
        margin: 8px 0;
    }}

    /* === Chart Containers === */
    .chart-container {{
        background: var(--chart-bg);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px;
        margin: 16px 24px;
    }}

    .chart-container-full {{
        background: var(--chart-bg);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px;
        margin: 16px;
    }}

    /* === Stats Grid === */
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin: 16px 24px;
    }}

    .stat-card {{
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px 16px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }}

    .stat-label {{
        font-size: 11px;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }}

    .stat-value {{
        font-size: 20px;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-primary);
    }}

    /* === Data Table === */
    .data-table-container {{
        margin: 16px 24px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        overflow: hidden;
    }}

    /* === Tab Content Layout === */
    .tab-content {{
        padding: 16px 24px;
    }}

    .tab-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-top: 16px;
    }}

    .tab-grid-2col {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
        margin-top: 16px;
    }}

    /* === Analysis Text === */
    .analysis-text {{
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px;
        margin: 16px 24px;
    }}

    .analysis-text h4 {{
        font-size: 14px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 8px;
    }}

    .analysis-text p {{
        font-size: 13px;
        color: var(--text-secondary);
        line-height: 1.6;
    }}

    /* === Pareto Badge === */
    .pareto-badge {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        background: var(--accent-light);
        color: var(--accent);
        border-radius: 3px;
        font-size: 11px;
        font-weight: 600;
    }}

    /* === Empty State === */
    .empty-state {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 48px;
        color: var(--text-muted);
    }}

    .empty-state svg {{
        width: 48px;
        height: 48px;
        margin-bottom: 16px;
        opacity: 0.5;
    }}
    """


def get_dark_mode_script() -> str:
    """JavaScript for theme toggle functionality."""
    return """
    // Theme toggle functionality
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize from localStorage or system preference
        const savedTheme = localStorage.getItem('sram-dashboard-theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const initialTheme = savedTheme || (prefersDark ? 'dark' : 'light');

        setTheme(initialTheme);

        // Theme button click handlers
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const theme = this.getAttribute('data-theme');
                setTheme(theme);
                localStorage.setItem('sram-dashboard-theme', theme);
            });
        });
    });

    function setTheme(theme) {
        // Update CSS variables
        const styles = document.querySelector('style[data-theme]');
        const isDark = theme === 'dark';
        const colors = isDark ? {
            '--bg-primary': '#0f1419',
            '--bg-secondary': '#1a1f2e',
            '--bg-tertiary': '#242938',
            '--bg-hover': '#2d3344',
            '--bg-card': '#1a1f2e',
            '--border': '#374151',
            '--border-light': '#2d3344',
            '--text-primary': '#f1f5f9',
            '--text-secondary': '#94a3b8',
            '--text-muted': '#64748b',
            '--accent': '#3b82f6',
            '--accent-hover': '#60a5fa',
            '--accent-light': '#1e3a5f',
            '--status-pass': '#22c55e',
            '--status-fail': '#ef4444',
            '--status-warning': '#f59e0b',
            '--chart-bg': '#1a1f2e',
            '--chart-grid': '#2d3344',
            '--shadow': 'rgba(0,0,0,0.3)',
            '--shadow-hover': 'rgba(0,0,0,0.4)',
        } : {
            '--bg-primary': '#ffffff',
            '--bg-secondary': '#f8f9fa',
            '--bg-tertiary': '#e9ecef',
            '--bg-hover': '#dee2e6',
            '--bg-card': '#ffffff',
            '--border': '#ced4da',
            '--border-light': '#e9ecef',
            '--text-primary': '#212529',
            '--text-secondary': '#495057',
            '--text-muted': '#6c757d',
            '--accent': '#2563eb',
            '--accent-hover': '#1d4ed8',
            '--accent-light': '#dbeafe',
            '--status-pass': '#16a34a',
            '--status-fail': '#dc2626',
            '--status-warning': '#d97706',
            '--chart-bg': '#ffffff',
            '--chart-grid': '#f1f3f5',
            '--shadow': 'rgba(0,0,0,0.08)',
            '--shadow-hover': 'rgba(0,0,0,0.12)',
        };

        // Apply CSS variables to root
        const root = document.documentElement;
        Object.entries(colors).forEach(([prop, value]) => {
            root.style.setProperty(prop, value);
        });

        // Update button states
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-theme') === theme);
        });

        // Dispatch custom event for Plotly charts
        window.dispatchEvent(new CustomEvent('theme-change', { detail: { theme } }));
    }
    """


# =============================================================================
# Dashboard Class
# =============================================================================

class Dashboard:
    """Interactive dashboard for MOGA results with theme support."""

    def __init__(self, results: dict, config: dict, output_dir: Path):
        self.results = results
        self.config = config
        self.output_dir = output_dir
        self.current_theme = 'dark'  # Default theme

        # Build data structures
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
        df['Solution ID'] = range(1, len(df) + 1)

        return df

    def _build_convergence_data(self) -> pd.DataFrame:
        """Build convergence history DataFrame."""
        history = self.results.get('history', [])
        return pd.DataFrame(history)

    def _build_param_df(self) -> pd.DataFrame:
        """Build parameter distribution DataFrame."""
        return self.pareto_df.drop(
            columns=['Area', 'Power', 'Delay', 'Solution ID'],
            errors='ignore'
        )

    def _get_chart_template(self) -> go.Layout:
        """Get Plotly layout template based on current theme."""
        is_dark = self.current_theme == 'dark'
        return {
            'layout': {
                'paper_bgcolor': 'transparent',
                'plot_bgcolor': 'transparent',
                'font': {
                    'family': 'IBM Plex Sans, sans-serif',
                    'size': 12,
                    'color': '#64748b' if is_dark else '#495057',
                },
                'axis': {
                    'gridcolor': '#2d3344' if is_dark else '#f1f3f5',
                    'linecolor': '#374151' if is_dark else '#ced4da',
                    'tickcolor': '#374151' if is_dark else '#ced4da',
                },
                'colorscale': {
                    'sequential': [[0, '#3b82f6'], [1, '#1e3a5f'] if is_dark else ['#93c5fd', '#2563eb']],
                },
            }
        }

    def _create_app(self) -> Dash:
        """Create and configure Dash application."""
        app = Dash(__name__)
        app.title = "SRAM DSO-MOGA Dashboard"

        # Register callbacks before layout
        self._register_callbacks(app)

        app.layout = self._create_layout()
        return app

    def _create_layout(self) -> html.Div:
        """Create main dashboard layout."""
        return html.Div([
            # Theme stylesheet
            html.Style(get_theme_css(Theme.DARK)),

            # Theme toggle script
            html.Script(get_dark_mode_script()),

            # Header
            self._create_header(),

            # Main content
            html.Div([
                # Left sidebar
                self._create_sidebar(),

                # Main content area with tabs
                html.Div([
                    dcc.Tabs(
                        id='tabs',
                        value='tab-overview',
                        children=self._create_tabs(),
                        className='main-tabs',
                    ),
                ], className='main-content'),

                # Right summary panel
                self._create_summary_panel(),
            ], className='main-layout'),

            # Hidden data store
            dcc.Store(id='store-results', data=self._serialize_results()),
            dcc.Store(id='store-theme', data=self.current_theme),
        ], className='dashboard-container')

    def _create_tabs(self) -> List[dcc.Tab]:
        """Create tab components."""
        tabs_config = [
            ('tab-overview', 'Overview', self._create_overview_tab),
            ('tab-3d', '3D Pareto', self._create_3d_tab),
            ('tab-2d', '2D Projections', self._create_2d_tab),
            ('tab-conv', 'Convergence', self._create_convergence_tab),
            ('tab-params', 'Parameters', self._create_params_tab),
            ('tab-solutions', 'Solutions', self._create_solutions_tab),
        ]

        return [
            dcc.Tab(label=label, value=value, children=creator())
            for value, label, creator in tabs_config
        ]

    def _create_header(self) -> html.Div:
        """Create header bar with logo, badges, and theme toggle."""
        top_name = self.config.get('top_name', 'SRAM DSO-MOGA')
        n_pareto = len(self.pareto_df)

        return html.Div([
            html.Div([
                html.Span('SRAM DSO-MOGA', className='logo'),
                html.Span('Multi-Objective Genetic Algorithm', className='subtitle'),
            ], className='header-left'),

            html.Div([
                html.Div([
                    html.Span(f'Design: {top_name}', className='header-badge'),
                    html.Span(f'{n_pareto} Pareto Solutions', className='header-badge accent'),
                ], className='header-badges'),

                html.Div([
                    html.Div([
                        html.Button(
                            html.Svg(viewBox='0 0 24 24', fill='none', stroke='currentColor',
                                     children=[
                                         html.Path(d='M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707'),
                                         html.Circle(cx='12', cy='12', r='4', stroke_width='2'),
                                     ]),
                            className='theme-btn active',
                            id='btn-light',
                            data_theme='light',
                            title='Light Mode'
                        ),
                        html.Button(
                            html.Svg(viewBox='0 0 24 24', fill='none', stroke='currentColor',
                                     children=[
                                         html.Path(d='M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z'),
                                     ]),
                            className='theme-btn',
                            id='btn-dark',
                            data_theme='dark',
                            title='Dark Mode'
                        ),
                    ], className='theme-toggle'),
                ], className='status-indicator'),
            ], className='header-right'),
        ], className='header-bar')

    def _create_sidebar(self) -> html.Div:
        """Create left sidebar with controls."""
        return html.Div([
            html.Div([
                html.Span('Configuration', className='section-title'),
                html.Div([
                    html.Label('Population Size', className='form-label'),
                    dcc.Input(
                        id='input-pop-size', type='number',
                        value=self.config.get('algorithm', {}).get('pop_size', 80),
                        className='input-field'
                    ),
                ], className='form-group'),
                html.Div([
                    html.Label('Generations', className='form-label'),
                    dcc.Input(
                        id='input-n-gen', type='number',
                        value=self.config.get('algorithm', {}).get('n_gen', 60),
                        className='input-field'
                    ),
                ], className='form-group'),
                html.Div([
                    html.Label('Algorithm', className='form-label'),
                    dcc.Dropdown(
                        id='input-algo',
                        options=[
                            {'label': 'NSGA-II', 'value': 'NSGA-II'},
                            {'label': 'NSGA-III', 'value': 'NSGA-III'},
                        ],
                        value=self.config.get('algorithm', {}).get('name', 'NSGA-II'),
                        className='dropdown'
                    ),
                ], className='form-group'),
            ], className='sidebar-section'),

            html.Div([
                html.Span('Objective Filters', className='section-title'),
                html.Div([
                    html.Label('Area Range (um²)', className='form-label'),
                    dcc.RangeSlider(
                        id='slider-area',
                        min=0, max=100, step=1,
                        value=[0, 100],
                        className='range-slider'
                    ),
                    html.Div(id='slider-area-labels', className='slider-labels'),
                ], className='form-group'),
                html.Div([
                    html.Label('Power Range (uW)', className='form-label'),
                    dcc.RangeSlider(
                        id='slider-power',
                        min=0, max=1000, step=10,
                        value=[0, 1000],
                        className='range-slider'
                    ),
                    html.Div(id='slider-power-labels', className='slider-labels'),
                ], className='form-group'),
                html.Div([
                    html.Label('Delay Range (ps)', className='form-label'),
                    dcc.RangeSlider(
                        id='slider-delay',
                        min=0, max=1000, step=10,
                        value=[0, 1000],
                        className='range-slider'
                    ),
                    html.Div(id='slider-delay-labels', className='slider-labels'),
                ], className='form-group'),
            ], className='sidebar-section'),

            html.Div([
                html.Span('Actions', className='section-title'),
                html.Div([
                    html.Button('Apply Filters', id='btn-apply', className='btn btn-primary'),
                    html.Button('Reset', id='btn-reset', className='btn btn-secondary'),
                ], className='btn-group'),
            ], className='sidebar-section'),
        ], className='sidebar')

    def _create_summary_panel(self) -> html.Div:
        """Create right summary panel with key metrics."""
        objectives = self.pareto_df[['Area', 'Power', 'Delay']]

        # Calculate statistics
        stats = {
            'area': {
                'min': objectives['Area'].min() if len(objectives) else 0,
                'mean': objectives['Area'].mean() if len(objectives) else 0,
            },
            'power': {
                'min': objectives['Power'].min() if len(objectives) else 0,
                'mean': objectives['Power'].mean() if len(objectives) else 0,
            },
            'delay': {
                'min': objectives['Delay'].min() if len(objectives) else 0,
                'mean': objectives['Delay'].mean() if len(objectives) else 0,
            },
        }

        return html.Div([
            html.Div('Pareto Summary', className='panel-header'),
            html.Div([
                # Best Area
                html.Div([
                    html.Div('Best Area', className='metric-label'),
                    html.Div([
                        html.Span(f"{stats['area']['min']:.3f}", className='metric-value'),
                        html.Span('um²', className='metric-unit'),
                    ]),
                    html.Div(f'Mean: {stats["area"]["mean"]:.3f}', className='metric-subtext'),
                ], className='metric-card'),

                # Best Power
                html.Div([
                    html.Div('Best Power', className='metric-label'),
                    html.Div([
                        html.Span(f"{stats['power']['min']:.2f}", className='metric-value'),
                        html.Span('uW', className='metric-unit'),
                    ]),
                    html.Div(f'Mean: {stats["power"]["mean"]:.2f}', className='metric-subtext'),
                ], className='metric-card'),

                # Best Delay
                html.Div([
                    html.Div('Best Delay', className='metric-label'),
                    html.Div([
                        html.Span(f"{stats['delay']['min']:.2f}", className='metric-value'),
                        html.Span('ps', className='metric-unit'),
                    ]),
                    html.Div(f'Mean: {stats["delay"]["mean"]:.2f}', className='metric-subtext'),
                ], className='metric-card'),

                html.Hr(className='divider'),

                # Total Solutions
                html.Div([
                    html.Div('Pareto Size', className='metric-label'),
                    html.Div([
                        html.Span(f"{len(self.pareto_df)}", className='metric-value'),
                    ]),
                    html.Div('Non-dominated solutions', className='metric-subtext'),
                ], className='metric-card'),
            ], className='panel-content'),
        ], className='right-panel')

    def _create_overview_tab(self) -> html.Div:
        """Create overview tab with summary charts."""
        if len(self.pareto_df) == 0:
            return html.Div([
                html.Div('Optimization Results Summary', className='tab-header'),
                html.Div([
                    html.H3('No Pareto Solutions Found'),
                    html.P('Run the optimization first to see results.'),
                ], className='empty-state'),
            ])

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Area vs Power', 'Area vs Delay', 'Power vs Delay'],
            horizontal_spacing=0.08,
        )

        obj = self.pareto_df[['Area', 'Power', 'Delay']].values
        colors = obj[:, 2]  # Use Delay for coloring

        # Area vs Power
        fig.add_trace(
            go.Scatter(
                x=obj[:, 0], y=obj[:, 1],
                mode='markers',
                marker=dict(size=8, color=colors, colorscale='Viridis', showscale=False),
                hovertemplate='Area: %{x:.3f}<br>Power: %{y:.2f}<extra></extra>',
            ),
            row=1, col=1
        )

        # Area vs Delay
        fig.add_trace(
            go.Scatter(
                x=obj[:, 0], y=obj[:, 2],
                mode='markers',
                marker=dict(size=8, color=colors, colorscale='Viridis', showscale=False),
                hovertemplate='Area: %{x:.3f}<br>Delay: %{y:.2f}<extra></extra>',
            ),
            row=1, col=2
        )

        # Power vs Delay
        fig.add_trace(
            go.Scatter(
                x=obj[:, 1], y=obj[:, 2],
                mode='markers',
                marker=dict(size=8, color=obj[:, 0], colorscale='Plasma', showscale=False),
                hovertemplate='Power: %{x:.2f}<br>Delay: %{y:.2f}<extra></extra>',
            ),
            row=1, col=3
        )

        fig.update_layout(
            title='Pareto Front Overview',
            height=380,
            showlegend=False,
            title_font_size=14,
            title_x=0.5,
            margin=dict(l=40, r=40, t=50, b=40),
        )

        # Update axes
        for i in range(1, 4):
            fig.update_xaxes(title_text=['Area (um²)', 'Area (um²)', 'Power (uW)'][i-1], row=1, col=i)
            fig.update_yaxes(title_text=['Power (uW)', 'Delay (ps)', 'Delay (ps)'][i-1], row=1, col=i)

        return html.Div([
            html.Div([
                html.H3('Optimization Results Summary'),
                html.P(f'Found {len(self.pareto_df)} Pareto-optimal solutions across 3 objectives'),
            ], className='tab-header'),
            html.Div(go.Figure(fig), className='chart-container'),
            self._create_quick_stats(),
        ])

    def _create_3d_tab(self) -> html.Div:
        """Create 3D Pareto visualization tab."""
        if len(self.pareto_df) == 0:
            return html.Div([html.Div('No data available', className='empty-state')])

        obj = self.pareto_df[['Area', 'Power', 'Delay']].values

        fig = go.Figure(data=[go.Scatter3d(
            x=obj[:, 0],
            y=obj[:, 1],
            z=obj[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=obj[:, 2],
                colorscale='Viridis',
                opacity=0.85,
                line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
            ),
            text=[f'Solution {i+1}<br>Area: {obj[i,0]:.3f}<br>Power: {obj[i,1]:.2f}<br>Delay: {obj[i,2]:.2f}'
                  for i in range(len(obj))],
            hoverinfo='text',
        )])

        fig.update_layout(
            title='3D Pareto Front Visualization',
            scene=dict(
                xaxis_title='Area (um²)',
                yaxis_title='Power (uW)',
                zaxis_title='Delay (ps)',
                xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#2d3344'),
                yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#2d3344'),
                zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#2d3344'),
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0),
            title_font_size=14,
            title_x=0.5,
        )

        return html.Div([
            dcc.Graph(figure=fig, className='chart-container-full'),
        ])

    def _create_2d_tab(self) -> html.Div:
        """Create 2D projection charts."""
        if len(self.pareto_df) == 0:
            return html.Div([html.Div('No data available', className='empty-state')])

        obj = self.pareto_df[['Area', 'Power', 'Delay']].values

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['', '', '', '', '', ''],
            horizontal_spacing=0.1,
            vertical_spacing=0.12,
        )

        # Row 1: Objective pairs
        fig.add_trace(
            go.Scatter(x=obj[:, 0], y=obj[:, 1], mode='markers',
                      marker=dict(color=obj[:, 2], colorscale='Viridis', size=7),
                      hovertemplate='Area: %{x:.3f}<br>Power: %{y:.2f}<extra></extra>'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=obj[:, 0], y=obj[:, 2], mode='markers',
                      marker=dict(color=obj[:, 1], colorscale='Plasma', size=7),
                      hovertemplate='Area: %{x:.3f}<br>Delay: %{y:.2f}<extra></extra>'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=obj[:, 1], y=obj[:, 2], mode='markers',
                      marker=dict(color=obj[:, 0], colorscale='Cividis', size=7),
                      hovertemplate='Power: %{x:.2f}<br>Delay: %{y:.2f}<extra></extra>'),
            row=1, col=3
        )

        # Row 2: Parallel coordinates style (normalized)
        obj_norm = (obj - obj.min(axis=0)) / (obj.max(axis=0) - obj.min(axis=0) + 1e-10)

        fig.add_trace(
            go.Scatter(x=obj_norm[:, 0], y=obj_norm[:, 1], mode='markers',
                      marker=dict(color=range(len(obj)), colorscale='Viridis', size=7),
                      showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=obj_norm[:, 0], y=obj_norm[:, 2], mode='markers',
                      marker=dict(color=range(len(obj)), colorscale='Viridis', size=7),
                      showlegend=False),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=obj_norm[:, 1], y=obj_norm[:, 2], mode='markers',
                      marker=dict(color=range(len(obj)), colorscale='Viridis', size=7),
                      showlegend=False),
            row=2, col=3
        )

        # Add axis labels
        fig.update_xaxes(title_text='Area', row=1, col=1)
        fig.update_yaxes(title_text='Power', row=1, col=1)
        fig.update_xaxes(title_text='Area', row=1, col=2)
        fig.update_yaxes(title_text='Delay', row=1, col=2)
        fig.update_xaxes(title_text='Power', row=1, col=3)
        fig.update_yaxes(title_text='Delay', row=1, col=3)

        # Normalized axis labels
        fig.update_xaxes(title_text='Area (norm)', row=2, col=1)
        fig.update_yaxes(title_text='Power (norm)', row=2, col=1)
        fig.update_xaxes(title_text='Area (norm)', row=2, col=2)
        fig.update_yaxes(title_text='Delay (norm)', row=2, col=2)
        fig.update_xaxes(title_text='Power (norm)', row=2, col=3)
        fig.update_yaxes(title_text='Delay (norm)', row=2, col=3)

        fig.update_layout(
            title='2D Objective Projections (Top: Raw, Bottom: Normalized)',
            height=650,
            showlegend=False,
            title_font_size=13,
            title_x=0.5,
            margin=dict(l=40, r=40, t=50, b=40),
        )

        return html.Div([dcc.Graph(figure=fig)])

    def _create_convergence_tab(self) -> html.Div:
        """Create convergence analysis tab."""
        conv_df = self.convergence_data

        if len(conv_df) == 0:
            return html.Div([html.Div('No convergence data available', className='empty-state')])

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Pareto Front Size per Generation', 'Average Fitness per Generation'],
            vertical_spacing=0.15,
        )

        # Front 0 size
        fig.add_trace(
            go.Scatter(
                x=conv_df['gen'],
                y=conv_df.get('front0_size', [0]*len(conv_df)),
                mode='lines+markers',
                name='Front 0 Size',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=6),
            ),
            row=1, col=1
        )

        # Average fitness if available
        if 'avg_fitness' in conv_df.columns:
            avg_fit = conv_df['avg_fitness'].tolist()
            if avg_fit and isinstance(avg_fit[0], list):
                # Multi-objective: plot each objective
                obj_labels = ['Area', 'Power', 'Delay']
                colors = ['#3b82f6', '#22c55e', '#f59e0b']
                for i, (label, color) in enumerate(zip(obj_labels, colors)):
                    fig.add_trace(
                        go.Scatter(
                            x=conv_df['gen'],
                            y=[f[i] if len(f) > i else 0 for f in avg_fit],
                            mode='lines',
                            name=f'Avg {label}',
                            line=dict(color=color, width=1.5, dash='dash'),
                        ),
                        row=2, col=1
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=conv_df['gen'],
                        y=avg_fit,
                        mode='lines+markers',
                        name='Avg Fitness',
                        line=dict(color='#22c55e', width=2),
                    ),
                    row=2, col=1
                )

        fig.update_layout(
            height=500,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            title_font_size=13,
            title_x=0.5,
            margin=dict(l=50, r=50, t=50, b=40),
        )

        return html.Div([
            dcc.Graph(figure=fig),
            html.Div([
                html.H4('Convergence Analysis'),
                html.P('The Pareto front size should stabilize as the algorithm converges to the optimal set of non-dominated solutions. The average fitness trends indicate how each objective evolves across generations.'),
            ], className='analysis-text'),
        ])

    def _create_params_tab(self) -> html.Div:
        """Create parameter distribution tab."""
        param_cols = [c for c in self.param_df.columns if c and c not in ['Area', 'Power', 'Delay', 'Solution ID']]

        if not param_cols:
            return html.Div([html.Div('No parameter data available', className='empty-state')])

        n_params = len(param_cols)
        n_rows = (n_params + 2) // 3

        fig = make_subplots(
            rows=n_rows, cols=3,
            subplot_titles=param_cols[:12],  # Limit to 12 for readability
            vertical_spacing=0.2,
            horizontal_spacing=0.1,
        )

        colors = px.colors.qualitative.Set2

        for i, col in enumerate(param_cols[:12]):
            row = i // 3 + 1
            col_idx = i % 3 + 1

            # Get unique values and counts
            vals = self.param_df[col].astype(str)
            val_counts = vals.value_counts()

            fig.add_trace(
                go.Bar(
                    x=val_counts.index,
                    y=val_counts.values,
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f'{col}: %{{x}}<br>Count: %{{y}}<extra></extra>',
                ),
                row=row, col=col_idx
            )

        fig.update_layout(
            title=f'Parameter Distributions ({len(param_cols)} parameters)',
            height=max(300, 120 * n_rows),
            showlegend=False,
            title_font_size=13,
            title_x=0.5,
            margin=dict(l=40, r=40, t=50, b=40),
        )

        return html.Div([dcc.Graph(figure=fig)])

    def _create_solutions_tab(self) -> html.Div:
        """Create solutions table tab."""
        display_df = self.pareto_df.copy()
        display_df['ID'] = range(1, len(display_df) + 1)

        # Reorder columns
        cols = ['ID'] + [c for c in display_df.columns if c != 'ID']
        display_df = display_df[cols]

        return html.Div([
            dash_table.DataTable(
                data=display_df.to_dict('records'),
                columns=[{'name': c, 'id': c} for c in display_df.columns],
                page_size=20,
                sort_action='native',
                filter_action='native',
                style_cell={
                    'textAlign': 'left',
                    'fontFamily': "'IBM Plex Mono', monospace",
                    'fontSize': '12px',
                    'padding': '8px 12px',
                },
                style_header={
                    'backgroundColor': 'var(--bg-tertiary)',
                    'fontWeight': '600',
                    'textTransform': 'uppercase',
                    'fontSize': '11px',
                    'letterSpacing': '0.5px',
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': 'var(--bg-secondary)'},
                    {'if': {'filter_query': '{ID} <= 5'}, 'backgroundColor': 'var(--accent-light)'},
                ],
                style_table={
                    'overflowX': 'auto',
                    'maxHeight': '600px',
                },
            ),
        ], className='data-table-container')

    def _create_quick_stats(self) -> html.Div:
        """Create quick statistics section."""
        obj = self.pareto_df[['Area', 'Power', 'Delay']]

        return html.Div([
            html.Div('Quick Statistics', className='section-title'),
            html.Div([
                html.Div([
                    html.Span('Mean Area:', className='stat-label'),
                    html.Span(f"{obj['Area'].mean():.3f} um²", className='stat-value'),
                ], className='stat-card'),
                html.Div([
                    html.Span('Mean Power:', className='stat-label'),
                    html.Span(f"{obj['Power'].mean():.2f} uW", className='stat-value'),
                ], className='stat-card'),
                html.Div([
                    html.Span('Mean Delay:', className='stat-label'),
                    html.Span(f"{obj['Delay'].mean():.2f} ps", className='stat-value'),
                ], className='stat-card'),
                html.Div([
                    html.Span('Std Area:', className='stat-label'),
                    html.Span(f"{obj['Area'].std():.3f}", className='stat-value'),
                ], className='stat-card'),
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
            Output('tabs', 'value'),
            [Input('btn-reset', 'n_clicks')]
        )
        def reset_filters(n_clicks):
            return 'tab-overview'

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

    def run(self, port: int = 8050, debug: bool = False) -> None:
        """Run the dashboard server."""
        self.app.run(debug=debug, port=port, host='0.0.0.0')

    def save_html(self, path: Optional[Path] = None) -> Path:
        """Export dashboard as standalone HTML."""
        path = path or self.output_dir / 'dashboard.html'

        obj = self.pareto_df[['Area', 'Power', 'Delay']].values

        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{'type': 'scatter3d'}, None, None],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]
            ],
            subplot_titles=['3D Pareto Front', 'Area vs Power', 'Area vs Delay', '', 'Power vs Delay', ''],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        fig.add_trace(
            go.Scatter3d(
                x=obj[:, 0], y=obj[:, 1], z=obj[:, 2],
                mode='markers',
                marker=dict(size=5, color=obj[:, 2], colorscale='Viridis'),
            ),
            row=1, col=1
        )

        fig.add_trace(go.Scatter(x=obj[:, 0], y=obj[:, 1], mode='markers'), row=2, col=1)
        fig.add_trace(go.Scatter(x=obj[:, 0], y=obj[:, 2], mode='markers'), row=2, col=2)
        fig.add_trace(go.Scatter(x=obj[:, 1], y=obj[:, 2], mode='markers'), row=2, col=3)

        fig.update_layout(
            title='SRAM DSO-MOGA Results',
            height=800,
            showlegend=False,
        )

        fig.write_html(str(path), include_plotlyjs='cdn')
        return path


def create_dashboard(results: dict, config: dict, output_dir: Path) -> Dashboard:
    """Factory function to create dashboard."""
    return Dashboard(results, config, output_dir)

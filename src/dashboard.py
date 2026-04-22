"""
SRAM DSO-MOGA: Professional EDA-Style Dashboard.

Professional EDA tool aesthetic for multi-objective genetic algorithm optimization results.
Inspired by Cadence Virtuoso, Synopsys Design Compiler - industrial/utilitarian design.

Features:
- Light/dark theme with persistent toggle
- Left sidebar for controls and filters
- Right panel for visualizations and data
- Real-time Pareto front analysis
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
    State, Input, Output, callback_context as cb_context,
)


# =============================================================================
# EDA Theme System
# =============================================================================

class EDATheme:
    """Professional EDA tool theme with light/dark mode support."""

    # Dark Theme - Industrial EDA aesthetic
    DARK = {
        'name': 'dark',
        'bg_primary': '#0d1117',      # GitHub dark-style background
        'bg_secondary': '#161b22',    # Elevated panels
        'bg_tertiary': '#21262d',    # Cards and inputs
        'bg_hover': '#30363d',        # Hover states
        'bg_input': '#0d1117',        # Input backgrounds
        'border_primary': '#30363d',  # Borders
        'border_secondary': '#21262d',
        'border_active': '#58a6ff',
        'text_primary': '#e6edf3',    # Primary text
        'text_secondary': '#8b949e',  # Secondary text
        'text_muted': '#6e7681',     # Muted text
        'text_accent': '#58a6ff',     # Accent text
        'status_pass': '#3fb950',     # Green - pass
        'status_fail': '#f85149',     # Red - fail
        'status_warning': '#d29922',  # Amber - warning
        'status_info': '#58a6ff',     # Blue - info
        'accent_primary': '#58a6ff',  # Primary blue
        'accent_secondary': '#79c0ff',
        'button_primary': '#238636',
        'button_hover': '#2ea043',
        'button_active': '#1a7f37',
        'graph_bg': '#0d1117',
        'shadow_sm': '0 1px 2px rgba(0, 0, 0, 0.3)',
        'shadow_md': '0 4px 6px rgba(0, 0, 0, 0.4)',
        'shadow_lg': '0 10px 15px rgba(0, 0, 0, 0.5)',
    }

    # Light Theme - Clean technical white
    LIGHT = {
        'name': 'light',
        'bg_primary': '#ffffff',
        'bg_secondary': '#f6f8fa',
        'bg_tertiary': '#eaeef2',
        'bg_hover': '#d0d7de',
        'bg_input': '#ffffff',
        'border_primary': '#d0d7de',
        'border_secondary': '#eaeef2',
        'border_active': '#0969da',
        'text_primary': '#1f2328',
        'text_secondary': '#656d76',
        'text_muted': '#8c959f',
        'text_accent': '#0969da',
        'status_pass': '#1a7f37',
        'status_fail': '#cf222e',
        'status_warning': '#9a6700',
        'status_info': '#0969da',
        'accent_primary': '#0969da',
        'accent_secondary': '#0550ae',
        'button_primary': '#2ea043',
        'button_hover': '#238636',
        'button_active': '#1a7f37',
        'graph_bg': '#ffffff',
        'shadow_sm': '0 1px 2px rgba(0, 0, 0, 0.08)',
        'shadow_md': '0 4px 6px rgba(0, 0, 0, 0.1)',
        'shadow_lg': '0 10px 15px rgba(0, 0, 0, 0.15)',
    }

    @classmethod
    def get(cls, mode: str) -> dict:
        return cls.DARK if mode == 'dark' else cls.LIGHT


# =============================================================================
# CSS Styles
# =============================================================================

def get_dash_css(theme: dict) -> str:
    """Generate complete CSS for EDA-style dashboard."""
    return f"""
/* === Reset & Base === */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
    --bg-primary: {theme['bg_primary']};
    --bg-secondary: {theme['bg_secondary']};
    --bg-tertiary: {theme['bg_tertiary']};
    --bg-hover: {theme['bg_hover']};
    --bg-input: {theme['bg_input']};
    --border-primary: {theme['border_primary']};
    --border-secondary: {theme['border_secondary']};
    --border-active: {theme['border_active']};
    --text-primary: {theme['text_primary']};
    --text-secondary: {theme['text_secondary']};
    --text-muted: {theme['text_muted']};
    --text-accent: {theme['text_accent']};
    --status-pass: {theme['status_pass']};
    --status-fail: {theme['status_fail']};
    --status-warning: {theme['status_warning']};
    --status-info: {theme['status_info']};
    --accent-primary: {theme['accent_primary']};
    --accent-secondary: {theme['accent_secondary']};
    --button-primary: {theme['button_primary']};
    --button-hover: {theme['button_hover']};
    --button-active: {theme['button_active']};
    --graph-bg: {theme['graph_bg']};
    --shadow-sm: {theme['shadow_sm']};
    --shadow-md: {theme['shadow_md']};
    --shadow-lg: {theme['shadow_lg']};

    /* Typography */
    --font-display: 'JetBrains Mono', 'Fira Code', 'SF Mono', Consolas, monospace;
    --font-body: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-data: 'JetBrains Mono', 'Fira Code', 'SF Mono', Consolas, monospace;

    /* Spacing */
    --space-xs: 4px;
    --space-sm: 8px;
    --space-md: 12px;
    --space-lg: 16px;
    --space-xl: 20px;
    --space-2xl: 24px;

    /* Radius */
    --radius-sm: 3px;
    --radius-md: 5px;
    --radius-lg: 8px;
}}

html, body {{
    height: 100%;
    overflow: hidden;
}}

body {{
    font-family: var(--font-body);
    font-size: 13px;
    line-height: 1.5;
    color: var(--text-primary);
    background: var(--bg-primary);
    transition: background-color 0.2s ease, color 0.2s ease;
}}

/* === Scrollbar === */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: var(--bg-secondary); }}
::-webkit-scrollbar-thumb {{ background: var(--border-primary); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent-primary); }}
::-webkit-scrollbar-corner {{ background: var(--bg-secondary); }}

/* === App Container === */
.app-container {{
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
}}

/* === Header Bar === */
.header-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 44px;
    padding: 0 var(--space-lg);
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-primary);
    flex-shrink: 0;
}}

.header-left {{
    display: flex;
    align-items: center;
    gap: var(--space-lg);
}}

.header-logo {{
    font-family: var(--font-display);
    font-size: 14px;
    font-weight: 700;
    color: var(--text-accent);
    letter-spacing: 0.05em;
}}

.header-subtitle {{
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.02em;
}}

.header-right {{
    display: flex;
    align-items: center;
    gap: var(--space-md);
}}

.header-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    font-size: 11px;
    font-family: var(--font-data);
    background: var(--bg-tertiary);
    border: 1px solid var(--border-primary);
    border-radius: 12px;
    color: var(--text-secondary);
}}

.header-badge.accent {{
    background: rgba(88, 166, 255, 0.15);
    border-color: var(--accent-primary);
    color: var(--accent-primary);
}}

.status-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--status-pass);
}}

/* === Theme Toggle === */
.theme-toggle {{
    display: flex;
    align-items: center;
    padding: 4px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-primary);
    border-radius: 20px;
    gap: 2px;
}}

.theme-btn {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border: none;
    border-radius: 16px;
    background: transparent;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
}}

.theme-btn:hover {{
    background: var(--bg-hover);
    color: var(--text-primary);
}}

.theme-btn.active {{
    background: var(--accent-primary);
    color: white;
}}

/* === Main Content === */
.main-content {{
    display: flex;
    flex: 1;
    overflow: hidden;
}}

/* === Left Sidebar === */
.sidebar {{
    width: 280px;
    min-width: 280px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-primary);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
}}

.sidebar-section {{
    padding: var(--space-md);
    border-bottom: 1px solid var(--border-secondary);
}}

.sidebar-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: var(--space-md);
}}

.sidebar-title {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
}}

.form-group {{
    margin-bottom: var(--space-md);
}}

.form-group:last-child {{
    margin-bottom: 0;
}}

.form-label {{
    display: block;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: var(--space-xs);
}}

.form-select {{
    width: 100%;
    padding: 6px 10px;
    font-size: 12px;
    font-family: var(--font-body);
    color: var(--text-primary);
    background: var(--bg-input);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: border-color 0.2s;
}}

.form-select:focus {{
    outline: none;
    border-color: var(--accent-primary);
}}

.range-slider-container {{
    padding: var(--space-sm) 0;
}}

.range-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    font-family: var(--font-data);
    color: var(--text-muted);
    margin-top: var(--space-xs);
}}

/* === Right Panel === */
.right-panel {{
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg-primary);
}}

.panel-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-md) var(--space-lg);
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-primary);
    flex-shrink: 0;
}}

.panel-title {{
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
}}

.panel-tabs {{
    display: flex;
    gap: var(--space-xs);
}}

.panel-tab {{
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.2s;
}}

.panel-tab:hover {{
    color: var(--text-primary);
    background: var(--bg-hover);
}}

.panel-tab.active {{
    color: var(--accent-primary);
    background: rgba(88, 166, 255, 0.1);
    border-color: var(--accent-primary);
}}

/* === Chart Area === */
.chart-container {{
    flex: 1;
    padding: var(--space-lg);
    overflow: auto;
}}

/* === Metrics Grid === */
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--space-md);
    padding: var(--space-lg);
}}

.metric-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-md);
    padding: var(--space-md);
    transition: all 0.2s;
}}

.metric-card:hover {{
    border-color: var(--accent-primary);
    box-shadow: var(--shadow-sm);
}}

.metric-label {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: var(--space-xs);
}}

.metric-value {{
    font-size: 24px;
    font-weight: 700;
    font-family: var(--font-data);
    color: var(--text-primary);
    line-height: 1.2;
}}

.metric-value.best {{
    color: var(--status-pass);
}}

.metric-unit {{
    font-size: 12px;
    font-weight: 400;
    color: var(--text-muted);
    margin-left: 2px;
}}

.metric-subtext {{
    font-size: 10px;
    color: var(--text-muted);
    margin-top: var(--space-xs);
}}

/* === Data Table === */
.data-table-wrapper {{
    flex: 1;
    overflow: auto;
    padding: var(--space-lg);
}}

/* === Stats Cards === */
.stats-row {{
    display: flex;
    gap: var(--space-md);
    padding: var(--space-md) var(--space-lg);
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-primary);
}}

.stat-item {{
    display: flex;
    align-items: center;
    gap: var(--space-sm);
}}

.stat-value {{
    font-size: 16px;
    font-weight: 700;
    font-family: var(--font-data);
    color: var(--text-primary);
}}

.stat-label {{
    font-size: 11px;
    color: var(--text-muted);
}}

/* === Empty State === */
.empty-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
}}

.empty-state-icon {{
    font-size: 48px;
    margin-bottom: var(--space-lg);
    opacity: 0.5;
}}

/* === Tabs === */
.tabs-container {{
    display: flex;
    flex-direction: column;
    flex: 1;
    overflow: hidden;
}}

.tab-content {{
    flex: 1;
    overflow: auto;
}}

/* Dash overrides */
.dash-tabs {{
    background: var(--bg-secondary) !important;
    border-bottom: 1px solid var(--border-primary) !important;
}}

.dash-tab {{
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 16px !important;
}}

.dash-tab:hover {{
    color: var(--text-primary) !important;
    background: var(--bg-hover) !important;
}}

.dash-tab--selected {{
    color: var(--accent-primary) !important;
    border-top: 2px solid var(--accent-primary) !important;
}}

/* Data Table */
.dash-table {{
    background: var(--bg-secondary) !important;
}}

.dash-table-container {{
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-primary) !important;
    border-radius: var(--radius-md) !important;
}}

.dash-header {{
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 10px !important;
    letter-spacing: 0.05em;
}}

.dash-cell {{
    color: var(--text-primary) !important;
    font-family: var(--font-data) !important;
    font-size: 12px !important;
}}
"""


def get_theme_script() -> str:
    """JavaScript for theme toggle functionality."""
    return """
    // Theme Toggle Logic
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize theme from localStorage or system preference
        const savedTheme = localStorage.getItem('sram-dashboard-theme') || 'dark';
        setTheme(savedTheme);

        // Theme button click handlers
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const theme = this.dataset.theme;
                setTheme(theme);
                localStorage.setItem('sram-dashboard-theme', theme);
            });
        });
    });

    function setTheme(theme) {
        const isDark = theme === 'dark';
        const colors = isDark ? {
            '--bg-primary': '#0d1117',
            '--bg-secondary': '#161b22',
            '--bg-tertiary': '#21262d',
            '--bg-hover': '#30363d',
            '--bg-input': '#0d1117',
            '--border-primary': '#30363d',
            '--border-secondary': '#21262d',
            '--border-active': '#58a6ff',
            '--text-primary': '#e6edf3',
            '--text-secondary': '#8b949e',
            '--text-muted': '#6e7681',
            '--text-accent': '#58a6ff',
            '--status-pass': '#3fb950',
            '--status-fail': '#f85149',
            '--status-warning': '#d29922',
            '--status-info': '#58a6ff',
            '--accent-primary': '#58a6ff',
            '--accent-secondary': '#79c0ff',
            '--button-primary': '#238636',
            '--button-hover': '#2ea043',
            '--button-active': '#1a7f37',
            '--graph-bg': '#0d1117',
            '--shadow-sm': '0 1px 2px rgba(0, 0, 0, 0.3)',
            '--shadow-md': '0 4px 6px rgba(0, 0, 0, 0.4)',
            '--shadow-lg': '0 10px 15px rgba(0, 0, 0, 0.5)',
        } : {
            '--bg-primary': '#ffffff',
            '--bg-secondary': '#f6f8fa',
            '--bg-tertiary': '#eaeef2',
            '--bg-hover': '#d0d7de',
            '--bg-input': '#ffffff',
            '--border-primary': '#d0d7de',
            '--border-secondary': '#eaeef2',
            '--border-active': '#0969da',
            '--text-primary': '#1f2328',
            '--text-secondary': '#656d76',
            '--text-muted': '#8c959f',
            '--text-accent': '#0969da',
            '--status-pass': '#1a7f37',
            '--status-fail': '#cf222e',
            '--status-warning': '#9a6700',
            '--status-info': '#0969da',
            '--accent-primary': '#0969da',
            '--accent-secondary': '#0550ae',
            '--button-primary': '#2ea043',
            '--button-hover': '#238636',
            '--button-active': '#1a7f37',
            '--graph-bg': '#ffffff',
            '--shadow-sm': '0 1px 2px rgba(0, 0, 0, 0.08)',
            '--shadow-md': '0 4px 6px rgba(0, 0, 0, 0.1)',
            '--shadow-lg': '0 10px 15px rgba(0, 0, 0, 0.15)',
        };

        // Apply CSS variables to root
        const root = document.documentElement;
        Object.entries(colors).forEach(([prop, value]) => {
            root.style.setProperty(prop, value);
        });

        // Update button states
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === theme);
        });

        // Dispatch event for Plotly charts
        window.dispatchEvent(new CustomEvent('theme-change', { detail: { theme } }));
    }
    """


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
        """Create main dashboard layout."""
        theme = EDATheme.get(self.current_theme)

        return html.Div([
            # Header
            self._create_header(),

            # Main content
            html.Div([
                # Left sidebar - controls
                self._create_sidebar(),

                # Right panel - visualizations
                self._create_right_panel(),
            ], className='main-content'),
        ], className='app-container')

    def _create_header(self) -> html.Div:
        """Create header bar with logo, badges, and theme toggle."""
        top_name = self.config.get('top_name', 'SRAM DSO-MOGA')
        n_pareto = len(self.pareto_df)
        algo_name = self.config.get('algorithm', {}).get('name', 'NSGA-II')

        return html.Div([
            # Left: Logo and title
            html.Div([
                html.Span('SRAM DSO-MOGA', className='header-logo'),
                html.Span('Multi-Objective Genetic Algorithm', className='header-subtitle'),
            ], className='header-left'),

            # Right: Status badges and theme toggle
            html.Div([
                html.Span(f'Design: {top_name}', className='header-badge'),
                html.Span(f'{n_pareto} Pareto', className='header-badge accent'),
                html.Span(f'Algo: {algo_name}', className='header-badge'),

                # Theme toggle
                html.Div([
                    html.Button('☀', className='theme-btn active', id='btn-light', title='Light Mode'),
                    html.Button('☾', className='theme-btn', id='btn-dark', title='Dark Mode'),
                ], className='theme-toggle'),

                # Online indicator
                html.Div([
                    html.Span(className='status-dot'),
                    html.Span('Ready', className='header-badge'),
                ]),
            ], className='header-right'),
        ], className='header-bar')

    def _create_sidebar(self) -> html.Div:
        """Create left sidebar with controls."""
        return html.Div([
            # Algorithm Settings
            html.Div([
                html.Span('Algorithm Settings', className='sidebar-title'),
            ], className='sidebar-section'),

            html.Div([
                html.Div([
                    html.Label('Population Size', className='form-label'),
                    dcc.Input(
                        id='input-pop-size', type='number',
                        value=self.config.get('algorithm', {}).get('pop_size', 80),
                        className='form-select',
                        style={'width': '100%', 'padding': '6px 10px', 'background': 'var(--bg-input)',
                               'border': '1px solid var(--border-primary)', 'border-radius': '3px',
                               'color': 'var(--text-primary)', 'fontSize': '12px'}
                    ),
                ], className='form-group'),
                html.Div([
                    html.Label('Generations', className='form-label'),
                    dcc.Input(
                        id='input-n-gen', type='number',
                        value=self.config.get('algorithm', {}).get('n_gen', 60),
                        className='form-select',
                        style={'width': '100%', 'padding': '6px 10px', 'background': 'var(--bg-input)',
                               'border': '1px solid var(--border-primary)', 'border-radius': '3px',
                               'color': 'var(--text-primary)', 'fontSize': '12px'}
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
                        className='form-select',
                        style={'background': 'var(--bg-input)'}
                    ),
                ], className='form-group'),
            ], className='sidebar-section'),

            # Objective Filters
            html.Div([
                html.Span('Objective Filters', className='sidebar-title'),
            ], className='sidebar-section'),

            html.Div([
                html.Div([
                    html.Label('Area Range (um²)', className='form-label'),
                    dcc.RangeSlider(
                        id='slider-area',
                        min=0, max=100, step=1,
                        value=[0, 100],
                        className='range-slider-container',
                        marks=None,
                    ),
                    html.Div(id='slider-area-labels', className='range-labels'),
                ], className='form-group'),
                html.Div([
                    html.Label('Power Range (uW)', className='form-label'),
                    dcc.RangeSlider(
                        id='slider-power',
                        min=0, max=1000, step=10,
                        value=[0, 1000],
                        className='range-slider-container',
                        marks=None,
                    ),
                    html.Div(id='slider-power-labels', className='range-labels'),
                ], className='form-group'),
                html.Div([
                    html.Label('Delay Range (ps)', className='form-label'),
                    dcc.RangeSlider(
                        id='slider-delay',
                        min=0, max=1000, step=10,
                        value=[0, 1000],
                        className='range-slider-container',
                        marks=None,
                    ),
                    html.Div(id='slider-delay-labels', className='range-labels'),
                ], className='form-group'),
            ], className='sidebar-section'),

            # Visualization Settings
            html.Div([
                html.Span('Visualization', className='sidebar-title'),
            ], className='sidebar-section'),

            html.Div([
                html.Div([
                    html.Label('Chart Type', className='form-label'),
                    dcc.Dropdown(
                        id='input-chart-type',
                        options=[
                            {'label': '2D Scatter', 'value': '2d'},
                            {'label': '3D Scatter', 'value': '3d'},
                            {'label': 'Parallel Coordinates', 'value': 'parallel'},
                        ],
                        value='2d',
                        className='form-select',
                        style={'background': 'var(--bg-input)'}
                    ),
                ], className='form-group'),
                html.Div([
                    html.Label('Color By', className='form-label'),
                    dcc.Dropdown(
                        id='input-color-by',
                        options=[
                            {'label': 'Area', 'value': 'Area'},
                            {'label': 'Power', 'value': 'Power'},
                            {'label': 'Delay', 'value': 'Delay'},
                        ],
                        value='Delay',
                        className='form-select',
                        style={'background': 'var(--bg-input)'}
                    ),
                ], className='form-group'),
            ], className='sidebar-section'),

            # Stats Summary
            html.Div([
                html.Span('Summary', className='sidebar-title'),
            ], className='sidebar-section'),

            html.Div([
                html.Div([html.Span('Total Solutions', className='form-label'),
                         html.Span(f"{len(self.pareto_df)}", style={'fontFamily': 'var(--font-data)', 'fontSize': '18px', 'fontWeight': '700'})],
                        style={'display': 'flex', 'flexDirection': 'column', 'gap': '2px'}),
            ], className='sidebar-section'),
        ], className='sidebar')

    def _create_right_panel(self) -> html.Div:
        """Create right panel with visualizations."""
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
            # Top metrics bar
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

            # Tabs for different views
            dcc.Tabs(id='tabs', value='tab-overview', children=[
                dcc.Tab(label='Overview', value='tab-overview', children=self._create_overview_tab()),
                dcc.Tab(label='3D View', value='tab-3d', children=self._create_3d_tab()),
                dcc.Tab(label='Convergence', value='tab-conv', children=self._create_convergence_tab()),
                dcc.Tab(label='Solutions', value='tab-solutions', children=self._create_solutions_tab()),
            ], className='tabs-container'),
        ], className='right-panel')

    def _create_overview_tab(self) -> html.Div:
        """Create overview tab with 2D scatter plots."""
        obj = self.pareto_df[['Area', 'Power', 'Delay']].values

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Area vs Power', 'Area vs Delay', 'Power vs Delay'],
            horizontal_spacing=0.08,
        )

        colors = obj[:, 2]  # Delay

        # Area vs Power
        fig.add_trace(go.Scatter(
            x=obj[:, 0], y=obj[:, 1],
            mode='markers',
            marker=dict(size=8, color=colors, colorscale='Viridis', showscale=False,
                      line=dict(width=0.5, color='rgba(255,255,255,0.3)')),
            hovertemplate='Area: %{x:.3f}<br>Power: %{y:.2f}<extra></extra>',
        ), row=1, col=1)

        # Area vs Delay
        fig.add_trace(go.Scatter(
            x=obj[:, 0], y=obj[:, 2],
            mode='markers',
            marker=dict(size=8, color=colors, colorscale='Viridis', showscale=False,
                      line=dict(width=0.5, color='rgba(255,255,255,0.3)')),
            hovertemplate='Area: %{x:.3f}<br>Delay: %{y:.2f}<extra></extra>',
        ), row=1, col=2)

        # Power vs Delay
        fig.add_trace(go.Scatter(
            x=obj[:, 1], y=obj[:, 2],
            mode='markers',
            marker=dict(size=8, color=obj[:, 0], colorscale='Plasma', showscale=False,
                      line=dict(width=0.5, color='rgba(255,255,255,0.3)')),
            hovertemplate='Power: %{x:.2f}<br>Delay: %{y:.2f}<extra></extra>',
        ), row=1, col=3)

        fig.update_layout(
            title='Pareto Front Overview',
            height=400,
            showlegend=False,
            title_font_size=13,
            title_x=0.5,
            paper_bgcolor='var(--graph-bg)',
            plot_bgcolor='var(--graph-bg)',
            margin=dict(l=40, r=40, t=50, b=40),
        )

        for i in range(1, 4):
            fig.update_xaxes(title_text=['Area (um²)', 'Area (um²)', 'Power (uW)'][i-1],
                          gridcolor='var(--border-secondary)', row=1, col=i)
            fig.update_yaxes(title_text=['Power (uW)', 'Delay (ps)', 'Delay (ps)'][i-1],
                          gridcolor='var(--border-secondary)', row=1, col=i)

        return html.Div([
            dcc.Graph(figure=fig, className='chart-container'),
        ])

    def _create_3d_tab(self) -> html.Div:
        """Create 3D visualization tab."""
        obj = self.pareto_df[['Area', 'Power', 'Delay']].values

        fig = go.Figure(data=[go.Scatter3d(
            x=obj[:, 0], y=obj[:, 1], z=obj[:, 2],
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
            title='3D Pareto Front',
            scene=dict(
                xaxis_title='Area (um²)',
                yaxis_title='Power (uW)',
                zaxis_title='Delay (ps)',
                xaxis=dict(backgroundcolor='var(--graph-bg)', gridcolor='var(--border-secondary)'),
                yaxis=dict(backgroundcolor='var(--graph-bg)', gridcolor='var(--border-secondary)'),
                zaxis=dict(backgroundcolor='var(--graph-bg)', gridcolor='var(--border-secondary)'),
            ),
            height=550,
            paper_bgcolor='var(--graph-bg)',
            margin=dict(l=0, r=0, t=50, b=0),
            title_font_size=13,
            title_x=0.5,
        )

        return html.Div([dcc.Graph(figure=fig, className='chart-container')])

    def _create_convergence_tab(self) -> html.Div:
        """Create convergence analysis tab."""
        conv_df = self.convergence_data

        if len(conv_df) == 0:
            return html.Div([html.Div('No convergence data available', className='empty-state')])

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Pareto Front Size', 'Average Fitness'],
            vertical_spacing=0.15,
        )

        # Front 0 size
        fig.add_trace(go.Scatter(
            x=conv_df['gen'],
            y=conv_df.get('front0_size', [0]*len(conv_df)),
            mode='lines+markers',
            name='Front 0',
            line=dict(color='#3fb950', width=2),
            marker=dict(size=6),
        ), row=1, col=1)

        # Average fitness
        if 'avg_fitness' in conv_df.columns:
            avg_fit = conv_df['avg_fitness'].tolist()
            if avg_fit and isinstance(avg_fit[0], list):
                obj_labels = ['Area', 'Power', 'Delay']
                colors = ['#58a6ff', '#3fb950', '#f0883e']
                for i, (label, color) in enumerate(zip(obj_labels, colors)):
                    fig.add_trace(go.Scatter(
                        x=conv_df['gen'],
                        y=[f[i] if len(f) > i else 0 for f in avg_fit],
                        mode='lines',
                        name=f'Avg {label}',
                        line=dict(color=color, width=1.5, dash='dash'),
                    ), row=2, col=1)

        fig.update_layout(
            height=450,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            paper_bgcolor='var(--graph-bg)',
            plot_bgcolor='var(--graph-bg)',
            title_font_size=13,
            title_x=0.5,
            margin=dict(l=50, r=50, t=50, b=40),
        )

        return html.Div([dcc.Graph(figure=fig, className='chart-container')])

    def _create_solutions_tab(self) -> html.Div:
        """Create solutions table tab."""
        display_df = self.pareto_df.copy()
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
                    'fontFamily': 'var(--font-data)',
                    'fontSize': '12px',
                    'padding': '8px 12px',
                    'backgroundColor': 'var(--bg-secondary)',
                    'color': 'var(--text-primary)',
                },
                style_header={
                    'backgroundColor': 'var(--bg-tertiary)',
                    'fontWeight': '600',
                    'textTransform': 'uppercase',
                    'fontSize': '10px',
                    'letterSpacing': '0.05em',
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': 'var(--bg-primary)'},
                ],
                style_table={'overflowX': 'auto'},
            ),
        ], className='data-table-wrapper')

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
            specs=[
                [{'type': 'scatter3d'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ],
            subplot_titles=['3D Pareto', 'Area vs Power', 'Area vs Delay', 'Power vs Delay'],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
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
            height=700,
            showlegend=False,
        )

        fig.write_html(str(path), include_plotlyjs='cdn', full_html=True)
        return path


def create_dashboard(results: dict, config: dict, output_dir: Path) -> Dashboard:
    """Factory function to create dashboard."""
    return Dashboard(results, config, output_dir)

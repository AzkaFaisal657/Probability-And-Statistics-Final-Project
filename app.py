from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy.stats as scipy_stats
from scipy.stats import binom, poisson, norm
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ─── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv('student_entrepreneurial_projects.csv')

# Pre-train multiple regression model
FEATURES = ['technical_skill_score', 'business_skill_score',
            'avg_team_experience', 'team_size',
            'market_growth_rate', 'competition_level']
X_all = df[FEATURES]
y_all = df['competitiveness_score']
reg_model = LinearRegression().fit(X_all, y_all)

NUMERIC_VARS = [
    'competitiveness_score', 'market_size_usd', 'market_growth_rate',
    'team_size', 'technical_skill_score', 'business_skill_score',
    'social_media_mentions', 'mentor_feedback_score',
    'avg_team_experience', 'competition_level'
]
NUMERIC_LABELS = {
    'competitiveness_score': 'Competitiveness Score',
    'market_size_usd': 'Market Size (USD)',
    'market_growth_rate': 'Market Growth Rate (%)',
    'team_size': 'Team Size',
    'technical_skill_score': 'Technical Skill Score',
    'business_skill_score': 'Business Skill Score',
    'social_media_mentions': 'Social Media Mentions',
    'mentor_feedback_score': 'Mentor Feedback Score',
    'avg_team_experience': 'Avg Team Experience',
    'competition_level': 'Competition Level',
}
CAT_VARS = ['project_domain', 'funding_stage', 'innovation_type', 'education_level']
CAT_LABELS = {
    'project_domain': 'Project Domain',
    'funding_stage': 'Funding Stage',
    'innovation_type': 'Innovation Type',
    'education_level': 'Education Level',
}

# ─── Color Palette ────────────────────────────────────────────────────────────
BG            = '#ffffff'
CARD_BG       = '#ffffff'
PANEL_BG      = '#f9f9f9'
HEADER_BG     = '#7c3f6e'
TEXT          = '#1a1a2e'
TEXT_SEC      = '#555555'
TEXT_ITALIC   = '#888888'
TEXT_WHITE    = '#ffffff'
PRIMARY       = '#8b4f7a'
PRIMARY_LIGHT = '#c084a0'
GREEN         = '#52b788'
RED           = '#e05c6a'
AMBER         = '#f5a623'
BORDER        = '#e8e8e8'
SHADOW        = '0 2px 8px rgba(0,0,0,0.08)'
CHART_COLORS  = ['#8b4f7a', '#52b788', '#e05c6a', '#f5a623', '#6baed6', '#c084a0', '#74c476']

CHART_LAYOUT = dict(
    template='plotly_white',
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff',
    font=dict(color='#333333', family='Inter, Segoe UI, sans-serif'),
    title_font=dict(size=15, color='#1a1a2e'),
    margin=dict(t=50, l=50, r=30, b=50),
)

# ─── App Init ─────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ─── Custom CSS ───────────────────────────────────────────────────────────────
app.index_string = '''<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Student Entrepreneurial Projects | P&S Dashboard</title>
    {%favicon%}
    {%css%}
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body { font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif; background: #ffffff; color: #1a1a2e; min-height: 100vh; }
      .tab--selected { border-bottom: 3px solid #7c3f6e !important; color: #7c3f6e !important; font-weight: 600 !important; background: transparent !important; }
      .tab { color: #555555 !important; background: transparent !important; border: none !important; border-bottom: 3px solid transparent !important; padding: 12px 20px !important; font-size: 14px !important; cursor: pointer !important; font-family: 'Inter', 'Segoe UI', sans-serif !important; }
      .tab:hover { color: #7c3f6e !important; }
      .Select-value { background: #f0f0f0 !important; border: 1px solid #e0e0e0 !important; color: #333 !important; border-radius: 4px !important; }
      ::-webkit-scrollbar { width: 6px; }
      ::-webkit-scrollbar-track { background: #f9f9f9; }
      ::-webkit-scrollbar-thumb { background: #c084a0; border-radius: 3px; }
      .stat-card { transition: all 0.2s ease; cursor: default; }
      .stat-card:hover { box-shadow: 0 4px 16px rgba(139, 79, 122, 0.15) !important; transform: translateY(-1px); }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>'''

# ─── Helpers ──────────────────────────────────────────────────────────────────
def card(children, style=None):
    base = {'background': CARD_BG, 'border': f'1px solid {BORDER}', 'borderRadius': '8px',
            'boxShadow': SHADOW, 'padding': '20px', 'marginBottom': '20px'}
    if style:
        base.update(style)
    return html.Div(children, style=base)

def nugget(text):
    return html.P(text, style={'color': TEXT_ITALIC, 'fontStyle': 'italic', 'fontSize': '13px', 'marginBottom': '12px'})

def section_label(text):
    return html.P(text, style={'color': TEXT_SEC, 'fontSize': '13px', 'fontWeight': '600',
                               'marginBottom': '6px', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'})

def metric_card(title, value):
    return html.Div([
        html.Div([
            html.Div([
                html.Div(str(value), style={'fontSize': '26px', 'fontWeight': '700', 'color': TEXT, 'lineHeight': '1', 'marginBottom': '4px'}),
                html.Div(title, style={'fontSize': '12px', 'color': TEXT_SEC, 'fontWeight': '500'}),
            ], style={'flex': '1'}),
            html.Div(style={'width': '8px', 'borderRadius': '4px', 'background': PRIMARY_LIGHT, 'minHeight': '50px', 'alignSelf': 'stretch'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '16px'}),
    ], className='stat-card', style={'background': CARD_BG, 'border': f'1px solid {BORDER}', 'borderRadius': '8px', 'boxShadow': SHADOW, 'padding': '16px 18px'})

def small_metric_card(title, value, color=PRIMARY):
    return html.Div([
        html.Div(str(value), style={'fontSize': '20px', 'fontWeight': '700', 'color': color, 'lineHeight': '1', 'marginBottom': '2px'}),
        html.Div(title, style={'fontSize': '11px', 'color': TEXT_SEC, 'fontWeight': '500'}),
    ], style={'background': CARD_BG, 'border': f'1px solid {BORDER}', 'borderRadius': '8px',
              'boxShadow': SHADOW, 'padding': '14px 16px', 'flex': '1', 'minWidth': '100px'})

def freq_table_html(headers, rows):
    header_cells = [html.Th(h, style={'background': PRIMARY, 'color': '#fff', 'padding': '8px 12px',
                                      'textAlign': 'left', 'fontSize': '12px', 'fontWeight': '600',
                                      'border': f'1px solid {BORDER}'}) for h in headers]
    data_rows = []
    for i, row in enumerate(rows):
        bg = '#f9f9f9' if i % 2 == 0 else '#ffffff'
        cells = [html.Td(cell, style={'padding': '7px 12px', 'fontSize': '12px',
                                      'border': f'1px solid {BORDER}', 'background': bg, 'color': TEXT}) for cell in row]
        data_rows.append(html.Tr(cells))
    return html.Table([html.Thead(html.Tr(header_cells)), html.Tbody(data_rows)],
                      style={'width': '100%', 'borderCollapse': 'collapse'})

def skew_badge(skew_val):
    abs_skew = abs(skew_val)
    if abs_skew < 0.5:
        label, color = 'Approximately Symmetric', GREEN
    elif abs_skew < 1.0:
        label, color = ('Positive Skew' if skew_val > 0 else 'Negative Skew'), AMBER
    else:
        label, color = ('Positive Skew' if skew_val > 0 else 'Negative Skew'), RED
    return html.Span(label, style={'background': color, 'color': '#fff', 'borderRadius': '20px',
                                   'padding': '3px 10px', 'fontSize': '11px', 'fontWeight': '600'})

def filter_row(*items):
    children = []
    for lbl, dd in items:
        children.append(html.Div([section_label(lbl), dd], style={'flex': '1', 'minWidth': '180px'}))
    return html.Div(children, style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap',
                                     'marginBottom': '16px', 'background': PANEL_BG, 'borderRadius': '8px',
                                     'padding': '14px 16px', 'border': f'1px solid {BORDER}'})

def domain_dd(id_):
    opts = sorted(df['project_domain'].unique())
    return dcc.Dropdown(id=id_, options=[{'label': o, 'value': o} for o in opts],
                        value=opts, multi=True, style={'fontSize': '13px'})

def stage_dd(id_):
    opts = sorted(df['funding_stage'].unique())
    return dcc.Dropdown(id=id_, options=[{'label': o, 'value': o} for o in opts],
                        value=opts, multi=True, style={'fontSize': '13px'})

def innovation_dd(id_):
    opts = sorted(df['innovation_type'].unique())
    return dcc.Dropdown(id=id_, options=[{'label': o, 'value': o} for o in opts],
                        value=opts, multi=True, style={'fontSize': '13px'})

# ─── Layout ───────────────────────────────────────────────────────────────────
app.layout = html.Div([

    html.Div([
        html.Div([
            html.Div([
                html.H1('Student Entrepreneurial Projects Analysis',
                        style={'color': TEXT_WHITE, 'fontWeight': '700', 'fontSize': '36px',
                               'marginBottom': '10px', 'lineHeight': '1.2'}),
                html.P('Comprehensive interactive dashboard exploring student entrepreneurial competitiveness patterns. '
                       'This multi-angle analysis reveals how team composition, market dynamics, skill levels, '
                       'funding stages, and innovation strategies influence project success enabling data-driven '
                       'insights through diverse statistical and probabilistic perspectives.',
                       style={'color': 'rgba(255,255,255,0.82)', 'fontSize': '14px',
                              'lineHeight': '1.6', 'marginBottom': '18px', 'maxWidth': '780px'}),
                html.Div([
                    html.Span([
                        html.Span(' ', style={'marginRight': '5px'}),
                        html.Span('Submitted: ', style={'fontWeight': '600'}),
                        'May 2026'
                    ], style={'background': 'rgba(255,255,255,0.18)', 'color': TEXT_WHITE,
                        'borderRadius': '6px', 'padding': '5px 14px', 'fontSize': '12px',
                        'marginRight': '10px', 'display': 'inline-flex', 'alignItems': 'center'}),
                    html.Span([
                        html.Span(' ', style={'marginRight': '5px'}),
                        html.Span('Team Leader: ', style={'fontWeight': '600'}),
                        'Azka Faisal'
                    ], style={'background': 'rgba(255,255,255,0.18)', 'color': TEXT_WHITE,
                        'borderRadius': '6px', 'padding': '5px 14px', 'fontSize': '12px',
                        'marginRight': '10px', 'display': 'inline-flex', 'alignItems': 'center'}),
                    html.Span([
                        html.Span(' ', style={'marginRight': '5px'}),
                        html.Span('Data Source: ', style={'fontWeight': '600'}),
                        'Student Entrepreneurial Projects (5,000 rows)'
                    ], style={'background': 'rgba(255,255,255,0.18)', 'color': TEXT_WHITE,
                        'borderRadius': '6px', 'padding': '5px 14px', 'fontSize': '12px',
                        'display': 'inline-flex', 'alignItems': 'center'}),
                ]),
            ], style={'maxWidth': '960px'}),
        ], style={'padding': '40px 40px 36px 40px'}),
    ], style={'background': HEADER_BG, 'width': '100%'}),

    html.Div([
        dcc.Tabs(id='main-tabs', value='tab-1', children=[

            # ── TAB 1: OVERVIEW ──────────────────────────────────────────────
            dcc.Tab(label='Overview Dashboard', value='tab-1', className='tab', selected_className='tab--selected', children=[
                html.Div([
                    filter_row(('Domain', domain_dd('t1-domain')), ('Funding Stage', stage_dd('t1-stage')), ('Innovation Type', innovation_dd('t1-innovation'))),
                    html.P(id='t1-row-count', style={'fontStyle': 'italic', 'color': TEXT_ITALIC, 'fontSize': '13px', 'marginBottom': '14px'}),
                    html.Div(id='t1-metrics', style={'display': 'flex', 'gap': '14px', 'marginBottom': '20px', 'flexWrap': 'wrap'}),
                    html.Div([
                        html.Div([card([nugget('NUGGET: Which domain attracts the most projects? Does any domain dominate the entrepreneurial landscape?'),
                                        dcc.Graph(id='t1-pie', config={'displayModeBar': False}, style={'height': '320px'})],
                                       style={'marginBottom': '0', 'height': '100%'})],
                                 style={'flex': '1', 'minWidth': '320px'}),
                        html.Div([card([nugget('NUGGET: Which funding stage produces the most competitive projects? Look for median differences and spread.'),
                                        dcc.Graph(id='t1-box', config={'displayModeBar': False}, style={'height': '320px'})],
                                       style={'marginBottom': '0', 'height': '100%'})],
                                 style={'flex': '1', 'minWidth': '320px'}),
                    ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px', 'alignItems': 'stretch'}),
                    html.Div([
                        html.Div([card([nugget('NUGGET: Is there a dominant innovation strategy? Disruptive, incremental or radical — which prevails?'),
                                        dcc.Graph(id='t1-bar', config={'displayModeBar': False}, style={'height': '320px'})],
                                       style={'marginBottom': '0', 'height': '100%'})],
                                 style={'flex': '1', 'minWidth': '320px'}),
                        html.Div([card([nugget('NUGGET: What is the shape of the competitiveness score distribution? Is it normal, skewed, or uniform?'),
                                        dcc.Graph(id='t1-hist', config={'displayModeBar': False}, style={'height': '320px'})],
                                       style={'marginBottom': '0', 'height': '100%'})],
                                 style={'flex': '1', 'minWidth': '320px'}),
                    ], style={'display': 'flex', 'gap': '16px', 'alignItems': 'stretch'}),
                ], style={'padding': '20px'}),
            ]),

            # ── TAB 2: FREQUENCY & DISTRIBUTIONS ────────────────────────────
            dcc.Tab(label='Frequency & Distributions', value='tab-2', className='tab', selected_className='tab--selected', children=[
                html.Div([
                    filter_row(('Domain', domain_dd('t2-domain')), ('Funding Stage', stage_dd('t2-stage'))),
                    card([
                        html.H3('Section A — Quantitative Frequency Distribution', style={'fontSize': '15px', 'color': TEXT, 'marginBottom': '4px'}),
                        nugget('NUGGET: How is the selected numeric variable distributed across the dataset? What does its shape tell us?'),
                        html.Div([section_label('Select Numeric Variable'),
                                  dcc.Dropdown(id='t2-num-var',
                                               options=[{'label': NUMERIC_LABELS[v], 'value': v} for v in NUMERIC_VARS],
                                               value='competitiveness_score', clearable=False,
                                               style={'fontSize': '13px', 'marginBottom': '16px'})]),
                        html.Div([
                            html.Div([dcc.Graph(id='t2-num-hist', config={'displayModeBar': False}, style={'height': '320px'})],
                                     style={'flex': '65', 'minWidth': '280px'}),
                            html.Div([html.Div(id='t2-freq-table')],
                                     style={'flex': '35', 'minWidth': '220px', 'overflowX': 'auto'}),
                        ], style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start'}),
                    ]),
                    card([
                        html.H3('Section B — Qualitative Frequency Distribution', style={'fontSize': '15px', 'color': TEXT, 'marginBottom': '4px'}),
                        nugget('NUGGET: Which category dominates? What proportion does each group represent?'),
                        html.Div([section_label('Select Categorical Variable'),
                                  dcc.Dropdown(id='t2-cat-var',
                                               options=[{'label': CAT_LABELS[v], 'value': v} for v in CAT_VARS],
                                               value='project_domain', clearable=False,
                                               style={'fontSize': '13px', 'marginBottom': '16px'})]),
                        html.Div([
                            html.Div([dcc.Graph(id='t2-cat-bar', config={'displayModeBar': False}, style={'height': '280px'})],
                                     style={'flex': '1', 'minWidth': '280px'}),
                            html.Div([dcc.Graph(id='t2-cat-pie', config={'displayModeBar': False}, style={'height': '280px'})],
                                     style={'flex': '1', 'minWidth': '280px'}),
                        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px'}),
                        html.Div(id='t2-cat-table'),
                    ]),
                ], style={'padding': '20px'}),
            ]),

            # ── TAB 3: EDA ───────────────────────────────────────────────────
            dcc.Tab(label='EDA & Shape of Data', value='tab-3', className='tab', selected_className='tab--selected', children=[
                html.Div([
                    html.Div([section_label('Select Numeric Variable'),
                              dcc.Dropdown(id='t3-var',
                                           options=[{'label': NUMERIC_LABELS[v], 'value': v} for v in NUMERIC_VARS],
                                           value='competitiveness_score', clearable=False,
                                           style={'fontSize': '13px'})],
                             style={'background': PANEL_BG, 'borderRadius': '8px', 'padding': '14px 16px',
                                    'marginBottom': '16px', 'border': f'1px solid {BORDER}', 'maxWidth': '400px'}),
                    card([
                        html.H3('Section A — Box Plot + Statistical Summary', style={'fontSize': '15px', 'color': TEXT, 'marginBottom': '4px'}),
                        nugget('NUGGET: Where are the outliers? Is the data symmetric or skewed? What do Q1, Q3 and the fences tell us?'),
                        html.Div([
                            html.Div([dcc.Graph(id='t3-boxplot', config={'displayModeBar': False}, style={'height': '320px'})],
                                     style={'flex': '60', 'minWidth': '280px'}),
                            html.Div([html.Div(id='t3-stats-table')], style={'flex': '40', 'minWidth': '240px'}),
                        ], style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start'}),
                    ]),
                    card([
                        html.H3('Section B — Histogram with Mean/Median Lines', style={'fontSize': '15px', 'color': TEXT, 'marginBottom': '4px'}),
                        nugget('NUGGET: Does the histogram confirm the skewness direction? Where do mean and median fall relative to each other?'),
                        dcc.Graph(id='t3-meanmedian', config={'displayModeBar': False}, style={'height': '320px'}),
                    ]),
                    card([
                        html.H3('Section C — Grouped Box Plot by Education Level', style={'fontSize': '15px', 'color': TEXT, 'marginBottom': '4px'}),
                        nugget('NUGGET: Does education level influence competitiveness? Compare medians and spreads across groups.'),
                        dcc.Graph(id='t3-edbox', config={'displayModeBar': False}, style={'height': '320px'}),
                    ]),
                    card([
                        html.H3('Section E — Coefficient of Variation', style={'fontSize': '15px', 'color': TEXT, 'marginBottom': '4px'}),
                        nugget('NUGGET: Which variable has the highest relative variability? CV converts SD into percentage form, allowing comparison across different units.'),
                        dcc.Graph(id='t3-cv', config={'displayModeBar': False}, style={'height': '360px'}),
                        html.P('CV is used to compare two datasets with different units. Higher CV = more relative variation.',
                               style={'fontSize': '12px', 'color': TEXT_ITALIC, 'fontStyle': 'italic', 'marginTop': '8px'}),
                    ]),
                ], style={'padding': '20px'}),
            ]),

            # ── TAB 4: PROBABILITY ───────────────────────────────────────────
            dcc.Tab(label='Probability Distributions', value='tab-4', className='tab', selected_className='tab--selected', children=[
                html.Div([
                    card([
                        html.H3('Section A — Binomial Distribution', style={'fontSize': '15px', 'color': TEXT, 'marginBottom': '4px'}),
                        nugget('NUGGET: Given n trials and probability p, what is the chance of exactly k successes? Binomial has two parameters: n and p.'),
                        html.Div([
                            html.Div([section_label('n (trials)'),
                                      dcc.Input(id='binom-n', type='number', value=20, min=1, max=100, step=1,
                                               style={'width': '100%', 'padding': '8px', 'borderRadius': '6px', 'border': f'1px solid {BORDER}', 'fontSize': '14px'})],
                                     style={'flex': '1'}),
                            html.Div([section_label('p (success probability)'),
                                      dcc.Input(id='binom-p', type='number', value=0.3, min=0.01, max=0.99, step=0.01,
                                               style={'width': '100%', 'padding': '8px', 'borderRadius': '6px', 'border': f'1px solid {BORDER}', 'fontSize': '14px'})],
                                     style={'flex': '1'}),
                            html.Div([section_label('k (successes)'),
                                      dcc.Slider(id='binom-k', min=0, max=20, step=1, value=6,
                                                marks=None, tooltip={"placement": "bottom", "always_visible": True})],
                                     style={'flex': '2', 'paddingTop': '4px'}),
                        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '16px', 'alignItems': 'flex-end', 'flexWrap': 'wrap'}),
                        dcc.Graph(id='binom-chart', config={'displayModeBar': False}, style={'height': '280px'}),
                        html.Div(id='binom-metrics', style={'display': 'flex', 'gap': '10px', 'marginTop': '14px', 'flexWrap': 'wrap'}),
                        html.P('P(X=k) = C(n,k) · pᵏ · (1-p)ⁿ⁻ᵏ',
                               style={'fontStyle': 'italic', 'color': TEXT_ITALIC, 'fontSize': '12px', 'marginTop': '10px'}),
                    ]),
                    card([
                        html.H3('Section B — Poisson Distribution', style={'fontSize': '15px', 'color': TEXT, 'marginBottom': '4px'}),
                        nugget('NUGGET: Poisson models the number of events in a fixed interval. It has one parameter: λ (average rate). Mean = Variance = λ.'),
                        html.Div([
                            html.Div([section_label('λ (lambda — average rate)'),
                                      dcc.Input(id='pois-lam', type='number', value=3.0, min=0.1, max=20, step=0.1,
                                               style={'width': '100%', 'padding': '8px', 'borderRadius': '6px', 'border': f'1px solid {BORDER}', 'fontSize': '14px'})],
                                     style={'flex': '1', 'maxWidth': '200px'}),
                            html.Div([section_label('k (number of events)'),
                                      dcc.Slider(id='pois-k', min=0, max=20, step=1, value=3,
                                                marks=None, tooltip={"placement": "bottom", "always_visible": True})],
                                     style={'flex': '2', 'paddingTop': '4px'}),
                        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '16px', 'alignItems': 'flex-end', 'flexWrap': 'wrap'}),
                        dcc.Graph(id='pois-chart', config={'displayModeBar': False}, style={'height': '280px'}),
                        html.Div(id='pois-metrics', style={'display': 'flex', 'gap': '10px', 'marginTop': '14px', 'flexWrap': 'wrap'}),
                        html.P('P(X=k) = (e⁻λ · λᵏ) / k!',
                               style={'fontStyle': 'italic', 'color': TEXT_ITALIC, 'fontSize': '12px', 'marginTop': '10px'}),
                    ]),
                    card([
                        html.H3('Section C — Normal Distribution', style={'fontSize': '15px', 'color': TEXT, 'marginBottom': '4px'}),
                        nugget('NUGGET: The normal distribution is symmetric and bell-shaped. Z-score tells you how many standard deviations x is from the mean.'),
                        html.Div([
                            html.Div([section_label('μ (mean)'),
                                      dcc.Input(id='norm-mu', type='number',
                                               value=round(float(df['competitiveness_score'].mean()), 2), step=0.01,
                                               style={'width': '100%', 'padding': '8px', 'borderRadius': '6px', 'border': f'1px solid {BORDER}', 'fontSize': '14px'})],
                                     style={'flex': '1', 'maxWidth': '200px'}),
                            html.Div([section_label('σ (std dev)'),
                                      dcc.Input(id='norm-sigma', type='number',
                                               value=round(float(df['competitiveness_score'].std()), 2), min=0.01, step=0.01,
                                               style={'width': '100%', 'padding': '8px', 'borderRadius': '6px', 'border': f'1px solid {BORDER}', 'fontSize': '14px'})],
                                     style={'flex': '1', 'maxWidth': '200px'}),
                            html.Div([section_label('x value'),
                                      dcc.Slider(id='norm-x', min=0, max=50, step=0.5, value=25,
                                                marks=None, tooltip={"placement": "bottom", "always_visible": True})],
                                     style={'flex': '2', 'paddingTop': '4px'}),
                        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '16px', 'alignItems': 'flex-end', 'flexWrap': 'wrap'}),
                        dcc.Graph(id='norm-chart', config={'displayModeBar': False}, style={'height': '320px'}),
                        html.Div(id='norm-metrics', style={'display': 'flex', 'gap': '10px', 'marginTop': '14px', 'flexWrap': 'wrap'}),
                        html.P('Z = (X − μ) / σ',
                               style={'fontStyle': 'italic', 'color': TEXT_ITALIC, 'fontSize': '12px', 'marginTop': '10px'}),
                    ]),
                ], style={'padding': '20px'}),
            ]),

            # ── TAB 5: REGRESSION ────────────────────────────────────────────
            dcc.Tab(label='Regression & Predictions', value='tab-5', className='tab', selected_className='tab--selected', children=[
                html.Div([
                    card([
                        html.H3('Section A — Simple Linear Regression Explorer', style={'fontSize': '15px', 'color': TEXT, 'marginBottom': '4px'}),
                        nugget('NUGGET: Which two variables have the strongest linear relationship? The regression line shows the best fit, and R² tells you how well it fits.'),
                        html.Div([
                            html.Div([section_label('X Variable'),
                                      dcc.Dropdown(id='reg-x',
                                                   options=[{'label': NUMERIC_LABELS[v], 'value': v} for v in NUMERIC_VARS],
                                                   value='technical_skill_score', clearable=False, style={'fontSize': '13px'})],
                                     style={'flex': '1', 'minWidth': '200px'}),
                            html.Div([section_label('Y Variable'),
                                      dcc.Dropdown(id='reg-y',
                                                   options=[{'label': NUMERIC_LABELS[v], 'value': v} for v in NUMERIC_VARS],
                                                   value='competitiveness_score', clearable=False, style={'fontSize': '13px'})],
                                     style={'flex': '1', 'minWidth': '200px'}),
                        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px', 'flexWrap': 'wrap'}),
                        html.Div([
                            html.Div([dcc.Graph(id='reg-scatter', config={'displayModeBar': False}, style={'height': '360px'})],
                                     style={'flex': '65', 'minWidth': '280px'}),
                            html.Div([html.Div(id='reg-stats')], style={'flex': '35', 'minWidth': '220px'}),
                        ], style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start'}),
                    ]),
                    card([
                        html.H3('Section B — Multiple Regression Predictor', style={'fontSize': '15px', 'color': TEXT, 'marginBottom': '4px'}),
                        nugget('NUGGET: Using 6 input features, the model predicts competitiveness score. Move the sliders and see how each feature affects the prediction.'),
                        html.Div([
                            html.Div([
                                html.Div([section_label('Technical Skill Score (0–10)'),
                                          dcc.Slider(0, 10, 1, value=5, id='sl-tech',
                                                    marks={i: str(i) for i in range(0, 11, 2)},
                                                    tooltip={"placement": "bottom", "always_visible": True})], style={'marginBottom': '24px'}),
                                html.Div([section_label('Business Skill Score (0–10)'),
                                          dcc.Slider(0, 10, 1, value=5, id='sl-bus',
                                                    marks={i: str(i) for i in range(0, 11, 2)},
                                                    tooltip={"placement": "bottom", "always_visible": True})], style={'marginBottom': '24px'}),
                                html.Div([section_label('Avg Team Experience (0–5 yrs)'),
                                          dcc.Slider(0, 5, 0.5, value=2.5, id='sl-exp',
                                                    marks={i: str(i) for i in range(0, 6)},
                                                    tooltip={"placement": "bottom", "always_visible": True})], style={'marginBottom': '24px'}),
                                html.Div([section_label('Team Size (1–6)'),
                                          dcc.Slider(1, 6, 1, value=3, id='sl-team',
                                                    marks={i: str(i) for i in range(1, 7)},
                                                    tooltip={"placement": "bottom", "always_visible": True})], style={'marginBottom': '24px'}),
                                html.Div([section_label('Market Growth Rate (0–25%)'),
                                          dcc.Slider(0, 25, 0.5, value=12.5, id='sl-growth',
                                                    marks={i: str(i) for i in range(0, 26, 5)},
                                                    tooltip={"placement": "bottom", "always_visible": True})], style={'marginBottom': '24px'}),
                                html.Div([section_label('Competition Level (0–10)'),
                                          dcc.Slider(0, 10, 1, value=5, id='sl-comp',
                                                    marks={i: str(i) for i in range(0, 11, 2)},
                                                    tooltip={"placement": "bottom", "always_visible": True})]),
                            ], style={'flex': '40', 'minWidth': '260px', 'background': PANEL_BG,
                                      'borderRadius': '8px', 'padding': '20px', 'border': f'1px solid {BORDER}'}),
                            html.Div([
                                html.Div([
                                    html.P('Predicted Competitiveness Score', style={'color': 'rgba(255,255,255,0.8)',
                                        'fontSize': '12px', 'fontWeight': '500', 'marginBottom': '8px', 'textAlign': 'center'}),
                                    html.Div(id='pred-output', style={'fontSize': '42px', 'fontWeight': '700',
                                        'color': TEXT_WHITE, 'textAlign': 'center'}),
                                    html.P('/ 50.00', style={'textAlign': 'center', 'color': 'rgba(255,255,255,0.6)',
                                        'fontSize': '14px', 'marginTop': '4px'}),
                                ], style={'background': f'linear-gradient(135deg, {PRIMARY} 0%, {HEADER_BG} 100%)',
                                          'borderRadius': '12px', 'padding': '30px 20px', 'marginBottom': '16px'}),
                                dcc.Graph(id='pred-scatter', config={'displayModeBar': False}, style={'height': '280px'}),
                            ], style={'flex': '60', 'minWidth': '280px'}),
                        ], style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start', 'flexWrap': 'wrap'}),
                    ]),
                ], style={'padding': '20px'}),
            ]),

        ], style={'borderBottom': f'1px solid {BORDER}'}),
    ], style={'padding': '0 40px'}),

    html.Div([
        html.P('Probability & Statistics | FAST-NUCES | Spring 2026 | Interactive Data Analysis Dashboard',
               style={'textAlign': 'center', 'color': TEXT_SEC, 'margin': '0', 'fontSize': '13px'}),
    ], style={'background': PANEL_BG, 'padding': '18px 24px', 'borderTop': f'1px solid {BORDER}', 'marginTop': '20px'}),

], style={'fontFamily': "'Inter', 'Segoe UI', sans-serif", 'background': BG, 'minHeight': '100vh'})


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — TAB 1
# ═══════════════════════════════════════════════════════════════════════════════

def filter_df(domains, stages, innovations=None):
    mask = pd.Series([True] * len(df))
    if domains:
        mask &= df['project_domain'].isin(domains)
    if stages:
        mask &= df['funding_stage'].isin(stages)
    if innovations:
        mask &= df['innovation_type'].isin(innovations)
    return df[mask]

@app.callback(
    Output('t1-row-count', 'children'),
    Output('t1-metrics', 'children'),
    Output('t1-pie', 'figure'),
    Output('t1-box', 'figure'),
    Output('t1-bar', 'figure'),
    Output('t1-hist', 'figure'),
    Input('t1-domain', 'value'),
    Input('t1-stage', 'value'),
    Input('t1-innovation', 'value'),
)
def tab1_update(domains, stages, innovations):
    dff = filter_df(domains, stages, innovations)
    n = len(dff)
    row_count = f'{n:,} / 5,000 rows'
    metrics = [
        metric_card('Total Projects', f'{n:,}'),
        metric_card('Avg Competitiveness', f"{dff['competitiveness_score'].mean():.2f} / 50"),
        metric_card('Avg Team Size', f"{dff['team_size'].mean():.1f} members"),
        metric_card('Avg Market Growth', f"{dff['market_growth_rate'].mean():.1f}%"),
        metric_card('Unique Domains', dff['project_domain'].nunique()),
    ]
    pie_data = dff['project_domain'].value_counts().reset_index()
    pie_data.columns = ['project_domain', 'count']
    fig_pie = px.pie(pie_data, names='project_domain', values='count',
                     title='Project Distribution by Domain', color_discrete_sequence=CHART_COLORS)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(**CHART_LAYOUT)

    fig_box = px.box(dff, x='funding_stage', y='competitiveness_score',
                     color='funding_stage', title='Competitiveness by Funding Stage',
                     color_discrete_sequence=CHART_COLORS)
    fig_box.update_layout(**CHART_LAYOUT, showlegend=False)

    bar_data = dff['innovation_type'].value_counts().reset_index()
    bar_data.columns = ['innovation_type', 'count']
    fig_bar = px.bar(bar_data, x='innovation_type', y='count',
                     title='Projects by Innovation Type', color_discrete_sequence=[PRIMARY])
    fig_bar.update_layout(**CHART_LAYOUT)

    fig_hist = px.histogram(dff, x='competitiveness_score', nbins=25,
                             title='Competitiveness Score Distribution', color_discrete_sequence=[PRIMARY])
    fig_hist.update_layout(**CHART_LAYOUT)
    return row_count, metrics, fig_pie, fig_box, fig_bar, fig_hist


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — TAB 2
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output('t2-num-hist', 'figure'),
    Output('t2-freq-table', 'children'),
    Input('t2-domain', 'value'),
    Input('t2-stage', 'value'),
    Input('t2-num-var', 'value'),
)
def tab2_quantitative(domains, stages, var):
    dff = filter_df(domains, stages)
    series = dff[var].dropna()
    fig = px.histogram(series, nbins=25, title=f'{NUMERIC_LABELS[var]} — Histogram',
                       color_discrete_sequence=[PRIMARY])
    fig.update_layout(**CHART_LAYOUT)
    labels_cut, _ = pd.cut(series, bins=8, retbins=True)
    freq = labels_cut.value_counts().sort_index()
    total = freq.sum()
    rows, cum = [], 0
    for interval, f in freq.items():
        cum += f
        rows.append([f'{interval.left:.2f} – {interval.right:.2f}', f'{f:,}', f'{f/total*100:.1f}%', f'{cum:,}'])
    table = freq_table_html(['Class Interval', 'Frequency', 'Relative Freq (%)', 'Cumulative Freq'], rows)
    return fig, table


@app.callback(
    Output('t2-cat-bar', 'figure'),
    Output('t2-cat-pie', 'figure'),
    Output('t2-cat-table', 'children'),
    Input('t2-domain', 'value'),
    Input('t2-stage', 'value'),
    Input('t2-cat-var', 'value'),
)
def tab2_qualitative(domains, stages, var):
    dff = filter_df(domains, stages)
    counts = dff[var].value_counts().reset_index()
    counts.columns = [var, 'count']
    total = counts['count'].sum()
    fig_bar = px.bar(counts, x=var, y='count', title=f'Count by {CAT_LABELS[var]}',
                     color_discrete_sequence=[PRIMARY])
    fig_bar.update_layout(**CHART_LAYOUT)
    fig_pie = px.pie(counts, names=var, values='count', title=f'Distribution of {CAT_LABELS[var]}',
                     color_discrete_sequence=CHART_COLORS)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(**CHART_LAYOUT)
    rows, cum = [], 0
    for _, row in counts.iterrows():
        cum += row['count']
        rows.append([row[var], f"{row['count']:,}", f"{row['count']/total*100:.1f}%", f'{cum:,}'])
    table = freq_table_html(['Category', 'Frequency', 'Relative Freq (%)', 'Cumulative Freq'], rows)
    return fig_bar, fig_pie, table


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — TAB 3
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output('t3-boxplot', 'figure'),
    Output('t3-stats-table', 'children'),
    Output('t3-meanmedian', 'figure'),
    Output('t3-edbox', 'figure'),
    Output('t3-cv', 'figure'),
    Input('t3-var', 'value'),
)
def tab3_update(var):
    series = df[var].dropna()
    label = NUMERIC_LABELS[var]
    mean = float(series.mean())
    median = float(series.median())
    try:
        mode_val = float(series.mode().iloc[0])
    except Exception:
        mode_val = float('nan')
    var_val = float(series.var())
    std_val = float(series.std())
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = int(((series < lower_fence) | (series > upper_fence)).sum())
    try:
        skew_val = float(scipy_stats.skew(series))
        kurt_val = float(scipy_stats.kurtosis(series))
    except Exception:
        skew_val, kurt_val = 0.0, 0.0

    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        x=series,
        boxpoints='outliers',
        marker=dict(color=RED, size=5, opacity=0.6),
        line=dict(color=PRIMARY, width=1.5),
        fillcolor='rgba(192, 132, 160, 0.25)',
        name=label,
        hovertemplate='%{x:.2f}<extra></extra>',
    ))
    fig_box.update_layout(
        **CHART_LAYOUT,
        title='Box Plot with Outlier Detection',
        xaxis=dict(title=label),
        yaxis=dict(showticklabels=False),
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='Inter, Segoe UI, sans-serif',
                        bordercolor=BORDER),
    )

    stat_rows_data = [
        ('Mean', f'{mean:.4f}'), ('Median', f'{median:.4f}'),
        ('Mode', f'{mode_val:.4f}' if not np.isnan(mode_val) else 'N/A'),
        ('Variance', f'{var_val:.4f}'), ('Std Deviation', f'{std_val:.4f}'),
        ('Q1 (25th pct)', f'{q1:.4f}'), ('Q3 (75th pct)', f'{q3:.4f}'),
        ('IQR', f'{iqr:.4f}'), ('Lower Fence', f'{lower_fence:.4f}'),
        ('Upper Fence', f'{upper_fence:.4f}'), ('Outliers', str(outliers)),
        ('Kurtosis', f'{kurt_val:.4f}'),
    ]
    tbl_rows = []
    for i, (k, v) in enumerate(stat_rows_data):
        bg = '#f9f9f9' if i % 2 == 0 else '#ffffff'
        tbl_rows.append(html.Tr([
            html.Td(k, style={'padding': '7px 10px', 'fontSize': '12px', 'color': TEXT_SEC,
                              'border': f'1px solid {BORDER}', 'background': bg, 'fontWeight': '500'}),
            html.Td(v, style={'padding': '7px 10px', 'fontSize': '12px', 'color': TEXT,
                              'border': f'1px solid {BORDER}', 'background': bg, 'fontWeight': '600'}),
        ]))
    tbl_rows.append(html.Tr([
        html.Td('Skewness', style={'padding': '7px 10px', 'fontSize': '12px', 'color': TEXT_SEC,
                                    'border': f'1px solid {BORDER}', 'background': '#ffffff', 'fontWeight': '500'}),
        html.Td([f'{skew_val:.4f}  ', skew_badge(skew_val)],
                style={'padding': '7px 10px', 'fontSize': '12px', 'border': f'1px solid {BORDER}', 'background': '#ffffff'}),
    ]))
    stats_tbl = html.Table(html.Tbody(tbl_rows), style={'width': '100%', 'borderCollapse': 'collapse'})

    fig_mm = px.histogram(df, x=var, nbins=30, title=f'Distribution Shape — Mean vs Median ({label})',
                          color_discrete_sequence=[PRIMARY])
    fig_mm.add_vline(x=mean, line_dash='dash', line_color=RED,
                     annotation_text='Mean', annotation_position='top right', annotation_font_color=RED)
    fig_mm.add_vline(x=median, line_dash='dash', line_color=GREEN,
                     annotation_text='Median', annotation_position='top left', annotation_font_color=GREEN)
    fig_mm.update_layout(**CHART_LAYOUT)

    fig_edbox = px.box(df, x='education_level', y='competitiveness_score',
                       color='education_level', title='Competitiveness by Education Level',
                       color_discrete_sequence=CHART_COLORS)
    fig_edbox.update_layout(**CHART_LAYOUT, showlegend=False)

    cv_data = {}
    for v in NUMERIC_VARS:
        s = df[v].dropna()
        if s.mean() != 0:
            cv_data[NUMERIC_LABELS[v]] = abs(s.std() / s.mean() * 100)
    cv_df = pd.DataFrame(list(cv_data.items()), columns=['Variable', 'CV (%)'])
    cv_df = cv_df.sort_values('CV (%)', ascending=True)
    fig_cv = px.bar(cv_df, x='CV (%)', y='Variable', orientation='h',
                    title='Coefficient of Variation (%) — Comparing Relative Variability Across Variables',
                    color_discrete_sequence=[PRIMARY])
    fig_cv.update_layout(**CHART_LAYOUT)

    return fig_box, stats_tbl, fig_mm, fig_edbox, fig_cv


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — TAB 4
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output('binom-k', 'max'),
    Input('binom-n', 'value'),
)
def update_binom_k_max(n):
    return int(n or 20)


@app.callback(
    Output('norm-x', 'min'),
    Output('norm-x', 'max'),
    Output('norm-x', 'value'),
    Input('norm-mu', 'value'),
    Input('norm-sigma', 'value'),
)
def update_norm_x_range(mu, sigma):
    mu = float(mu or df['competitiveness_score'].mean())
    sigma = max(float(sigma or df['competitiveness_score'].std()), 0.01)
    return round(mu - 4 * sigma, 2), round(mu + 4 * sigma, 2), round(mu, 2)


@app.callback(
    Output('binom-chart', 'figure'),
    Output('binom-metrics', 'children'),
    Input('binom-n', 'value'),
    Input('binom-p', 'value'),
    Input('binom-k', 'value'),
)
def tab4_binomial(n, p, k):
    n = int(n or 20)
    p = float(p or 0.3)
    k = min(int(k or 0), n)
    ks = np.arange(0, n + 1)
    pmf = binom.pmf(ks, n, p)
    colors_bar = [RED if ki == k else PRIMARY_LIGHT for ki in ks]
    fig = go.Figure(go.Bar(x=ks, y=pmf, marker_color=colors_bar, name='PMF'))
    fig.update_layout(**CHART_LAYOUT, title=f'Binomial Distribution B(n={n}, p={p})',
                      xaxis_title='k (successes)', yaxis_title='P(X=k)')
    px_eq_k = binom.pmf(k, n, p)
    px_le_k = binom.cdf(k, n, p)
    metrics = [
        small_metric_card(f'P(X = {k})', f'{px_eq_k:.4f}', PRIMARY),
        small_metric_card(f'P(X ≤ {k})', f'{px_le_k:.4f}', GREEN),
        small_metric_card(f'P(X > {k})', f'{1-px_le_k:.4f}', RED),
        small_metric_card('Mean (np)', f'{n*p:.3f}', AMBER),
        small_metric_card('Variance (npq)', f'{n*p*(1-p):.3f}', PRIMARY_LIGHT),
    ]
    return fig, metrics


@app.callback(
    Output('pois-chart', 'figure'),
    Output('pois-metrics', 'children'),
    Input('pois-lam', 'value'),
    Input('pois-k', 'value'),
)
def tab4_poisson(lam, k):
    lam = float(lam or 3.0)
    k = int(k or 0)
    ks = np.arange(0, 21)
    pmf = poisson.pmf(ks, lam)
    colors_bar = [RED if ki == k else PRIMARY_LIGHT for ki in ks]
    fig = go.Figure(go.Bar(x=ks, y=pmf, marker_color=colors_bar, name='PMF'))
    fig.update_layout(**CHART_LAYOUT, title=f'Poisson Distribution (λ={lam})',
                      xaxis_title='k', yaxis_title='P(X=k)')
    px_eq_k = poisson.pmf(k, lam)
    px_le_k = poisson.cdf(k, lam)
    metrics = [
        small_metric_card(f'P(X = {k})', f'{px_eq_k:.4f}', PRIMARY),
        small_metric_card(f'P(X ≤ {k})', f'{px_le_k:.4f}', GREEN),
        small_metric_card(f'P(X > {k})', f'{1-px_le_k:.4f}', RED),
        small_metric_card('Mean (λ)', f'{lam:.2f}', AMBER),
        small_metric_card('Variance (λ)', f'{lam:.2f}', PRIMARY_LIGHT),
    ]
    return fig, metrics


@app.callback(
    Output('norm-chart', 'figure'),
    Output('norm-metrics', 'children'),
    Input('norm-mu', 'value'),
    Input('norm-sigma', 'value'),
    Input('norm-x', 'value'),
)
def tab4_normal(mu, sigma, x):
    mu = float(mu or df['competitiveness_score'].mean())
    sigma = max(float(sigma or df['competitiveness_score'].std()), 0.01)
    x = float(x or mu)
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ys = norm.pdf(xs, mu, sigma)
    xs_shade = np.linspace(mu - 4 * sigma, x, 300)
    ys_shade = norm.pdf(xs_shade, mu, sigma)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs_shade.tolist() + [x, mu - 4 * sigma], y=ys_shade.tolist() + [0, 0],
        fill='toself', fillcolor='rgba(139, 79, 122, 0.2)',
        line=dict(color='rgba(0,0,0,0)'), name=f'P(X ≤ {x:.1f})'))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
                             line=dict(color=PRIMARY, width=2.5), name='PDF'))
    fig.add_vline(x=x, line_dash='dash', line_color=RED,
                  annotation_text=f'x={x:.1f}', annotation_font_color=RED)
    fig.update_layout(**CHART_LAYOUT, title=f'Normal Distribution N(μ={mu:.2f}, σ={sigma:.2f})',
                      xaxis_title='x', yaxis_title='f(x)')
    px_le = float(norm.cdf(x, mu, sigma))
    z = (x - mu) / sigma
    metrics = [
        small_metric_card(f'P(X ≤ {x:.1f})', f'{px_le:.4f}', PRIMARY),
        small_metric_card(f'P(X > {x:.1f})', f'{1-px_le:.4f}', RED),
        small_metric_card('Z-score', f'{z:.4f}', AMBER),
    ]
    return fig, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS — TAB 5
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output('reg-scatter', 'figure'),
    Output('reg-stats', 'children'),
    Input('reg-x', 'value'),
    Input('reg-y', 'value'),
)
def tab5_regression(x_var, y_var):
    dff = df[[x_var, y_var]].dropna()
    x_data = dff[x_var].values.ravel()
    y_data = dff[y_var].values.ravel()
    m, b = np.polyfit(x_data, y_data, 1)
    y_pred = m * x_data + b
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    try:
        r_val, p_val = scipy_stats.pearsonr(x_data, y_data)
    except Exception:
        r_val, p_val = 0.0, 1.0
    if abs(r_val) >= 0.7:
        interp = 'Strong ' + ('positive' if r_val > 0 else 'negative') + ' relationship'
    elif abs(r_val) >= 0.3:
        interp = 'Moderate ' + ('positive' if r_val > 0 else 'negative') + ' relationship'
    elif abs(r_val) >= 0.1:
        interp = 'Weak linear relationship'
    else:
        interp = 'No linear relationship'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers',
                             marker=dict(color=PRIMARY_LIGHT, size=5, opacity=0.5), name='Data'))
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    fig.add_trace(go.Scatter(x=x_line, y=m * x_line + b, mode='lines',
                             line=dict(color=RED, width=2), name='Regression line'))
    fig.add_annotation(x=0.05, y=0.95, xref='paper', yref='paper',
                       text=f'y = {m:.3f}x + {b:.3f}<br>R² = {r2:.4f}',
                       showarrow=False, bgcolor='rgba(255,255,255,0.85)',
                       bordercolor=BORDER, borderwidth=1, font=dict(size=12, color=TEXT))
    fig.update_layout(**CHART_LAYOUT, title=f'{NUMERIC_LABELS[y_var]} vs {NUMERIC_LABELS[x_var]}',
                      xaxis_title=NUMERIC_LABELS[x_var], yaxis_title=NUMERIC_LABELS[y_var])
    stat_items = [('Slope (m)', f'{m:.4f}'), ('Intercept (b)', f'{b:.4f}'),
                  ('R² (fit quality)', f'{r2:.4f}'), ('Pearson r', f'{r_val:.4f}'),
                  ('p-value', f'{p_val:.4e}')]
    tbl_rows = []
    for i, (k, v) in enumerate(stat_items):
        bg = '#f9f9f9' if i % 2 == 0 else '#ffffff'
        tbl_rows.append(html.Tr([
            html.Td(k, style={'padding': '7px 10px', 'fontSize': '12px', 'color': TEXT_SEC,
                              'border': f'1px solid {BORDER}', 'background': bg, 'fontWeight': '500'}),
            html.Td(v, style={'padding': '7px 10px', 'fontSize': '12px', 'color': TEXT,
                              'border': f'1px solid {BORDER}', 'background': bg, 'fontWeight': '600'}),
        ]))
    tbl_rows.append(html.Tr([
        html.Td('Interpretation', style={'padding': '7px 10px', 'fontSize': '12px', 'color': TEXT_SEC,
                                          'border': f'1px solid {BORDER}', 'background': '#ffffff', 'fontWeight': '500'}),
        html.Td(interp, style={'padding': '7px 10px', 'fontSize': '12px', 'color': PRIMARY,
                                'border': f'1px solid {BORDER}', 'background': '#ffffff', 'fontWeight': '600'}),
    ]))
    stats_tbl = html.Table(html.Tbody(tbl_rows), style={'width': '100%', 'borderCollapse': 'collapse'})
    return fig, stats_tbl


@app.callback(
    Output('pred-output', 'children'),
    Output('pred-scatter', 'figure'),
    Input('sl-tech', 'value'),
    Input('sl-bus', 'value'),
    Input('sl-exp', 'value'),
    Input('sl-team', 'value'),
    Input('sl-growth', 'value'),
    Input('sl-comp', 'value'),
)
def tab5_predict(tech, bus, exp, team, growth, comp):
    input_df = pd.DataFrame([[tech, bus, exp, team, growth, comp]], columns=FEATURES)
    pred = float(reg_model.predict(input_df)[0])
    pred = max(0, min(50, pred))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['technical_skill_score'], y=df['competitiveness_score'],
                             mode='markers', marker=dict(color=PRIMARY_LIGHT, size=4, opacity=0.4), name='Actual data'))
    fig.add_trace(go.Scatter(x=[tech], y=[pred], mode='markers',
                             marker=dict(color=RED, size=16, symbol='star'), name='Your prediction'))
    fig.update_layout(**CHART_LAYOUT, title='Technical Skill Score vs Competitiveness (your prediction as ★)',
                      xaxis_title='Technical Skill Score', yaxis_title='Competitiveness Score')
    return f'{pred:.2f}', fig


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
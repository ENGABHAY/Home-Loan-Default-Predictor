import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from predictor import LoanDefaultPredictor

st.set_page_config(
    page_title="Home Credit — Risk Intelligence",
    page_icon="🏛️",
    layout="wide"
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #0a0e1a; color: #d4dbe8; }

[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2a40;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4a90d9; font-size: 0.7rem;
    letter-spacing: 0.12em; text-transform: uppercase; margin-top: 1.2rem;
}

/* Header */
.header-band {
    background: linear-gradient(135deg, #0d1220 0%, #112240 60%, #0d1b34 100%);
    border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 28px 36px; margin-bottom: 28px;
    display: flex; align-items: center; gap: 18px;
}
.header-band .logo { font-size: 2.4rem; }
.header-band h1 {
    font-size: 1.7rem; font-weight: 600; color: #e8eef8;
    margin: 0 0 4px 0; letter-spacing: -0.02em;
}
.header-band p {
    font-size: 0.85rem; color: #6b82a0;
    margin: 0; font-weight: 300; letter-spacing: 0.04em;
}

/* Landing cards */
.landing-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 16px; margin: 8px 0 28px 0;
}
.landing-card {
    background: #0d1220; border: 1px solid #1e2a40;
    border-radius: 10px; padding: 22px 24px;
    position: relative; overflow: hidden;
}
.landing-card::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
}
.lc-blue::after  { background: linear-gradient(90deg, #1a4a8a, #4a90d9); }
.lc-teal::after  { background: linear-gradient(90deg, #0e4a4a, #22d3d3); }
.lc-indigo::after{ background: linear-gradient(90deg, #2a1a6a, #7c6aee); }

.landing-card .lc-icon { font-size: 1.8rem; margin-bottom: 12px; }
.landing-card h4 {
    font-size: 0.85rem; font-weight: 600;
    color: #c8d8f0; margin: 0 0 8px 0;
}
.landing-card p {
    font-size: 0.78rem; color: #4a6080;
    margin: 0; line-height: 1.6;
}

/* Pipeline steps */
.pipeline-row {
    display: flex; align-items: center; gap: 0;
    margin: 16px 0 28px 0;
}
.pipe-step {
    flex: 1; background: #0d1220; border: 1px solid #1e2a40;
    border-radius: 8px; padding: 14px 16px; text-align: center;
}
.pipe-step .ps-icon { font-size: 1.3rem; margin-bottom: 6px; }
.pipe-step .ps-title {
    font-size: 0.7rem; font-weight: 600;
    color: #c8d8f0; text-transform: uppercase; letter-spacing: 0.08em;
}
.pipe-step .ps-sub { font-size: 0.68rem; color: #4a6080; margin-top: 3px; }
.pipe-arrow {
    color: #1e3a5f; font-size: 1.2rem;
    padding: 0 6px; flex-shrink: 0;
}

/* Stats strip */
.stats-strip {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin: 8px 0 28px 0;
}
.stat-item {
    background: #0d1220; border: 1px solid #1e2a40;
    border-radius: 8px; padding: 14px 18px;
    display: flex; align-items: center; gap: 14px;
}
.stat-item .si-icon { font-size: 1.4rem; }
.stat-item .si-label {
    font-size: 0.67rem; color: #4a6080;
    text-transform: uppercase; letter-spacing: 0.1em;
}
.stat-item .si-val {
    font-size: 1rem; font-weight: 600;
    color: #c8d8f0; font-family: 'IBM Plex Mono', monospace;
}

/* Risk cards */
.risk-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 16px; margin: 8px 0 24px 0;
}
.risk-card {
    border-radius: 10px; padding: 22px 24px;
    position: relative; overflow: hidden;
}
.risk-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
}
.risk-low   { background: #071a12; border: 1px solid #1a3d28; }
.risk-medium{ background: #1a1505; border: 1px solid #3d3010; }
.risk-high  { background: #1a0808; border: 1px solid #3d1515; }
.risk-low::before   { background: #22c55e; }
.risk-medium::before{ background: #f59e0b; }
.risk-high::before  { background: #ef4444; }

.risk-card .badge {
    display: inline-block; font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px; margin-bottom: 12px;
}
.risk-low .badge   { background: #14532d; color: #4ade80; }
.risk-medium .badge{ background: #451a03; color: #fbbf24; }
.risk-high .badge  { background: #450a0a; color: #f87171; }

.risk-card .count {
    font-size: 2.4rem; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace; line-height: 1; margin-bottom: 4px;
}
.risk-low .count   { color: #4ade80; }
.risk-medium .count{ color: #fbbf24; }
.risk-high .count  { color: #f87171; }

.risk-card .pct { font-size: 0.8rem; color: #6b82a0; font-family: 'IBM Plex Mono', monospace; }
.risk-card .desc{ font-size: 0.78rem; color: #6b82a0; margin-top: 8px; line-height: 1.5; }

/* Metric row */
.metric-row {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin-bottom: 24px;
}
.metric-box {
    background: #0d1220; border: 1px solid #1e2a40;
    border-radius: 8px; padding: 16px 20px; text-align: center;
}
.metric-box .label {
    font-size: 0.68rem; color: #4a6080;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;
}
.metric-box .value {
    font-size: 1.5rem; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace; color: #c8d8f0;
}
.metric-box.highlight {
    border-color: #1e3a5f;
    background: linear-gradient(135deg, #0d1220, #0d1b34);
}
.metric-box .sublabel {
    font-size: 0.63rem; color: #344a64;
    margin-top: 4px; line-height: 1.4;
}
.conf-bar-wrap {
    background: #1a2236; border-radius: 4px;
    height: 5px; margin-top: 10px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #1a4a8a, #4a90d9);
}

/* Section label */
.section-label {
    font-size: 0.68rem; color: #4a6080;
    text-transform: uppercase; letter-spacing: 0.14em;
    border-bottom: 1px solid #1e2a40;
    padding-bottom: 8px; margin: 28px 0 16px 0;
}

/* Button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a4a8a, #1e5faa) !important;
    border: 1px solid #2a6ec0 !important; color: #e8eef8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important; letter-spacing: 0.04em !important;
    border-radius: 6px !important; height: 44px !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e5faa, #2070cc) !important;
}
.stAlert { border-radius: 8px !important; }
.stDownloadButton > button {
    background: #0d1220 !important; border: 1px solid #1e3a5f !important;
    color: #4a90d9 !important; border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
    <div class="logo">🏛️</div>
    <div>
        <h1>Home Credit — Risk Intelligence</h1>
        <p>LOAN DEFAULT PROBABILITY ENGINE &nbsp;·&nbsp; POWERED BY CATBOOST</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Predictor ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return LoanDefaultPredictor()

predictor = load_predictor()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Upload Data")
application_file = st.sidebar.file_uploader("Application Data (Required)", type=["csv"], key="app")

st.sidebar.markdown("### Auxiliary Files")
c1, c2 = st.sidebar.columns(2)
bureau_file = c1.file_uploader("Bureau", type=["csv"], key="bureau")
prev_file   = c2.file_uploader("Previous App", type=["csv"], key="prev")
c3, c4 = st.sidebar.columns(2)
pos_file = c3.file_uploader("POS Cash", type=["csv"], key="pos")
cc_file  = c4.file_uploader("Credit Card", type=["csv"], key="cc")
inst_file = st.sidebar.file_uploader("Installments", type=["csv"], key="inst")

# ── LANDING STATE ─────────────────────────────────────────────────────────────
if application_file is None:

    # Info cards
    st.markdown("""
    <div class="landing-grid">
        <div class="landing-card lc-blue">
            <div class="lc-icon">🧠</div>
            <h4>CatBoost Risk Model</h4>
            <p>Gradient boosted decision tree model trained on the Home Credit dataset.
               Handles categorical features natively with no preprocessing required.</p>
        </div>
        <div class="landing-card lc-teal">
            <div class="lc-icon">🗂️</div>
            <h4>Multi-Source Feature Engineering</h4>
            <p>Aggregates signals from bureau history, previous applications, POS cash,
               credit card balances, and installment payment behaviour.</p>
        </div>
        <div class="landing-card lc-indigo">
            <div class="lc-icon">📊</div>
            <h4>Three-Tier Risk Segmentation</h4>
            <p>Applicants are segmented into Low, Medium, and High risk bands with
               actionable underwriting recommendations for each tier.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline steps
    st.markdown('<div class="section-label">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-row">
        <div class="pipe-step">
            <div class="ps-icon">📁</div>
            <div class="ps-title">Upload CSVs</div>
            <div class="ps-sub">Application + optional aux files</div>
        </div>
        <div class="pipe-arrow">▶</div>
        <div class="pipe-step">
            <div class="ps-icon">⚙️</div>
            <div class="ps-title">Feature Engineering</div>
            <div class="ps-sub">Aggregations across all tables</div>
        </div>
        <div class="pipe-arrow">▶</div>
        <div class="pipe-step">
            <div class="ps-icon">🤖</div>
            <div class="ps-title">Model Inference</div>
            <div class="ps-sub">CatBoost predicts default probability</div>
        </div>
        <div class="pipe-arrow">▶</div>
        <div class="pipe-step">
            <div class="ps-icon">🎯</div>
            <div class="ps-title">Risk Segmentation</div>
            <div class="ps-sub">Low / Medium / High bands assigned</div>
        </div>
        <div class="pipe-arrow">▶</div>
        <div class="pipe-step">
            <div class="ps-icon">📥</div>
            <div class="ps-title">Export Results</div>
            <div class="ps-sub">Download full predictions as CSV</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model stats strip
    st.markdown('<div class="section-label">Model Specifications</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stats-strip">
        <div class="stat-item">
            <div class="si-icon">🎯</div>
            <div>
                <div class="si-label">Algorithm</div>
                <div class="si-val">CatBoost</div>
            </div>
        </div>
        <div class="stat-item">
            <div class="si-icon">🗃️</div>
            <div>
                <div class="si-label">Source Tables</div>
                <div class="si-val">7 CSVs</div>
            </div>
        </div>
        <div class="stat-item">
            <div class="si-icon">📐</div>
            <div>
                <div class="si-label">Risk Bands</div>
                <div class="si-val">3 Tiers</div>
            </div>
        </div>
        <div class="stat-item">
            <div class="si-icon">📤</div>
            <div>
                <div class="si-label">Output</div>
                <div class="si-val">P(Default)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("👆 Upload the **Application Data CSV** in the sidebar to run the risk assessment.")

# ── PREDICTION STATE ──────────────────────────────────────────────────────────
else:
    try:
        application_df = pd.read_csv(application_file)
        st.success(f"✓ Loaded **{len(application_df):,}** application records")

        st.markdown('<div class="section-label">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(application_df.head(), use_container_width=True)

        kwargs = {}
        if bureau_file: kwargs['bureau_df'] = pd.read_csv(bureau_file)
        if prev_file:   kwargs['prev_df']   = pd.read_csv(prev_file)
        if pos_file:    kwargs['pos_df']    = pd.read_csv(pos_file)
        if cc_file:     kwargs['cc_df']     = pd.read_csv(cc_file)
        if inst_file:   kwargs['inst_df']   = pd.read_csv(inst_file)

        if st.button("Run Risk Assessment", type="primary", use_container_width=True):
            with st.spinner("Analysing risk profiles…"):
                probs = predictor.predict_proba(application_df, **kwargs)

            result_df = pd.DataFrame({
                'SK_ID_CURR':          application_df['SK_ID_CURR'],
                'Default_Probability': probs.round(6),
                'Risk_Level':          pd.cut(probs,
                                              bins=[0, 0.3, 0.6, 1.0],
                                              labels=['Low', 'Medium', 'High'])
            })

            n_low    = int((result_df['Risk_Level'] == 'Low').sum())
            n_medium = int((result_df['Risk_Level'] == 'Medium').sum())
            n_high   = int((result_df['Risk_Level'] == 'High').sum())
            total    = len(result_df)

            # ── Summary metrics ──────────────────────────────────────────
            # Confidence = how far each prediction is from a 50/50 coin flip
            # max(p, 1-p) → 1.0 means perfectly certain, 0.5 means pure uncertainty
            avg_confidence = float(np.mean(np.maximum(probs, 1 - probs))) * 100
            conf_pct = f"{avg_confidence:.1f}%"

            st.markdown('<div class="section-label">Portfolio Summary</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="label">Total Applications</div>
                    <div class="value">{total:,}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Avg Default Probability</div>
                    <div class="value">{probs.mean():.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Median Probability</div>
                    <div class="value">{float(np.median(probs)):.4f}</div>
                </div>
                <div class="metric-box highlight">
                    <div class="label">Prediction Confidence</div>
                    <div class="value">{conf_pct}</div>
                    <div class="sublabel">Avg certainty across all predictions<br>(100% = fully certain, 50% = coin flip)</div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-fill" style="width:{avg_confidence:.1f}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Risk cards ───────────────────────────────────────────────
            st.markdown('<div class="section-label">Risk Segmentation</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="risk-grid">
                <div class="risk-card risk-low">
                    <span class="badge">✓ Low Risk</span>
                    <div class="count">{n_low:,}</div>
                    <div class="pct">{n_low/total*100:.1f}% of portfolio</div>
                    <div class="desc">Default probability &lt; 30%<br>Eligible for standard loan terms.</div>
                </div>
                <div class="risk-card risk-medium">
                    <span class="badge">⚠ Medium Risk</span>
                    <div class="count">{n_medium:,}</div>
                    <div class="pct">{n_medium/total*100:.1f}% of portfolio</div>
                    <div class="desc">Default probability 30–60%<br>Recommend additional verification.</div>
                </div>
                <div class="risk-card risk-high">
                    <span class="badge">✕ High Risk</span>
                    <div class="count">{n_high:,}</div>
                    <div class="pct">{n_high/total*100:.1f}% of portfolio</div>
                    <div class="desc">Default probability &gt; 60%<br>Flag for manual review or decline.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Charts ───────────────────────────────────────────────────
            st.markdown('<div class="section-label">Risk Analytics</div>', unsafe_allow_html=True)

            chart_col1, chart_col2 = st.columns(2)

            PLOT_BG    = "#0a0e1a"
            PAPER_BG   = "#0a0e1a"
            GRID_COLOR = "#1e2a40"
            FONT_COLOR = "#6b82a0"
            AXIS_COLOR = "#1e2a40"

            # Donut chart
            with chart_col1:
                donut = go.Figure(go.Pie(
                    labels=['Low Risk', 'Medium Risk', 'High Risk'],
                    values=[n_low, n_medium, n_high],
                    hole=0.62,
                    marker=dict(
                        colors=['#22c55e', '#f59e0b', '#ef4444'],
                        line=dict(color=PLOT_BG, width=3)
                    ),
                    textinfo='percent',
                    textfont=dict(family='IBM Plex Mono', size=12, color='#d4dbe8'),
                    hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>'
                ))
                donut.add_annotation(
                    text=f"<b>{total:,}</b><br><span style='font-size:11px'>Applications</span>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(family='IBM Plex Mono', size=16, color='#c8d8f0'),
                    align='center'
                )
                donut.update_layout(
                    title=dict(text="Portfolio Risk Distribution", font=dict(
                        family='IBM Plex Sans', size=13, color='#8a9ab8'), x=0.04, y=0.97),
                    paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                    margin=dict(t=40, b=20, l=20, r=20),
                    legend=dict(
                        font=dict(family='IBM Plex Sans', size=11, color=FONT_COLOR),
                        bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)',
                        orientation='h', x=0.5, xanchor='center', y=-0.05
                    ),
                    height=320
                )
                st.plotly_chart(donut, use_container_width=True, config={'displayModeBar': False})

            # Histogram
            with chart_col2:
                hist = go.Figure()
                hist.add_trace(go.Histogram(
                    x=probs,
                    nbinsx=40,
                    marker=dict(
                        color=probs,
                        colorscale=[[0, '#22c55e'], [0.3, '#22c55e'],
                                    [0.3, '#f59e0b'], [0.6, '#f59e0b'],
                                    [0.6, '#ef4444'], [1.0, '#ef4444']],
                        line=dict(color=PLOT_BG, width=0.5)
                    ),
                    hovertemplate='Probability: %{x:.2f}<br>Count: %{y}<extra></extra>',
                    opacity=0.85
                ))
                hist.add_vline(x=0.3, line_dash="dash", line_color="#f59e0b",
                               line_width=1.5, annotation_text="Medium",
                               annotation_font=dict(color="#f59e0b", size=10))
                hist.add_vline(x=0.6, line_dash="dash", line_color="#ef4444",
                               line_width=1.5, annotation_text="High",
                               annotation_font=dict(color="#ef4444", size=10))
                hist.update_layout(
                    title=dict(text="Default Probability Distribution", font=dict(
                        family='IBM Plex Sans', size=13, color='#8a9ab8'), x=0.04, y=0.97),
                    paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                    xaxis=dict(title='P(Default)', color=FONT_COLOR,
                               gridcolor=GRID_COLOR, zerolinecolor=AXIS_COLOR,
                               tickfont=dict(family='IBM Plex Mono', size=10)),
                    yaxis=dict(title='Count', color=FONT_COLOR,
                               gridcolor=GRID_COLOR, zerolinecolor=AXIS_COLOR,
                               tickfont=dict(family='IBM Plex Mono', size=10)),
                    margin=dict(t=40, b=40, l=50, r=20),
                    height=320,
                    bargap=0.05
                )
                st.plotly_chart(hist, use_container_width=True, config={'displayModeBar': False})

            # Cumulative risk curve  (full width)
            sorted_probs = np.sort(probs)
            pct_customers = np.linspace(0, 100, len(sorted_probs))

            area = go.Figure()
            area.add_trace(go.Scatter(
                x=pct_customers, y=sorted_probs,
                mode='lines',
                line=dict(color='#4a90d9', width=2),
                fill='tozeroy',
                fillcolor='rgba(74,144,217,0.08)',
                hovertemplate='Top %{x:.1f}% of applicants<br>P(Default): %{y:.4f}<extra></extra>',
                name='Cumulative Risk'
            ))
            area.add_hrect(y0=0, y1=0.3,   fillcolor='rgba(34,197,94,0.05)',  line_width=0)
            area.add_hrect(y0=0.3, y1=0.6, fillcolor='rgba(245,158,11,0.05)', line_width=0)
            area.add_hrect(y0=0.6, y1=1.0, fillcolor='rgba(239,68,68,0.05)',  line_width=0)
            area.update_layout(
                title=dict(text="Cumulative Default Probability — Ranked Applicants",
                           font=dict(family='IBM Plex Sans', size=13, color='#8a9ab8'), x=0.02, y=0.97),
                paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                xaxis=dict(title='Applicant Percentile (%)', color=FONT_COLOR,
                           gridcolor=GRID_COLOR, zerolinecolor=AXIS_COLOR,
                           tickfont=dict(family='IBM Plex Mono', size=10)),
                yaxis=dict(title='P(Default)', color=FONT_COLOR,
                           gridcolor=GRID_COLOR, zerolinecolor=AXIS_COLOR,
                           tickfont=dict(family='IBM Plex Mono', size=10),
                           range=[0, 1]),
                margin=dict(t=40, b=50, l=55, r=20),
                height=280,
                showlegend=False
            )
            st.plotly_chart(area, use_container_width=True, config={'displayModeBar': False})

            # ── Results table ────────────────────────────────────────────
            st.markdown('<div class="section-label">Individual Predictions (first 100)</div>', unsafe_allow_html=True)
            st.dataframe(result_df.head(100), use_container_width=True)

            st.download_button(
                label="📥 Download Full Predictions (.csv)",
                data=result_df.to_csv(index=False),
                file_name="loan_default_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")

st.caption("Home Credit Risk Intelligence · CatBoost · Internal Use Only")
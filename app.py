"""
app.py

Day 12: Sentinel Ring dashboard (PRD.md FR8).

DESIGN INTENT (not a default Streamlit look -- deliberate choices):
This is framed as a security-operations investigation console, not a generic
analytics dashboard. Dark, precise, monospace data labels, evidence
presented as case files -- because the subject is fraud investigation, and
the audience (a risk reviewer) needs to scan dense evidence fast, not admire
charts. Signature element: candidate rings as case cards + agent output
rendered as a structured case file (verdict badge + evidence list), not raw
JSON dumped on screen.

Palette: charcoal base (#0E1116), elevated panel (#161B22), hairline border
(#2A313C), text (#E6E8EB), risk HIGH/MEDIUM/LOW (#E5484D / #F5A623 /
#3FB950), accent (#58A6FF).
Type: IBM Plex Mono (headers, IDs, data), IBM Plex Sans (body).

Run with: streamlit run app.py
"""

import os
import sys
import json

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import DATA_DIR
from src.agent.investigator import investigate_ring

st.set_page_config(
    page_title="Sentinel Ring",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS: fonts, palette, card/badge system, hide default chrome ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg: #0E1116;
    --panel: #161B22;
    --border: #2A313C;
    --text: #E6E8EB;
    --text-dim: #8B949E;
    --risk-high: #E5484D;
    --risk-medium: #F5A623;
    --risk-low: #3FB950;
    --accent: #58A6FF;
}

#MainMenu, footer, header {visibility: hidden;}
.stApp { background-color: var(--bg); }
body, .stApp, p, div, span { font-family: 'IBM Plex Sans', sans-serif; color: var(--text); }

h1, h2, h3, .mono {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.02em;
}

.sentinel-header {
    display: flex; align-items: baseline; gap: 12px;
    border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px;
}
.sentinel-header .title {
    font-family: 'IBM Plex Mono', monospace; font-size: 40px; font-weight: 700;
    color: var(--text); letter-spacing: -0.01em;
}
.sentinel-header .subtitle { color: var(--text-dim); font-size: 13px; }
.status-pill {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; padding: 3px 10px;
    border-radius: 3px; background: var(--panel); border: 1px solid var(--border);
    color: var(--text-dim); margin-left: auto;
}

.case-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px 6px 0 0;
    padding: 14px 16px 10px 16px; margin-bottom: 0; border-left: 3px solid var(--border);
    border-bottom: none;
}
.case-card.risk-tier-high { border-left-color: var(--risk-high); }
.case-card.risk-tier-medium { border-left-color: var(--risk-medium); }
.case-card.risk-tier-low { border-left-color: var(--risk-low); }

.case-card .case-id {
    font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 600; color: var(--text);
}
.case-card .case-meta { font-size: 12px; color: var(--text-dim); margin-top: 4px; }
.risk-score-bar-bg { background: var(--border); border-radius: 3px; height: 6px; margin-top: 8px; overflow: hidden; }
.risk-score-bar-fill { height: 100%; border-radius: 3px; }

.attr-tag {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    background: #1F2530; border: 1px solid var(--border); color: var(--text-dim);
    padding: 2px 7px; border-radius: 3px; margin-right: 5px; margin-top: 6px;
}

.verdict-badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-weight: 700;
    font-size: 13px; padding: 5px 14px; border-radius: 4px; letter-spacing: 0.03em;
}
.verdict-HIGH { background: rgba(229,72,77,0.15); color: var(--risk-high); border: 1px solid var(--risk-high); }
.verdict-MEDIUM { background: rgba(245,166,35,0.15); color: var(--risk-medium); border: 1px solid var(--risk-medium); }
.verdict-LOW { background: rgba(63,185,80,0.15); color: var(--risk-low); border: 1px solid var(--risk-low); }

.action-badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    padding: 3px 10px; border-radius: 3px; background: var(--panel); border: 1px solid var(--accent);
    color: var(--accent); margin-left: 8px;
}

.evidence-item {
    font-size: 13px; padding: 8px 12px; background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; margin-bottom: 6px; border-left: 2px solid var(--accent);
}

.case-file-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.08em; margin: 16px 0 8px 0;
}

/* Style Streamlit's native buttons to match the console theme, and fuse
   them visually onto the card above (negative margin closes Streamlit's
   default inter-element gap; flat top + no top border makes it read as
   the card's bottom edge, not a separate floating button) */
.stButton {
    margin-top: -11px !important;
    margin-bottom: 10px !important;
}
.stButton > button {
    background: var(--panel) !important;
    color: var(--text-dim) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.03em !important;
    border-radius: 0 0 6px 6px !important;
    padding: 6px 12px !important;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: #1A2130 !important;
}
.stButton > button:active, .stButton > button:focus {
    background: var(--accent) !important;
    color: var(--bg) !important;
    border-color: var(--accent) !important;
}

/* Scrolling alert ticker -- live feed of top HIGH-risk candidates, styled
   like a monitoring-console alert bar. Motion is deliberate here (not
   decoration): it reinforces "this is an active monitoring system", the
   same logic real SOC/trading terminals use scrolling alert bars for. */
.ticker-wrap {
    width: 100%; overflow: hidden; background: var(--panel);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 6px 0; margin-bottom: 20px;
}
.ticker-track {
    display: inline-block; white-space: nowrap;
    animation: ticker-scroll 40s linear infinite;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
}
.ticker-track:hover { animation-play-state: paused; }
.ticker-item { display: inline-block; padding: 0 28px; color: var(--text-dim); }
.ticker-item .tick-id { color: var(--risk-high); font-weight: 600; }
.ticker-item .tick-sep { color: var(--border); margin: 0 6px; }
@keyframes ticker-scroll {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* Hero stats bar -- the single "at a glance" moment before the detail list.
   One dominant number (HIGH-risk count) flanked by supporting context, so
   a viewer knows the headline before reading any individual card. */
.hero-stats {
    display: flex; gap: 0; margin-bottom: 24px;
    border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
}
.hero-stat {
    flex: 1; padding: 16px 20px; border-right: 1px solid var(--border);
}
.hero-stat:last-child { border-right: none; }
.hero-stat .stat-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 30px; font-weight: 700; line-height: 1;
}
.hero-stat .stat-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 6px;
}
.hero-stat.primary { background: rgba(229,72,77,0.08); }
.hero-stat.primary .stat-value { color: var(--risk-high); }
</style>
""", unsafe_allow_html=True)


def _risk_tier(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    return "LOW"


def _risk_color(tier: str) -> str:
    return {"HIGH": "#E5484D", "MEDIUM": "#F5A623", "LOW": "#3FB950"}[tier]


@st.cache_data
def load_data():
    candidates = json.load(open(os.path.join(DATA_DIR, "candidate_rings.json")))
    rings_truth = json.load(open(os.path.join(DATA_DIR, "rings.json")))
    accounts = pd.read_csv(os.path.join(DATA_DIR, "accounts.csv"))
    ml_scores = pd.read_csv(os.path.join(DATA_DIR, "ml_detector_scores.csv"))
    edges = pd.read_csv(os.path.join(DATA_DIR, "relational_edges.csv"))
    return candidates, rings_truth, accounts, ml_scores, edges


candidates, rings_truth, accounts, ml_scores, edges = load_data()
candidates_sorted = sorted(candidates, key=lambda c: c["ring_risk_score"], reverse=True)

# --- Header ---
st.markdown(f"""
<div class="sentinel-header">
    <span class="title">◆ SENTINEL RING</span>
    <span class="subtitle">Coordinated-abuse detection console</span>
    <span class="status-pill">{len(candidates)} CANDIDATES · {len(accounts)} ACCOUNTS SCANNED</span>
</div>
""", unsafe_allow_html=True)

# --- Scrolling alert ticker: top HIGH-risk candidates ---
high_risk = [c for c in candidates_sorted if _risk_tier(c["ring_risk_score"]) == "HIGH"][:12]
ticker_items = "".join(
    f'<span class="ticker-item">⚠ <span class="tick-id">{c["candidate_id"]}</span>'
    f'<span class="tick-sep">·</span>{c["size"]} accounts'
    f'<span class="tick-sep">·</span>risk {c["ring_risk_score"]:.3f}</span>'
    for c in high_risk
)
# duplicate the sequence once so the scroll loop is seamless (translateX -50%)
st.markdown(f"""
<div class="ticker-wrap">
    <div class="ticker-track">{ticker_items}{ticker_items}</div>
</div>
""", unsafe_allow_html=True)

# --- Hero stats: the headline numbers before anyone reads a single card ---
n_high = sum(1 for c in candidates if _risk_tier(c["ring_risk_score"]) == "HIGH")
n_medium = sum(1 for c in candidates if _risk_tier(c["ring_risk_score"]) == "MEDIUM")
total_flagged_accounts = len(set(a for c in candidates for a in c["member_accounts"]))
avg_risk = sum(c["ring_risk_score"] for c in candidates) / len(candidates)

st.markdown(f"""
<div class="hero-stats">
    <div class="hero-stat primary">
        <div class="stat-value">{n_high}</div>
        <div class="stat-label">High-risk rings</div>
    </div>
    <div class="hero-stat">
        <div class="stat-value">{n_medium}</div>
        <div class="stat-label">Medium-risk rings</div>
    </div>
    <div class="hero-stat">
        <div class="stat-value">{total_flagged_accounts}</div>
        <div class="stat-label">Accounts in flagged clusters</div>
    </div>
    <div class="hero-stat">
        <div class="stat-value">{avg_risk:.2f}</div>
        <div class="stat-label">Avg. candidate risk score</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar filters ---
with st.sidebar:
    st.markdown('<div class="mono" style="font-size:14px; margin-bottom:12px;">FILTERS</div>', unsafe_allow_html=True)
    min_risk = st.slider("Minimum risk score", 0.0, 1.0, 0.3, 0.05)
    min_size = st.slider("Minimum ring size", 2, 30, 2)
    st.markdown("---")

    # Groq API key: prefer the server-side env var (set when we deploy this
    # ourselves), but fall back to letting the person testing it paste their
    # own free key -- this is what unblocks a reviewer running the app
    # without our terminal's environment variable. Never written to disk or
    # committed anywhere; lives only in this session.
    st.markdown('<div class="mono" style="font-size:14px; margin-bottom:8px;">AGENT ACCESS</div>', unsafe_allow_html=True)
    if os.environ.get("GROQ_API_KEY"):
        st.markdown('<div style="font-size:11px; color:#3FB950;">✓ Groq API key configured (server)</div>', unsafe_allow_html=True)
        st.session_state["groq_api_key"] = os.environ.get("GROQ_API_KEY")
    else:
        user_key = st.text_input(
            "Groq API key",
            type="password",
            help="Free at console.groq.com/keys. Used only for this session, never stored or sent anywhere else.",
            key="groq_key_input",
        )
        if user_key:
            st.session_state["groq_api_key"] = user_key
            st.markdown('<div style="font-size:11px; color:#3FB950;">✓ Key set for this session</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:11px; color:#F5A623;">⚠ Needed to run live investigations — get a free key at console.groq.com/keys</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:12px; color:#8B949E;">Sentinel Ring combines transaction-level ML, '
        'relational + temporal graph signals, and a bounded AI investigator agent to surface '
        'coordinated fraud rings for human review.</div>',
        unsafe_allow_html=True,
    )

filtered = [c for c in candidates_sorted if c["ring_risk_score"] >= min_risk and c["size"] >= min_size]

col_list, col_detail = st.columns([1, 1.4])

with col_list:
    st.markdown(f'<div class="case-file-header">CANDIDATE RINGS ({len(filtered)})</div>', unsafe_allow_html=True)

    if "selected_candidate" not in st.session_state:
        st.session_state.selected_candidate = filtered[0]["candidate_id"] if filtered else None

    for c in filtered[:30]:
        tier = _risk_tier(c["ring_risk_score"])
        color = _risk_color(tier)
        attrs_html = "".join(f'<span class="attr-tag">{a}</span>' for a in c["shared_attributes_involved"])
        is_selected = c["candidate_id"] == st.session_state.selected_candidate

        card_html = f"""
        <div class="case-card risk-tier-{tier.lower()}" style="{'outline: 1px solid ' + color + ';' if is_selected else ''}">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span class="case-id">{c['candidate_id']}</span>
                <span class="mono" style="font-size:12px; color:{color}; font-weight:700;">{c['ring_risk_score']:.3f}</span>
            </div>
            <div class="case-meta">{c['size']} accounts · density {c['density_score']:.2f}</div>
            <div class="risk-score-bar-bg"><div class="risk-score-bar-fill" style="width:{c['ring_risk_score']*100:.0f}%; background:{color};"></div></div>
            <div>{attrs_html}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        if st.button(f"View details →", key=f"btn_{c['candidate_id']}", use_container_width=True):
            st.session_state.selected_candidate = c["candidate_id"]
            st.rerun()

with col_detail:
    selected_id = st.session_state.get("selected_candidate")
    if not selected_id:
        st.markdown('<div style="color:#8B949E;">No candidates match the current filters.</div>', unsafe_allow_html=True)
    else:
        candidate = next(c for c in candidates if c["candidate_id"] == selected_id)
        tier = _risk_tier(candidate["ring_risk_score"])
        color = _risk_color(tier)

        st.markdown(f'<div class="case-file-header">CASE DETAIL — {selected_id}</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="case-card risk-tier-{tier.lower()}">
            <span class="verdict-badge verdict-{tier}">{tier} RISK</span>
            <div style="margin-top:10px; font-size:13px; color:#8B949E;">
                {candidate['size']} accounts &nbsp;·&nbsp;
                ring_risk_score {candidate['ring_risk_score']:.3f} &nbsp;·&nbsp;
                density {candidate['density_score']:.2f} &nbsp;·&nbsp;
                shared attributes: {', '.join(candidate['shared_attributes_involved'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Network graph for this candidate ---
        st.markdown('<div class="case-file-header">CONNECTION GRAPH</div>', unsafe_allow_html=True)
        members = candidate["member_accounts"]
        sub_edges = edges[edges["account_a"].isin(members) & edges["account_b"].isin(members)].copy()

        # De-clutter: a dense ring can have many redundant edges between the
        # same pair (one per shared attribute) plus near-complete connectivity
        # at larger sizes -- rendering all of them produces an unreadable
        # "hairball" rather than legible evidence. Keep at most the top 2
        # edges per account pair (by burst_score, the strongest signal),
        # which preserves the real structure while cutting visual noise.
        sub_edges = (
            sub_edges.sort_values("burst_score", ascending=False)
            .groupby(["account_a", "account_b"], as_index=False)
            .head(2)
        )

        net = Network(height="320px", width="100%", bgcolor="#161B22", font_color="#E6E8EB", directed=False)

        # Outlier detection: a member connected to the ring by only 1 edge
        # (vs. the tightly-connected core) is a genuinely meaningful signal --
        # a loosely-attached account, worth a reviewer's specific attention --
        # not noise to hide. Compute degree within this candidate's subgraph
        # and visually flag low-degree nodes instead of rendering every node
        # identically.
        # Only meaningful for candidates with enough members to have a real
        # "core" vs. "fringe" distinction -- on a 2-3 node graph everyone is
        # trivially at the same degree, so outlier-flagging there is
        # meaningless (found via Day 13 edge-case testing: a 2-node candidate
        # flagged BOTH nodes as outliers, which is not a real signal).
        degree_count = {m: 0 for m in members}
        for _, row in sub_edges.iterrows():
            degree_count[row["account_a"]] = degree_count.get(row["account_a"], 0) + 1
            degree_count[row["account_b"]] = degree_count.get(row["account_b"], 0) + 1
        max_degree = max(degree_count.values()) if degree_count else 1
        outlier_threshold = max(1, max_degree * 0.25)  # <25% of the core's connectivity = outlier
        outliers_meaningful = len(members) >= 4

        for m in members:
            deg = degree_count.get(m, 0)
            is_outlier = outliers_meaningful and deg <= outlier_threshold
            net.add_node(
                m, label=m.replace("acc_", "#"),
                color="#F5A623" if is_outlier else "#58A6FF",
                size=12 if is_outlier else 16,
                borderWidth=2 if is_outlier else 1,
                borderWidthSelected=3,
                title=f"{m} — {deg} connection(s) within this ring" + (" (weakly attached)" if is_outlier else ""),
                font={"size": 13, "color": "#F5A623" if is_outlier else "#E6E8EB", "face": "IBM Plex Mono"},
            )
        attr_colors = {"device_id": "#E5484D", "payment_token": "#F5A623", "billing_address_hash": "#3FB950"}
        for _, row in sub_edges.iterrows():
            net.add_edge(row["account_a"], row["account_b"],
                         color=attr_colors.get(row["shared_attribute"], "#8B949E"), width=1.5, smooth=False)

        # Straight edges (smooth=False above) + a settled, non-overlapping
        # physics layout instead of PyVis's default tangled curve style,
        # which is what caused the unreadable hairball look.
        net.set_options("""
        {
          "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -80, "springLength": 120, "avoidOverlap": 1},
            "stabilization": {"iterations": 150}
          },
          "interaction": {"hover": true}
        }
        """)

        graph_path = os.path.join(DATA_DIR, f"_graph_{selected_id}.html")
        net.save_graph(graph_path)
        with open(graph_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=330)

        st.markdown(
            '<div style="font-size:11px; color:#8B949E; margin-top:4px;">'
            '<span style="color:#E5484D;">●</span> device &nbsp;'
            '<span style="color:#F5A623;">●</span> payment &nbsp;'
            '<span style="color:#3FB950;">●</span> address &nbsp;&nbsp;|&nbsp;&nbsp;'
            '<span style="color:#F5A623;">◆</span> weakly-attached outlier (hover node for detail)'
            '</div>', unsafe_allow_html=True
        )

        # --- AI Investigator ---
        st.markdown('<div class="case-file-header">AI INVESTIGATOR</div>', unsafe_allow_html=True)

        if st.button("▶ RUN INVESTIGATION", key=f"investigate_{selected_id}", use_container_width=True):
            active_key = st.session_state.get("groq_api_key") or os.environ.get("GROQ_API_KEY")
            if not active_key:
                st.error("No Groq API key available. Paste your free key in the sidebar under AGENT ACCESS (get one at console.groq.com/keys).")
            else:
                with st.spinner("Agent investigating — gathering evidence..."):
                    try:
                        result = investigate_ring(candidate, api_key=active_key)
                        st.session_state[f"result_{selected_id}"] = result
                    except Exception as e:
                        st.error(f"Investigation failed: {str(e)[:300]}")

        result = st.session_state.get(f"result_{selected_id}")
        if result:
            r_tier = result.get("risk_tier", "MEDIUM")
            r_color = _risk_color(r_tier)
            action = result.get("recommended_action", "MANUAL_REVIEW")
            st.markdown(f"""
            <div class="case-card" style="border-left-color:{r_color};">
                <span class="verdict-badge verdict-{r_tier}">{r_tier}</span>
                <span class="action-badge">{action.replace('_', ' ')}</span>
                <div style="margin-top:10px; font-size:13px; color:#8B949E; font-style:italic;">
                    {result.get('reason', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="case-file-header">EVIDENCE</div>', unsafe_allow_html=True)
            for ev in result.get("evidence", []):
                st.markdown(f'<div class="evidence-item">{ev}</div>', unsafe_allow_html=True)

            meta_cols = st.columns(2)
            meta_cols[0].markdown(
                f'<div style="font-size:11px; color:#8B949E;">Tool calls used: {result.get("tool_calls_used", "—")}/6</div>',
                unsafe_allow_html=True)
            meta_cols[1].markdown(
                f'<div style="font-size:11px; color:#8B949E;">Hit call cap: {result.get("hit_call_cap", False)}</div>',
                unsafe_allow_html=True)

            # --- Audit trail: full record of this investigation, for
            # compliance/review -- not just the final verdict. This is what
            # makes the recommendation defensible to a human reviewer. ---
            st.markdown('<div class="case-file-header">AUDIT TRAIL</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="case-card" style="border-left-color:#8B949E;">
                <div style="font-size:12px; font-family:'IBM Plex Mono',monospace; color:#8B949E; line-height:2;">
                    candidate_id: {selected_id}<br>
                    ring_risk_score (precomputed, Day 7): {candidate['ring_risk_score']:.3f}<br>
                    tool_calls_used: {result.get('tool_calls_used', '—')} / 6 (hard cap)<br>
                    hit_call_cap: {result.get('hit_call_cap', False)}<br>
                    final_risk_tier: {result.get('risk_tier', '—')}<br>
                    final_recommended_action: {result.get('recommended_action', '—')}<br>
                    action_type: RECOMMENDATION ONLY — no irreversible action taken (Rules.md #2)<br>
                    human_review_required: {"YES" if result.get('recommended_action') != "AUTO_MONITOR" else "NO (low-touch monitoring)"}
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- False-positive / cost analysis: honest system-level performance,
# not just individual-case results. Pulled from Day 8's real evaluation
# against known ground truth -- reported as-is, including the limitations. ---
st.markdown('<div class="case-file-header" style="margin-top:32px;">SYSTEM PERFORMANCE — FALSE POSITIVE / COST ANALYSIS</div>', unsafe_allow_html=True)

eval_path = os.path.join(DATA_DIR, "ring_evaluation.json")
if os.path.exists(eval_path):
    eval_data = json.load(open(eval_path))
    recall = eval_data["recall"]
    precision = eval_data["precision"]
    n_true_rings = len(eval_data["ring_results"])
    n_found = sum(1 for r in eval_data["ring_results"] if r["found"])
    n_candidates_total = len(eval_data["candidate_results"])
    n_false_positives = sum(1 for c in eval_data["candidate_results"] if not c["is_true_positive"])

    perf_cols = st.columns(4)
    perf_cols[0].markdown(f"""
    <div class="hero-stat" style="border:1px solid var(--border); border-radius:6px;">
        <div class="stat-value" style="color:#3FB950;">{recall:.0%}</div>
        <div class="stat-label">Recall — real rings found ({n_found}/{n_true_rings})</div>
    </div>""", unsafe_allow_html=True)
    perf_cols[1].markdown(f"""
    <div class="hero-stat" style="border:1px solid var(--border); border-radius:6px;">
        <div class="stat-value" style="color:#F5A623;">{precision:.1%}</div>
        <div class="stat-label">Raw precision (whole candidate list)</div>
    </div>""", unsafe_allow_html=True)
    perf_cols[2].markdown(f"""
    <div class="hero-stat" style="border:1px solid var(--border); border-radius:6px;">
        <div class="stat-value" style="color:#E5484D;">{n_false_positives}</div>
        <div class="stat-label">False positives (out of {n_candidates_total} candidates)</div>
    </div>""", unsafe_allow_html=True)
    perf_cols[3].markdown(f"""
    <div class="hero-stat" style="border:1px solid var(--border); border-radius:6px;">
        <div class="stat-value" style="color:#58A6FF;">4-7x</div>
        <div class="stat-label">Precision lift in top-20 vs. flat list</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="case-card" style="border-left-color:#58A6FF; margin-top:12px;">'
        '<div style="font-size:11px; color:#58A6FF; font-family:\'IBM Plex Mono\',monospace; '
        'text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Why raw precision looks low</div>'
        '<div style="font-size:12px; color:#8B949E; line-height:1.6;">'
        'Louvain detection surfaces every dense cluster — including real neighborhoods, not just fraud. '
        'That\'s intentional: risk scoring and the AI Investigator triage this list down to what\'s worth reviewing. '
        'A false positive here costs a reviewer a few minutes — never a wrongful account action, '
        'since the system only recommends, never auto-blocks.'
        '</div></div>', unsafe_allow_html=True
    )
else:
    st.markdown('<div style="color:#8B949E; font-size:12px;">Run src/network/evaluate_rings.py to generate evaluation data.</div>', unsafe_allow_html=True)
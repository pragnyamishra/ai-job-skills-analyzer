"""
AI Job Skills Analyzer
Streamlit Frontend
"""

import os
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()
import os
for key in ["GROQ_API_KEY", "RAPID_API_KEY", "PINECONE_API_KEY", 
            "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"]:
    if key not in os.environ:
        try:
            os.environ[key] = st.secrets[key]
        except Exception:
            pass
from resume_parser import parse_resume
from agent import run_agent
from monitoring import get_call_history

st.set_page_config(
    page_title="AI Job Skills Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS + JS ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container { padding-top: 1rem; max-width: 1100px; }

    /* ── Header ── */
    .app-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(59,130,246,0.06) 100%);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.25rem;
    }
    .app-header h1 {
        font-size: 1.35rem; font-weight: 800; color: #F1F5F9;
        margin: 0; letter-spacing: -0.03em;
    }
    .app-header p {
        font-size: 0.82rem; color: #64748B; margin: 0.25rem 0 0 0;
    }

    /* ── Context line ── */
    .context-line { font-size: 0.82rem; color: #64748B; margin-bottom: 0.15rem; }
    .context-line strong { color: #E2E8F0; }

    /* ── Metric cards ── */
    .metrics-grid {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem; margin: 0.75rem 0 1.25rem 0;
    }
    .m-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px; padding: 1rem 1.1rem; transition: border-color 0.2s;
    }
    .m-card:hover { border-color: rgba(99,102,241,0.35); }
    .m-card .m-label {
        font-size: 0.68rem; font-weight: 600; color: #64748B;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem;
    }
    .m-card .m-value { font-size: 1.6rem; font-weight: 700; color: #F1F5F9; line-height: 1; }
    .m-card .m-value.accent { color: #818CF8; }
    .m-card .m-value.green  { color: #4ADE80; }
    .m-card .m-value.red    { color: #F87171; }

    /* ── Section headers ── */
    .sec-head {
        font-size: 0.72rem; font-weight: 700; color: #6366F1;
        text-transform: uppercase; letter-spacing: 0.08em; margin: 1.25rem 0 0.65rem 0;
    }

    /* ── Skill pills ── */
    .pill-wrap { display: flex; flex-wrap: wrap; gap: 6px; }
    .pill-match {
        display: inline-block; background: rgba(74,222,128,0.10);
        border: 1px solid rgba(74,222,128,0.25); color: #4ADE80;
        padding: 0.28rem 0.7rem; border-radius: 6px; font-size: 0.78rem; font-weight: 500;
    }
    .pill-gap {
        display: inline-block; background: rgba(248,113,113,0.08);
        border: 1px solid rgba(248,113,113,0.25); color: #F87171;
        padding: 0.28rem 0.7rem; border-radius: 6px; font-size: 0.78rem; font-weight: 500;
    }

    /* ── Gap table ── */
    .gap-table { margin-top: 0.5rem; }
    .gap-header, .gap-item {
        display: grid; grid-template-columns: 28px 1fr 130px 65px 60px;
        align-items: center; padding: 0.5rem 0.4rem;
    }
    .gap-header {
        border-bottom: 1px solid rgba(255,255,255,0.08);
        font-size: 0.68rem; font-weight: 600; color: #64748B;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .gap-item { border-bottom: 1px solid rgba(255,255,255,0.03); }
    .gap-item:hover { background: rgba(255,255,255,0.02); }
    .gap-item .g-num  { font-size: 0.78rem; color: #475569; font-weight: 500; }
    .gap-item .g-name { font-size: 0.85rem; color: #E2E8F0; font-weight: 500; }
    .gap-item .g-cat  { font-size: 0.78rem; color: #64748B; }
    .gap-item .g-freq { font-size: 0.78rem; color: #94A3B8; text-align: right; }
    .badge-high {
        background: rgba(248,113,113,0.15); color: #F87171;
        padding: 2px 7px; border-radius: 4px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    }
    .badge-med {
        background: rgba(251,191,36,0.15); color: #FBBF24;
        padding: 2px 7px; border-radius: 4px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    }
    .badge-low {
        background: rgba(148,163,184,0.12); color: #94A3B8;
        padding: 2px 7px; border-radius: 4px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    }

    /* ── Week cards ── */
    .week-card {
        background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06);
        border-left: 3px solid #6366F1; border-radius: 10px;
        padding: 1.1rem 1.35rem; margin-bottom: 0.65rem;
    }
    .week-card:hover { border-color: rgba(99,102,241,0.4); border-left-color: #818CF8; }
    .week-num { font-size: 0.65rem; font-weight: 700; color: #6366F1; text-transform: uppercase; letter-spacing: 0.06em; }
    .week-theme { font-size: 0.95rem; font-weight: 600; color: #F1F5F9; margin: 0.15rem 0 0.1rem 0; }
    .week-focus { font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.5rem; }
    .week-detail { font-size: 0.82rem; color: #CBD5E1; line-height: 1.5; margin: 0.12rem 0; }
    .week-detail a { color: #818CF8; text-decoration: none; }
    .week-detail a:hover { text-decoration: underline; }
    .week-sublabel {
        font-size: 0.67rem; font-weight: 700; color: #6366F1;
        text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.55rem; margin-bottom: 0.15rem;
    }

    /* ── Capstone ── */
    .capstone {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(59,130,246,0.04) 100%);
        border: 1px solid rgba(99,102,241,0.2); border-radius: 10px; padding: 1.25rem 1.5rem;
    }
    .capstone h4 { font-size: 1rem; font-weight: 700; color: #F1F5F9; margin: 0 0 0.4rem 0; }
    .capstone p { font-size: 0.85rem; color: #CBD5E1; line-height: 1.5; margin: 0; }
    .capstone .cap-skills { font-size: 0.78rem; color: #94A3B8; margin-top: 0.5rem; }

    /* ── Monitor log ── */
    .mon-row {
        display: grid; grid-template-columns: 42px 1fr 90px 70px;
        padding: 0.4rem 0.4rem; border-bottom: 1px solid rgba(255,255,255,0.03);
        font-size: 0.78rem; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; color: #94A3B8;
    }
    .mon-ok   { color: #4ADE80; font-weight: 600; }
    .mon-fail { color: #F87171; font-weight: 600; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.05); }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.02);
        border: 1px dashed rgba(255,255,255,0.10);
        border-radius: 8px;
    }
    .sidebar-label {
        font-size: 0.68rem; font-weight: 700; color: #64748B;
        text-transform: uppercase; letter-spacing: 0.06em; margin: 0.75rem 0 0.3rem 0;
    }
    .status-bar {
        display: flex; gap: 0.85rem; margin: 0.2rem 0 0.5rem 0; font-size: 0.75rem; color: #94A3B8;
    }
    .dot-ok   { color: #4ADE80; }
    .dot-miss { color: #F87171; }

    /* ── Empty state ── */
    .step-box {
        background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; padding: 1.35rem; text-align: center; height: 100%;
    }
    .step-box .step-num {
        font-size: 0.65rem; font-weight: 700; color: #6366F1;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;
    }
    .step-box h3 { font-size: 0.95rem; font-weight: 600; color: #F1F5F9; margin: 0.3rem 0 0.2rem 0; }
    .step-box p { font-size: 0.82rem; color: #64748B; margin: 0; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Plotly modebar ── */
    .js-plotly-plot .plotly .modebar { display: none !important; }

    /* ── Legend ── */
    .chart-legend { font-size: 0.75rem; color: #64748B; margin-top: -0.5rem; margin-bottom: 0.5rem; }
    </style>

    <script>
    // Nuclear cleanup: fix Material Symbols ligature text showing as raw text
    function cleanMaterialIcons() {
        // Known ligature patterns that Streamlit uses for icons
        const ligaturePatterns = [
            'keyboard_double_arrow', 'arrow_forward', 'arrow_back',
            'arrow_drop', 'expand_more', 'expand_less', 'chevron',
            'upload', 'close', 'menu', 'more_vert', 'more_horiz',
            'search', 'check', 'add', 'remove', 'delete', 'edit',
            'visibility', 'settings', 'info', 'warning', 'error',
            'navigate', 'first_page', 'last_page', 'unfold', 'sort',
            'filter', 'refresh', 'sync', 'cloud', 'file', 'folder',
            'download', 'open_in', 'launch', 'link', 'share',
            'play_arrow', 'pause', 'stop', 'skip', 'replay'
        ];

        document.querySelectorAll('span, button').forEach(function(el) {
            var text = (el.textContent || '').trim().toLowerCase();
            // Skip if element has children that are not text
            if (el.children.length > 0 && el.querySelector('span, div, p, a')) return;
            // Skip if text is too long (real content, not a ligature)
            if (text.length > 40) return;

            var isLigature = false;
            for (var i = 0; i < ligaturePatterns.length; i++) {
                if (text.indexOf(ligaturePatterns[i]) !== -1) {
                    isLigature = true;
                    break;
                }
            }

            if (isLigature) {
                // Check if it's a button — replace text instead of hiding
                if (el.tagName === 'BUTTON') {
                    if (text.indexOf('upload') !== -1) {
                        el.textContent = 'Browse files';
                    } else {
                        el.style.fontSize = '0px';
                        el.style.width = '20px';
                        el.style.height = '20px';
                        el.style.overflow = 'hidden';
                    }
                } else {
                    el.style.fontSize = '0px';
                    el.style.width = '0px';
                    el.style.overflow = 'hidden';
                    el.style.display = 'none';
                }
            }
        });
    }

    // Run multiple times to catch dynamically rendered elements
    setTimeout(cleanMaterialIcons, 300);
    setTimeout(cleanMaterialIcons, 800);
    setTimeout(cleanMaterialIcons, 1500);
    setTimeout(cleanMaterialIcons, 3000);

    // Watch for DOM changes (Streamlit re-renders often)
    var observer = new MutationObserver(function() {
        setTimeout(cleanMaterialIcons, 100);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <h1>AI Job Skills Analyzer</h1>
        <p>Real-time market intelligence. Personalized gap analysis. Actionable learning plans.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    groq_ok  = bool(os.getenv("GROQ_API_KEY"))
    rapid_ok = bool(os.getenv("RAPID_API_KEY"))
    pine_ok  = bool(os.getenv("PINECONE_API_KEY"))

    st.markdown(
        f"""<div class="status-bar">
            <span><span class="{"dot-ok" if groq_ok else "dot-miss"}">●</span> Groq</span>
            <span><span class="{"dot-ok" if rapid_ok else "dot-miss"}">●</span> JSearch</span>
            <span><span class="{"dot-ok" if pine_ok else "dot-miss"}">●</span> Pinecone</span>
        </div>""",
        unsafe_allow_html=True,
    )

    if not all([groq_ok, rapid_ok]):
        st.warning("Required API keys missing in .env")

    job_title = st.text_input("Target Role", placeholder="e.g. Data Engineer, ML Engineer")
    location  = st.text_input("Location", value="United States", placeholder="e.g. Remote, New York")

    resume_file = st.file_uploader(
        "Resume", type=["pdf", "docx", "txt"],
        help="PDF, DOCX, or TXT", label_visibility="visible",
    )
    resume_text_input = st.text_area(
        "Or paste resume text", height=80,
        placeholder="Paste your resume here...",
    )

    run_button = st.button("Analyze", type="primary", use_container_width=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def render_metrics(jobs, skills, matched, gaps):
    st.markdown(
        f"""
        <div class="metrics-grid">
            <div class="m-card"><div class="m-label">Jobs Analyzed</div><div class="m-value accent">{jobs}</div></div>
            <div class="m-card"><div class="m-label">Skills in Demand</div><div class="m-value">{skills}</div></div>
            <div class="m-card"><div class="m-label">Skills You Have</div><div class="m-value green">{matched}</div></div>
            <div class="m-card"><div class="m-label">Skills to Learn</div><div class="m-value red">{gaps}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pills(skills, css):
    html = '<div class="pill-wrap">'
    html += "".join(
        f'<span class="{css}">{s.get("skill", s) if isinstance(s, dict) else s}</span>'
        for s in skills
    )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_donut(score):
    if score >= 70:
        ring_color = "#4ADE80"
    elif score >= 40:
        ring_color = "#FBBF24"
    else:
        ring_color = "#818CF8"

    fig = go.Figure(data=[go.Pie(
        values=[score, 100 - score], labels=["Matched", "Gap"], hole=0.75,
        marker=dict(colors=[ring_color, "rgba(255,255,255,0.04)"]),
        textinfo="none", hovertemplate="%{label}: %{value}%<extra></extra>", rotation=90,
    )])
    fig.update_layout(
        showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=190,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{score}%</b><br><span style='font-size:10px;color:#64748B'>match</span>",
            x=0.5, y=0.5, font_size=26, font_color="#F1F5F9", showarrow=False,
        )],
    )
    return fig


def render_bar(market_skills, matched_names=None):
    matched_names = matched_names or set()
    top = market_skills[:14]
    names  = [s["skill"] for s in top][::-1]
    counts = [s.get("job_count", 0) for s in top][::-1]
    colors = []
    for s in top[::-1]:
        if s["skill"].lower() in matched_names:
            colors.append("#4ADE80")
        elif s.get("importance") == "High":
            colors.append("#6366F1")
        else:
            colors.append("#334155")

    fig = go.Figure(data=[go.Bar(
        x=counts, y=names, orientation="h", marker_color=colors,
        hovertemplate="%{y}: %{x} postings<extra></extra>",
    )])
    fig.update_layout(
        height=max(300, len(top) * 30), margin=dict(t=0, b=0, l=0, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Mentions", gridcolor="rgba(255,255,255,0.04)", title_font_size=10, tickfont_size=10),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont_size=11),
        font=dict(color="#CBD5E1", size=11), bargap=0.25,
    )
    return fig


def render_gap_table(gaps):
    header = """<div class="gap-table">
        <div class="gap-header"><div>#</div><div>Skill</div><div>Category</div><div>Freq</div><div>Priority</div></div>"""
    rows = ""
    for i, g in enumerate(gaps, 1):
        imp = g.get("importance", "Medium")
        badge_class = "badge-high" if imp == "High" else "badge-med" if imp == "Medium" else "badge-low"
        badge = f'<span class="{badge_class}">{imp}</span>'
        freq = g.get("frequency_pct", "")
        freq_str = f"{freq}%" if freq else ""
        rows += f"""<div class="gap-item">
            <div class="g-num">{i}</div><div class="g-name">{g['skill']}</div>
            <div class="g-cat">{g.get('category', 'Technical')}</div>
            <div class="g-freq">{freq_str}</div><div>{badge}</div>
        </div>"""
    st.markdown(header + rows + "</div>", unsafe_allow_html=True)


def render_week(week):
    resources_html = ""
    for r in week.get("resources", []):
        url, title, rtype = r.get("url", ""), r.get("title", "Resource"), r.get("type", "")
        if url and url.startswith("http"):
            resources_html += f'<div class="week-detail"><a href="{url}" target="_blank">{title}</a> ({rtype})</div>'
        else:
            resources_html += f'<div class="week-detail">{title} ({rtype})</div>'
    project = week.get("project", "")
    project_html = f'<div class="week-sublabel">Project</div><div class="week-detail">{project}</div>' if project else ""
    outcome = week.get("outcome", "")
    outcome_html = f'<div class="week-sublabel">Outcome</div><div class="week-detail">{outcome}</div>' if outcome else ""
    focus = ", ".join(week.get("focus_skills", []))
    st.markdown(f"""
        <div class="week-card">
            <div class="week-num">Week {week.get('week', '?')}</div>
            <div class="week-theme">{week.get('theme', '')}</div>
            <div class="week-focus">Focus: {focus}</div>
            {resources_html}{project_html}{outcome_html}
        </div>""", unsafe_allow_html=True)

# ── Main flow ────────────────────────────────────────────────────────────────

if run_button:
    if not job_title.strip():
        st.error("Please enter a target role.")
        st.stop()

    resume_text = ""
    if resume_file:
        resume_text = parse_resume(resume_file)
    elif resume_text_input.strip():
        resume_text = resume_text_input.strip()

    if not resume_text:
        st.warning("No resume provided. Running market analysis only.")

    with st.spinner("Analyzing..."):
        result = run_agent(job_title.strip(), location.strip(), resume_text)

    if result.get("error"):
        st.error(result["error"])
        st.stop()

    st.session_state["result"]   = result
    st.session_state["analyzed"] = True

# ── Results ──────────────────────────────────────────────────────────────────

if st.session_state.get("analyzed") and st.session_state.get("result"):
    result        = st.session_state["result"]
    gap_report    = result.get("gap_report") or {}
    market_skills = result.get("market_skills", [])
    plan          = result.get("learning_plan")
    jobs          = result.get("job_listings", [])

    st.markdown(
        f'<div class="context-line">'
        f'Analysis for <strong>{result["job_title"]}</strong> '
        f'based on {len(jobs)} live job postings</div>',
        unsafe_allow_html=True,
        
    )
    import streamlit.components.v1 as components
    components.html("""
    <script>
    function fixParent() {
        var doc = window.parent.document;
        if (!doc) return;
        doc.querySelectorAll('span, button').forEach(function(el) {
            var t = (el.textContent || '').trim();
            if (t === 'uploadUpload' || t === 'upload_fileUpload') {
                el.textContent = 'Upload';
            }
            if (/^(keyboard_double_arrow|arrow_forward_ios|expand_more|expand_less|unfold_more|unfold_less|close|more_vert)/.test(t)) {
                el.style.fontSize = '0px';
                el.style.overflow = 'hidden';
                el.style.width = '0px';
                el.style.display = 'none';
            }
        });
    }
    setInterval(fixParent, 500);
    </script>
    """, height=0)

    render_metrics(
        len(jobs), len(market_skills),
        gap_report.get("total_matched", 0), gap_report.get("total_gaps", 0),
    )

    matched_list  = gap_report.get("matched_skills", [])
    matched_names = {s.get("skill", "").lower() for s in matched_list if isinstance(s, dict)}

    col_chart, col_score = st.columns([3, 2])

    with col_chart:
        st.markdown('<div class="sec-head">Market Skill Demand</div>', unsafe_allow_html=True)
        st.plotly_chart(render_bar(market_skills, matched_names), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="chart-legend">🟢 You have it · 🟣 High demand · ⬛ Lower demand</div>', unsafe_allow_html=True)

    with col_score:
        if gap_report.get("match_score") is not None:
            st.markdown('<div class="sec-head">Your Match Score</div>', unsafe_allow_html=True)
            st.plotly_chart(render_donut(gap_report["match_score"]), use_container_width=True, config={"displayModeBar": False})
            if matched_list:
                st.markdown('<div class="sidebar-label">Skills You Have</div>', unsafe_allow_html=True)
                render_pills(matched_list, "pill-match")

    gaps = gap_report.get("skill_gaps", [])
    if gaps:
        st.markdown('<div class="sec-head">Skill Gaps</div>', unsafe_allow_html=True)
        render_pills(gaps, "pill-gap")
        with st.expander("📋 Detailed breakdown"):
            render_gap_table(gaps)

    if plan:
        st.markdown('<div class="sec-head">Personalized Learning Plan</div>', unsafe_allow_html=True)
        for week in plan.get("weeks", []):
            render_week(week)

        capstone = plan.get("capstone_project")
        if capstone:
            st.markdown('<div class="sec-head">Capstone Project</div>', unsafe_allow_html=True)
            if isinstance(capstone, dict):
                skills_str = ", ".join(capstone.get("skills_covered", []))
                st.markdown(f"""
                    <div class="capstone">
                        <h4>{capstone.get('title', '')}</h4>
                        <p>{capstone.get('description', '')}</p>
                        <div class="cap-skills">Skills covered: {skills_str}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(str(capstone))

    with st.expander("📊 API Monitoring"):
        history = get_call_history()
        if history:
            total_tokens = sum(c.get("total_tokens", 0) for c in history)
            avg_latency  = sum(c.get("latency_ms", 0) for c in history) / len(history)
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1: st.metric("API Calls", len(history))
            with mcol2: st.metric("Tokens Used", f"{total_tokens:,}")
            with mcol3: st.metric("Avg Latency", f"{avg_latency:.0f}ms")
            for c in history:
                ok = c.get("success", True)
                st.markdown(f"""<div class="mon-row">
                    <span class="{"mon-ok" if ok else "mon-fail"}">{"OK" if ok else "FAIL"}</span>
                    <span>{c.get('function', '?')}</span>
                    <span>{c.get('total_tokens', 0)} tokens</span>
                    <span>{c.get('latency_ms', 0):.0f}ms</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No API calls recorded yet.")

# ── Empty state ──────────────────────────────────────────────────────────────

if not st.session_state.get("analyzed"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="step-box">
            <div class="step-num">Step 1</div><h3>Upload Resume</h3><p>PDF, DOCX, or paste text</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="step-box">
            <div class="step-num">Step 2</div><h3>Pick Your Target Role</h3><p>Data Engineer, ML Engineer, etc.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="step-box">
            <div class="step-num">Step 3</div><h3>Get Your Plan</h3><p>Gap report and learning path</p>
        </div>""", unsafe_allow_html=True)

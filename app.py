"""
ShortlistIQ — Engineering Resume Intelligence & ATS Shortlister
Empowering final-year engineering students & early-career devs to beat ATS filters and land tech jobs.
"""

import os
import uuid
import time
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

# Import core modules
from core.database import (
    init_db, log_event, save_feedback, 
    get_analytics_metrics, get_recent_feedback, get_event_timeline
)
from core.parser import extract_text_from_pdf, extract_bullet_points, PDFParsingError
from core.analyzer import (
    analyze_with_gemini, recalculate_score, compute_tfidf_similarity, 
    get_missing_and_matched_keywords
)
from core.sample_data import SAMPLE_JOB_DESCRIPTIONS, DEMO_RESUME_TEXT

# --- Page Configuration ---
st.set_page_config(
    page_title="ShortlistIQ — Engineering Resume Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Database ---
init_db()

# --- Initialize Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    log_event(st.session_state.session_id, "session_start")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "working_resume_text" not in st.session_state:
    st.session_state.working_resume_text = ""

if "initial_score" not in st.session_state:
    st.session_state.initial_score = 0

if "current_score" not in st.session_state:
    st.session_state.current_score = 0

if "accepted_rewrites" not in st.session_state:
    st.session_state.accepted_rewrites = set()

if "rejected_rewrites" not in st.session_state:
    st.session_state.rejected_rewrites = set()

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# --- Secrets & API Key Resolution ---
def get_secret(key: str, default: str = "") -> str:
    """Safely retrieves keys from st.secrets, os.environ, or default."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

api_key = get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")
admin_password = get_secret("ADMIN_PASSWORD", "admin")

# --- Custom CSS for Polished Tech UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366F1 0%, #38BDF8 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #6366F1;
        transform: translateY(-2px);
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
    }
    .badge-missing {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-matched {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-category {
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .fix-card {
        background: #131D2E;
        border-left: 4px solid #6366F1;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .fix-card.priority-1 { border-left-color: #EF4444; }
    .fix-card.priority-2 { border-left-color: #F59E0B; }
    .fix-card.priority-3 { border-left-color: #10B981; }
    
    .rewrite-box {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
    }
    
    .score-badge {
        font-size: 1.8rem;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 10px;
        display: inline-block;
    }
    
    .score-high { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
    .score-mid { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }
    .score-low { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444; }
    
    .diff-old {
        color: #F87171;
        background: rgba(239, 68, 68, 0.1);
        padding: 6px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
    }
    .diff-new {
        color: #34D399;
        background: rgba(16, 185, 129, 0.1);
        padding: 6px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: App Navigation & Controls ---
with st.sidebar:
    st.markdown("### 🎯 **ShortlistIQ**")
    st.caption("Bridging Engineering Resumes to Tech Shortlists")
    
    nav_mode = st.radio(
        "Navigation",
        ["🚀 Resume Analyzer", "✍️ Rewrite Studio & Live Score", "📊 Product Analytics (Admin)"],
        index=0
    )
    
    st.divider()
    
    # Privacy Badge
    st.markdown("""
    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 8px; padding: 10px; font-size: 0.8rem; color: #94A3B8;">
        🔒 <b>100% Privacy Guarantee:</b> Resumes are processed in memory and never stored in the database.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("#### 📌 Quick Shortcuts")
    if st.button("🔄 Reset Workspace", use_container_width=True):
        st.session_state.analysis_result = None
        st.session_state.resume_text = ""
        st.session_state.working_resume_text = ""
        st.session_state.initial_score = 0
        st.session_state.current_score = 0
        st.session_state.accepted_rewrites = set()
        st.session_state.rejected_rewrites = set()
        st.session_state.feedback_submitted = False
        st.rerun()

# --- Helper Gauge Plot ---
def create_gauge_chart(score: int, title: str = "ATS Match Score") -> go.Figure:
    """Generates an aesthetic gauge chart for match score visualization."""
    bar_color = "#10B981" if score >= 75 else ("#F59E0B" if score >= 50 else "#EF4444")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#F8FAFC', 'family': 'Inter'}},
        number={'suffix': "%", 'font': {'size': 44, 'color': '#F8FAFC', 'family': 'Inter', 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "#1E293B",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
            ],
            'threshold': {
                'line': {'color': "#38BDF8", 'width': 3},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#F8FAFC"},
        height=240,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig

# ==========================================
# PAGE 1: RESUME ANALYZER
# ==========================================
if nav_mode == "🚀 Resume Analyzer":
    # Header Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">ShortlistIQ — Engineering Resume Analyzer</div>
        <div class="hero-subtitle">
            Don't let ATS algorithms filter your potential. Get instant match scores, missing tech keywords, weak section audits, and metric-driven bullet point rewrites in under 5 seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_input1, col_input2 = st.columns([1, 1], gap="medium")
    
    with col_input1:
        st.markdown("### 📄 **Step 1: Upload Engineering Resume**")
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF format, max 10MB)",
            type=["pdf"],
            help="Upload your latest PDF resume. ATS parses selectable text, not flattened graphics."
        )
        
        # 1-Click Demo Option
        st.markdown("— *or try a sample profile* —")
        if st.button("⚡ Load Demo CS Fresher Resume", help="Loads a realistic Computer Science fresher resume for quick testing"):
            st.session_state.resume_text = DEMO_RESUME_TEXT
            st.session_state.working_resume_text = DEMO_RESUME_TEXT
            st.success("Loaded Demo CS Resume! Now select or paste a Job Description.")

    with col_input2:
        st.markdown("### 🎯 **Step 2: Target Job Description**")
        jd_choice = st.selectbox(
            "Select a pre-loaded tech role or paste custom JD:",
            ["-- Paste Custom Job Description --"] + list(SAMPLE_JOB_DESCRIPTIONS.keys())
        )
        
        if jd_choice != "-- Paste Custom Job Description --":
            jd_input = st.text_area(
                "Job Description Content:",
                value=SAMPLE_JOB_DESCRIPTIONS[jd_choice],
                height=180
            )
        else:
            jd_input = st.text_area(
                "Paste the Job Description here:",
                value=st.session_state.jd_text,
                placeholder="Paste full tech job requirements, responsibilities, and required qualifications...",
                height=180
            )
    
    # Trigger Analysis Button
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze & Generate Shortlist Report", type="primary", use_container_width=True)
    
    if analyze_btn:
        # Check inputs
        resume_content = ""
        is_pdf_upload = False
        
        if uploaded_file is not None:
            is_pdf_upload = True
            try:
                with st.spinner("🔍 Parsing and extracting PDF text..."):
                    resume_content, meta = extract_text_from_pdf(uploaded_file)
                    st.session_state.resume_text = resume_content
                    st.session_state.working_resume_text = resume_content
            except PDFParsingError as pe:
                st.error(str(pe))
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error while reading PDF: {e}")
                st.stop()
        elif st.session_state.resume_text:
            resume_content = st.session_state.resume_text
        else:
            st.warning("⚠️ Please upload a PDF resume or click 'Load Demo CS Fresher Resume' to proceed.")
            st.stop()
            
        if not jd_input.strip() or len(jd_input.strip()) < 30:
            st.warning("⚠️ Please provide a valid Job Description with at least 30 characters.")
            st.stop()
            
        st.session_state.jd_text = jd_input
        
        # Log upload event (strictly anonymous, no resume text)
        log_event(st.session_state.session_id, "upload", {
            "is_pdf": is_pdf_upload,
            "resume_length": len(resume_content),
            "jd_length": len(jd_input)
        })
        
        # Run Intelligence Engine
        with st.spinner("⚡ Running TF-IDF Vectorization & Gemini Analysis (< 5s)..."):
            start_proc = time.time()
            extracted_bullets = extract_bullet_points(resume_content)
            
            result = analyze_with_gemini(
                resume_text=resume_content,
                jd_text=jd_input,
                extracted_bullets=extracted_bullets,
                api_key=api_key
            )
            elapsed = time.time() - start_proc
            
            st.session_state.analysis_result = result
            st.session_state.initial_score = result.get("match_score", 60)
            st.session_state.current_score = result.get("match_score", 60)
            st.session_state.accepted_rewrites = set()
            st.session_state.rejected_rewrites = set()
            
            # Log result viewed event
            log_event(st.session_state.session_id, "result_viewed", {
                "initial_score": result.get("match_score", 60),
                "elapsed_seconds": round(elapsed, 2),
                "is_ai": result.get("is_ai_generated", False)
            })
            
            st.success(f"✨ Analysis completed in {round(elapsed, 2)}s! Review your report below.")

    # --- Render Results ---
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        score = st.session_state.current_score
        
        st.divider()
        
        # --- Top Summary Row ---
        r1_col1, r1_col2 = st.columns([1, 2], gap="large")
        
        with r1_col1:
            st.plotly_chart(create_gauge_chart(score, "Shortlist Match Score"), use_container_width=True)
            
            # Score Status Description
            if score >= 75:
                st.markdown("""
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; padding: 12px; border-radius: 8px; text-align: center;">
                    <b style="color: #34D399;">🌟 Strong Shortlist Candidate</b><br>
                    <span style="font-size: 0.85rem; color: #CBD5E1;">High keyword alignment and quantifiable metrics. Low probability of automated ATS rejection.</span>
                </div>
                """, unsafe_allow_html=True)
            elif score >= 50:
                st.markdown("""
                <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; padding: 12px; border-radius: 8px; text-align: center;">
                    <b style="color: #FBBF24;">⚠️ Moderate Match (At Risk)</b><br>
                    <span style="font-size: 0.85rem; color: #CBD5E1;">Missing critical stack keywords or metrics. Implementing top fixes will push you into the top 15% tier.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #EF4444; padding: 12px; border-radius: 8px; text-align: center;">
                    <b style="color: #F87171;">🛑 High Rejection Risk</b><br>
                    <span style="font-size: 0.85rem; color: #CBD5E1;">Significant mismatch in tech stack and project depth. Apply the recommended rewrites and missing keywords.</span>
                </div>
                """, unsafe_allow_html=True)

        with r1_col2:
            st.markdown("### 📊 **Score Breakdown & Dimensions**")
            breakdown = res.get("score_breakdown", {})
            kw_match = breakdown.get("keyword_match", 65)
            struct_qual = breakdown.get("structural_quality", 70)
            tech_depth = breakdown.get("technical_depth", 60)
            
            b_c1, b_c2, b_c3 = st.columns(3)
            with b_c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #94A3B8; font-size: 0.85rem;">Keyword Overlap</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #38BDF8;">{kw_match}%</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Core stack match</div>
                </div>
                """, unsafe_allow_html=True)
            with b_c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #94A3B8; font-size: 0.85rem;">Project & Metric Impact</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #818CF8;">{struct_qual}%</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Quantifiable XYZ metrics</div>
                </div>
                """, unsafe_allow_html=True)
            with b_c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #94A3B8; font-size: 0.85rem;">Technical Depth</div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #34D399;">{tech_depth}%</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Architecture & Tools</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🏆 **Top 3 Prioritized Action Items**")
            top_fixes = res.get("top_fixes", [])
            for fix in top_fixes:
                p = fix.get("priority", 1)
                t = fix.get("title", "")
                a = fix.get("action", "")
                st.markdown(f"""
                <div class="fix-card priority-{p}">
                    <b style="color: #F8FAFC;">#{p} {t}</b>
                    <div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 4px;">{a}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # --- Middle Row: Missing vs Matched Keywords & Weak Sections ---
        col_kw, col_weak = st.columns([1, 1], gap="large")
        
        with col_kw:
            st.markdown("### 🔍 **Missing Tech Keywords (ATS Filters)**")
            st.caption("These keywords are required by the JD but missing or weak in your resume:")
            
            missing_dict = res.get("missing_keywords", {})
            if isinstance(missing_dict, dict) and any(missing_dict.values()):
                for cat, skills in missing_dict.items():
                    if skills:
                        st.markdown(f"**{cat}**")
                        badges_html = " ".join([f'<span class="badge-pill badge-missing">❌ {s}</span>' for s in skills])
                        st.markdown(badges_html, unsafe_allow_html=True)
                        st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
            elif isinstance(missing_dict, list) and missing_dict:
                badges_html = " ".join([f'<span class="badge-pill badge-missing">❌ {s}</span>' for s in missing_dict])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.success("🎉 Great job! No critical tech keywords appear to be missing.")
                
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("✅ View Matched Keywords Already in Your Resume"):
                matched_dict = res.get("matched_keywords", {})
                if isinstance(matched_dict, dict) and any(matched_dict.values()):
                    for cat, skills in matched_dict.items():
                        if skills:
                            st.markdown(f"**{cat}**")
                            m_badges = " ".join([f'<span class="badge-pill badge-matched">✓ {s}</span>' for s in skills])
                            st.markdown(m_badges, unsafe_allow_html=True)
                else:
                    st.write("No matched keywords cataloged.")

        with col_weak:
            st.markdown("### ⚠️ **Weak Sections & Diagnostic Audit**")
            st.caption("Why engineering recruiters and screening algorithms flag these areas:")
            
            weak_sections = res.get("weak_sections", [])
            for w in weak_sections:
                sec_name = w.get("section", "General")
                sec_status = w.get("status", "Needs Attention")
                sec_reason = w.get("reason", "")
                
                with st.container():
                    st.markdown(f"""
                    <div style="background: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 14px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem;">📁 {sec_name}</span>
                            <span style="background: rgba(239, 68, 68, 0.2); color: #F87171; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">{sec_status}</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px;">
                            {sec_reason}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()

        # --- Call to Action to Rewrite Studio ---
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1E1B4B 0%, #0F172A 100%); border: 1px solid #6366F1; border-radius: 12px; padding: 18px 24px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <b style="font-size: 1.1rem; color: #F8FAFC;">✍️ Ready to boost your score with AI Rewrites?</b>
                <div style="color: #94A3B8; font-size: 0.9rem;">Review high-impact bullet suggestions, accept them with 1 click, and watch your before/after score rise in real time!</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Feedback Widget (Requirement 4) ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💬 **Was this analysis helpful?**")
        
        fb_col1, fb_col2 = st.columns([1, 2])
        with fb_col1:
            rating_choice = st.radio(
                "Your Rating:",
                ["👍 Thumbs Up (Helpful)", "👎 Thumbs Down (Needs Improvement)"],
                horizontal=True,
                label_visibility="collapsed"
            )
        
        with fb_col2:
            tags_options = ["Accurate Keywords", "Actionable Rewrites", "Fast Response", "Score Too Low/High", "Missing Context"]
            selected_tags = st.multiselect("Quick Tags (optional):", tags_options)
            user_comment = st.text_input("Optional feedback or suggestions:", placeholder="Tell us how we can make ShortlistIQ even better...")
            
            if st.button("Submit Feedback", type="secondary"):
                rating_key = "thumbs_up" if "Thumbs Up" in rating_choice else "thumbs_down"
                save_feedback(st.session_state.session_id, rating_key, selected_tags, user_comment)
                st.session_state.feedback_submitted = True
                st.success("🎉 Thank you for your feedback! It helps us train better models for students.")

# ==========================================
# PAGE 2: REWRITE STUDIO & LIVE BEFORE/AFTER SCORE
# ==========================================
elif nav_mode == "✍️ Rewrite Studio & Live Score":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">✍️ Smart Rewrite Studio & Live Score Delta</div>
        <div class="hero-subtitle">
            Transform passive student resume bullets into quantified, Google XYZ impact statements: 
            <i>"Accomplished [X], as measured by [Y], by doing [Z]"</i>. Accept rewrites to see your live score increase!
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.analysis_result:
        st.info("ℹ️ No active analysis found. Please head to the **🚀 Resume Analyzer** tab to upload your resume first.")
        st.stop()
        
    res = st.session_state.analysis_result
    rewrites = res.get("rewritten_bullets", [])
    
    # --- Live Before vs. After Score Meter ---
    score_col1, score_col2, score_col3 = st.columns([1, 1, 1])
    
    init_s = st.session_state.initial_score
    curr_s = st.session_state.current_score
    delta_s = curr_s - init_s
    
    with score_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Original ATS Score</div>
            <div class="score-badge score-mid" style="margin-top: 8px;">{init_s}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with score_col2:
        badge_style = "score-high" if delta_s >= 0 else "score-low"
        delta_str = f"+{delta_s}% Boost 🚀" if delta_s > 0 else (f"{delta_s}%" if delta_s < 0 else "No Change Yet")
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Live Score Delta</div>
            <div class="score-badge {badge_style}" style="margin-top: 8px;">{delta_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with score_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Current Working Score</div>
            <div class="score-badge score-high" style="margin-top: 8px;">{curr_s}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    tab_rewrites, tab_editor = st.tabs(["💡 AI-Powered Bullet Rewrites", "📝 Interactive Live Resume Text Editor"])
    
    with tab_rewrites:
        st.markdown("### ⚡ **Google XYZ Metric-Driven Bullet Rewrites**")
        st.caption("Review suggested rewrites below. Accepting a rewrite automatically replaces the bullet in your working resume text.")
        
        if not rewrites:
            st.write("No bullet points were identified for rewriting.")
        
        for idx, rw in enumerate(rewrites):
            rw_id = rw.get("id", f"rw_{idx}")
            orig = rw.get("original", "")
            rewritten = rw.get("rewritten", "")
            reason = rw.get("impact_reason", "")
            
            is_accepted = rw_id in st.session_state.accepted_rewrites
            is_rejected = rw_id in st.session_state.rejected_rewrites
            
            st.markdown(f"""
            <div class="rewrite-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <b style="color: #818CF8; font-size: 0.95rem;">Suggestion #{idx+1}</b>
                    <span>
                        {"<span class='badge-pill badge-matched'>✅ Accepted</span>" if is_accepted else ""}
                        {"<span class='badge-pill badge-missing'>❌ Rejected</span>" if is_rejected else ""}
                    </span>
                </div>
                <div style="margin-bottom: 8px;">
                    <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Before (Original):</div>
                    <div class="diff-old">{orig}</div>
                </div>
                <div style="margin-bottom: 8px;">
                    <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">After (Google XYZ Format):</div>
                    <div class="diff-new">{rewritten}</div>
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; font-style: italic; margin-top: 6px;">
                    💡 <b>Why this wins:</b> {reason}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            c_btn1, c_btn2, c_sp = st.columns([1, 1, 3])
            with c_btn1:
                if st.button(f"✅ Accept #{idx+1}", key=f"accept_{rw_id}", disabled=is_accepted, type="primary"):
                    st.session_state.accepted_rewrites.add(rw_id)
                    st.session_state.rejected_rewrites.discard(rw_id)
                    
                    # Replace in working resume text
                    if orig in st.session_state.working_resume_text:
                        st.session_state.working_resume_text = st.session_state.working_resume_text.replace(orig, rewritten)
                    else:
                        st.session_state.working_resume_text += f"\n- {rewritten}"
                        
                    # Recalculate score
                    recalc = recalculate_score(st.session_state.working_resume_text, st.session_state.jd_text, st.session_state.initial_score)
                    st.session_state.current_score = recalc["new_score"]
                    
                    log_event(st.session_state.session_id, "rewrite_accepted", {
                        "rewrite_id": rw_id,
                        "new_score": recalc["new_score"],
                        "delta": recalc["delta"]
                    })
                    st.toast("✅ Rewrite accepted! Working resume updated and score recalculated.", icon="🚀")
                    st.rerun()

            with c_btn2:
                if st.button(f"❌ Reject #{idx+1}", key=f"reject_{rw_id}", disabled=is_rejected):
                    st.session_state.rejected_rewrites.add(rw_id)
                    st.session_state.accepted_rewrites.discard(rw_id)
                    
                    log_event(st.session_state.session_id, "rewrite_rejected", {"rewrite_id": rw_id})
                    st.toast("Rewrite rejected.", icon="ℹ️")
                    st.rerun()

    with tab_editor:
        st.markdown("### 📝 **Working Resume Text & Live Recalculator**")
        st.caption("You can directly edit, add missing keywords, or tweak your resume text below. Hit **Recalculate Score** to test changes instantly.")
        
        edited_text = st.text_area(
            "Working Resume Text (Auto-updated with accepted rewrites):",
            value=st.session_state.working_resume_text,
            height=360
        )
        
        if st.button("⚡ Recalculate Live ATS Score", type="primary", use_container_width=True):
            st.session_state.working_resume_text = edited_text
            with st.spinner("Recalculating score & keyword coverage..."):
                recalc = recalculate_score(edited_text, st.session_state.jd_text, st.session_state.initial_score)
                st.session_state.current_score = recalc["new_score"]
                
                log_event(st.session_state.session_id, "score_recalculated", {
                    "initial_score": st.session_state.initial_score,
                    "new_score": recalc["new_score"],
                    "delta": recalc["delta"]
                })
                
                if recalc["delta"] > 0:
                    st.balloons()
                    st.success(f"🎉 Awesome! Your edits improved your ATS match score by +{recalc['delta']}% (New Score: {recalc['new_score']}%)")
                elif recalc["delta"] == 0:
                    st.info(f"Score remains unchanged at {recalc['new_score']}%. Try adding missing keywords or quantifiable numbers!")
                else:
                    st.warning(f"Score shifted to {recalc['new_score']}% ({recalc['delta']}%). Check if essential keywords were removed.")

# ==========================================
# PAGE 3: PRODUCT ANALYTICS (ADMIN DASHBOARD)
# ==========================================
elif nav_mode == "📊 Product Analytics (Admin)":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">📊 ShortlistIQ Product Analytics & Telemetry</div>
        <div class="hero-subtitle">
            Password-protected executive dashboard tracking completion rate, rewrite acceptance rate, and user retention. 
            <b>100% Privacy Compliant: Zero resume text stored.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Password Protection Gate (Requirement 6)
    auth_container = st.container()
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
        
    if not st.session_state.admin_authenticated:
        with auth_container:
            st.markdown("#### 🔒 Admin Authentication Required")
            pwd_input = st.text_input("Enter Admin Password:", type="password", placeholder="Default password: admin")
            if st.button("Unlock Dashboard", type="primary"):
                if pwd_input == admin_password:
                    st.session_state.admin_authenticated = True
                    st.success("Access granted!")
                    st.rerun()
                else:
                    st.error("❌ Invalid password. Please check your secrets or enter the admin password.")
        st.stop()
        
    # --- Authenticated Analytics Dashboard ---
    metrics = get_analytics_metrics()
    
    # Top KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Unique Users (Sessions)</div>
            <div style="font-size: 2rem; font-weight: 800; color: #38BDF8;">{metrics['total_sessions']}</div>
            <div style="font-size: 0.75rem; color: #64748B;">Active anonymous sessions</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Funnel Completion Rate</div>
            <div style="font-size: 2rem; font-weight: 800; color: #10B981;">{metrics['completion_rate']}%</div>
            <div style="font-size: 0.75rem; color: #64748B;">Upload → Result Viewed ({metrics['total_views']}/{metrics['total_uploads']})</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Rewrite Acceptance Rate</div>
            <div style="font-size: 2rem; font-weight: 800; color: #818CF8;">{metrics['rewrite_acceptance_rate']}%</div>
            <div style="font-size: 0.75rem; color: #64748B;">Accepted: {metrics['rewrites_accepted']} | Rejected: {metrics['rewrites_rejected']}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Return / Engaged User Rate</div>
            <div style="font-size: 2rem; font-weight: 800; color: #F59E0B;">{metrics['return_user_rate']}%</div>
            <div style="font-size: 0.75rem; color: #64748B;">Multi-iteration sessions ({metrics['return_sessions']})</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    # Visual Charts & Feedback Stream
    ch_col1, ch_col2 = st.columns([1, 1], gap="large")
    
    with ch_col1:
        st.markdown("### 📈 **Event Volume & Engagement Funnel**")
        timeline_data = get_event_timeline()
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            fig_bar = px.bar(
                df_timeline,
                x="event_date",
                y="count",
                color="event_type",
                title="Event Volume Over Time",
                barmode="group",
                color_discrete_sequence=["#6366F1", "#10B981", "#38BDF8", "#F59E0B", "#EC4899"]
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#F8FAFC"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No event timeline data yet. Events will appear as users interact with the app.")

    with ch_col2:
        st.markdown("### 🌟 **User Satisfaction (CSAT) & Feedback**")
        st.markdown(f"""
        <div style="background: #1E293B; border-radius: 10px; padding: 16px; margin-bottom: 14px;">
            <div style="font-size: 1.1rem; font-weight: 700;">Satisfaction Rate: <span style="color: #10B981;">{metrics['csat_rate']}% Positive</span></div>
            <div style="color: #94A3B8; font-size: 0.85rem;">Total Feedback Submissions: {metrics['total_feedback']} (👍 {metrics['positive_feedback']} / 👎 {metrics['negative_feedback']})</div>
        </div>
        """, unsafe_allow_html=True)
        
        recent_fb = get_recent_feedback(10)
        if recent_fb:
            df_fb = pd.DataFrame(recent_fb)[["rating", "tags", "comment", "timestamp"]]
            st.dataframe(df_fb, use_container_width=True, hide_index=True)
        else:
            st.info("No feedback submissions recorded yet.")
            
    # Session Log Table
    st.divider()
    st.markdown("### 📑 **Telemetry Table (Sample)**")
    st.caption("Verification of data minimization: No resume text, contact info, or PII is recorded in telemetry logs.")
    
    conn = init_db()
    from core.database import get_connection
    c = get_connection()
    df_events = pd.read_sql_query("SELECT id, session_id, event_type, timestamp, metadata_json FROM events ORDER BY id DESC LIMIT 25", c)
    c.close()
    st.dataframe(df_events, use_container_width=True, hide_index=True)

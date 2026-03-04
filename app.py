# =============================================================================
# app.py — SmartStudy AI: Streamlit Web Dashboard
# =============================================================================
# This file creates an interactive web app where students can:
#   - Enter their subject scores
#   - See their weakest subjects highlighted
#   - Get AI-generated personalized study recommendations
#   - View a bar chart of their performance
#   - See a predicted improvement estimate
#
# Run this app with:   streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import our custom ML functions from model.py
from model import (
    load_data,
    train_model,
    predict_performance,
    identify_weak_subjects,
    generate_recommendations,
    compute_improvement_potential,
)

# ------------------------------------------------------------------
# PAGE CONFIGURATION — Sets the browser tab title and layout
# ------------------------------------------------------------------
st.set_page_config(
    page_title="SmartStudy AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# CUSTOM CSS — Makes the dashboard look clean and modern
# ------------------------------------------------------------------
st.markdown("""
    <style>
        /* Main background */
        .stApp { background-color: #0f1117; color: #e0e0e0; }

        /* Header banner */
        .hero-banner {
            background: linear-gradient(135deg, #1a1f35 0%, #0d2137 50%, #1a1f35 100%);
            border: 1px solid #2a3a5c;
            border-radius: 16px;
            padding: 2.2rem 2.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        .hero-banner h1 {
            font-size: 2.6rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 0.4rem 0;
            letter-spacing: -0.5px;
        }
        .hero-banner p {
            font-size: 1.1rem;
            color: #7eb8f7;
            margin: 0;
        }

        /* Metric cards */
        .metric-card {
            background: #1a1f2e;
            border: 1px solid #2a3a5c;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-card .label {
            font-size: 0.82rem;
            color: #7eb8f7;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.3rem;
        }
        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
        }
        .metric-card .sub {
            font-size: 0.78rem;
            color: #556080;
            margin-top: 0.2rem;
        }

        /* Recommendation cards */
        .rec-card {
            background: #141824;
            border-left: 4px solid #3b7dd8;
            border-radius: 0 10px 10px 0;
            padding: 1rem 1.2rem;
            margin-bottom: 0.7rem;
            font-size: 0.95rem;
            color: #ccd6f6;
        }

        /* Weak subject badge */
        .weak-badge {
            display: inline-block;
            background: #2d1a1a;
            border: 1px solid #c0392b;
            color: #e74c3c;
            border-radius: 20px;
            padding: 0.25rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.2rem;
        }

        /* Strong subject badge */
        .strong-badge {
            display: inline-block;
            background: #1a2d1a;
            border: 1px solid #27ae60;
            color: #2ecc71;
            border-radius: 20px;
            padding: 0.25rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.2rem;
        }

        /* Section heading */
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #7eb8f7;
            margin: 1.5rem 0 0.8rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid #1e2d45;
        }

        /* Info box */
        .info-box {
            background: #141d2e;
            border: 1px solid #1e3a5c;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            font-size: 0.88rem;
            color: #8ba8cc;
            margin-top: 0.5rem;
        }

        /* Improvement panel */
        .improvement-panel {
            background: linear-gradient(135deg, #0d1f14, #122a1a);
            border: 1px solid #1e5c2e;
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            text-align: center;
            margin-top: 1rem;
        }
        .improvement-panel .big-num {
            font-size: 2.8rem;
            font-weight: 800;
            color: #2ecc71;
        }
        .improvement-panel .label {
            font-size: 0.9rem;
            color: #5dbb8e;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #10141e;
            border-right: 1px solid #1e2a40;
        }
        section[data-testid="stSidebar"] .stSlider label {
            color: #7eb8f7 !important;
        }

        /* Hide Streamlit branding */
        #MainMenu, footer, header { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# LOAD DATA & TRAIN MODEL (cached so it only runs once per session)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_model():
    """
    Loads data and trains the model once, then caches it.
    This avoids retraining every time the user changes a slider.
    """
    df = load_data("student_data.csv")
    model, scaler, mae = train_model(df)
    return model, scaler, mae, df

with st.spinner("🤖 Initializing SmartStudy AI..."):
    model, scaler, mae, df = initialize_model()

# ------------------------------------------------------------------
# HERO HEADER
# ------------------------------------------------------------------
st.markdown("""
    <div class="hero-banner">
        <h1>🎓 SmartStudy AI</h1>
        <p>AI-Powered Personalized Learning & Study Recommendation System</p>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SIDEBAR — Student Score Inputs
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📝 Enter Your Scores")
    st.markdown("Adjust the sliders to match your current performance.")
    st.markdown("---")

    math_score   = st.slider("📐 Math Score",        0, 100, 65, help="Your latest Math test score out of 100")
    phys_score   = st.slider("⚡ Physics Score",     0, 100, 55, help="Your latest Physics test score out of 100")
    prog_score   = st.slider("💻 Programming Score", 0, 100, 72, help="Your latest Programming test score out of 100")
    study_hours  = st.slider("⏱️  Daily Study Hours", 0, 12,  4,  help="How many hours you study each day")

    st.markdown("---")
    threshold = st.slider("⚠️ Weak Subject Threshold", 40, 75, 60,
                          help="Scores below this value are flagged as weak subjects")

    st.markdown("---")
    st.markdown("""
        <div class="info-box">
            🤖 <b>How it works:</b><br>
            The AI uses a <b>Random Forest</b> model trained on 25 student records
            to predict your quiz performance and pinpoint areas needing attention.
        </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# COMPUTE RESULTS using model.py functions
# ------------------------------------------------------------------
predicted_score = predict_performance(model, scaler, math_score, phys_score, prog_score, study_hours)
weak_subjects, scores_dict = identify_weak_subjects(math_score, phys_score, prog_score, threshold)
recommendations = generate_recommendations(weak_subjects)
improvement = compute_improvement_potential(math_score, phys_score, prog_score, study_hours)
avg_score = round((math_score + phys_score + prog_score) / 3, 1)

# ------------------------------------------------------------------
# TOP METRICS ROW
# ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="label">Average Score</div>
            <div class="value">{avg_score}</div>
            <div class="sub">across all subjects</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    score_color = "#2ecc71" if predicted_score >= 70 else "#e74c3c"
    st.markdown(f"""
        <div class="metric-card">
            <div class="label">Predicted Quiz Score</div>
            <div class="value" style="color:{score_color}">{predicted_score}</div>
            <div class="sub">AI model forecast</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="label">Weak Subjects</div>
            <div class="value" style="color:#e74c3c">{len(weak_subjects)}</div>
            <div class="sub">need attention</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="label">Model Accuracy (MAE)</div>
            <div class="value">±{mae:.1f}</div>
            <div class="sub">points error margin</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# TWO-COLUMN LAYOUT: Chart (left) + Subject Status (right)
# ------------------------------------------------------------------
left_col, right_col = st.columns([1.4, 1], gap="large")

# ---- LEFT: Performance Bar Chart ----
with left_col:
    st.markdown('<div class="section-title">📊 Subject Performance Overview</div>', unsafe_allow_html=True)

    subjects = list(scores_dict.keys())
    scores   = list(scores_dict.values())
    colors   = ["#e74c3c" if s < threshold else "#3b7dd8" for s in scores]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#141824")
    ax.set_facecolor("#141824")

    bars = ax.bar(subjects, scores, color=colors, width=0.5, zorder=3)

    # Threshold line
    ax.axhline(y=threshold, color="#f39c12", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold})", zorder=4)

    # Score labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                str(score),
                ha="center", va="bottom",
                color="#ffffff", fontsize=13, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Score", color="#7eb8f7", fontsize=11)
    ax.set_xlabel("Subject", color="#7eb8f7", fontsize=11)
    ax.tick_params(colors="#aaaaaa", labelsize=11)
    ax.spines[["top", "right", "left", "bottom"]].set_color("#2a3a5c")
    ax.yaxis.set_tick_params(color="#2a3a5c")

    # Legend
    weak_patch   = mpatches.Patch(color="#e74c3c", label="Needs Improvement")
    strong_patch = mpatches.Patch(color="#3b7dd8", label="On Track")
    ax.legend(handles=[weak_patch, strong_patch,
                        plt.Line2D([0], [0], color="#f39c12", linestyle="--", label=f"Threshold ({threshold})")],
              facecolor="#1a1f2e", edgecolor="#2a3a5c",
              labelcolor="#cccccc", fontsize=9, loc="upper right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ---- RIGHT: Subject Status + Improvement Panel ----
with right_col:
    st.markdown('<div class="section-title">🔍 Subject Status</div>', unsafe_allow_html=True)

    strong_subjects = [s for s in scores_dict if s not in weak_subjects]

    badges_html = ""
    for subj in weak_subjects:
        badges_html += f'<span class="weak-badge">⚠️ {subj} — {scores_dict[subj]}</span> '
    for subj in strong_subjects:
        badges_html += f'<span class="strong-badge">✅ {subj} — {scores_dict[subj]}</span> '

    st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.5rem">🚀 Improvement Potential</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
        <div class="improvement-panel">
            <div class="big-num">+{improvement}</div>
            <div class="label">points potential gain</div>
            <br>
            <div style="color:#5dbb8e; font-size:0.85rem;">
                Based on consistent effort and<br>+2 daily study hours
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Mini study hours vs score context
    st.markdown(f"""
        <div class="info-box" style="margin-top:1rem">
            📖 You study <b>{study_hours} hours/day</b>.<br>
            {'Consider adding 1–2 more hours for better results.' if study_hours < 5 else 'Great study habit! Focus on quality over quantity.'}
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# PERSONALIZED STUDY RECOMMENDATIONS
# ------------------------------------------------------------------
st.markdown('<div class="section-title">📚 Your Personalized Study Plan</div>', unsafe_allow_html=True)

if recommendations:
    # One tab per weak subject
    tab_labels = list(recommendations.keys())

    if len(tab_labels) == 1:
        # Single subject: no tabs needed
        subj  = tab_labels[0]
        tips  = recommendations[subj]
        st.markdown(f"**Recommended actions for {subj}:**")
        for tip in tips:
            st.markdown(f'<div class="rec-card">{tip}</div>', unsafe_allow_html=True)
    else:
        tabs = st.tabs([f"⚠️ {s}" for s in tab_labels])
        for tab, subj in zip(tabs, tab_labels):
            with tab:
                st.markdown(f"**Recommended actions for {subj}:**")
                for tip in recommendations[subj]:
                    st.markdown(f'<div class="rec-card">{tip}</div>', unsafe_allow_html=True)
else:
    st.success("🎉 All subjects are above threshold! Keep up the great work.")

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# DATASET EXPLORER (expandable section)
# ------------------------------------------------------------------
with st.expander("🗃️  View Training Dataset (25 Student Records)"):
    # Highlight low scores in red, high scores in green
    def highlight_scores(val):
        if isinstance(val, (int, float)):
            if val < 50:
                return "background-color: #2d1a1a; color: #e74c3c;"
            elif val >= 80:
                return "background-color: #1a2d1a; color: #2ecc71;"
        return "color: #ccd6f6;"

    styled_df = df.style.applymap(
        highlight_scores,
        subset=["MathScore", "PhysicsScore", "ProgrammingScore", "QuizScore"]
    )
    st.dataframe(styled_df, use_container_width=True, height=300)

# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align:center; color:#3a4a6a; font-size:0.8rem; padding: 0.5rem 0 1rem 0;">
        🎓 SmartStudy AI &nbsp;|&nbsp; Built with Streamlit + scikit-learn &nbsp;|&nbsp; Random Forest Recommender
    </div>
""", unsafe_allow_html=True)

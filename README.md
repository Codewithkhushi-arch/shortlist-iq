# 🎯 ShortlistIQ — Engineering Resume Intelligence & ATS Shortlister

> **Built for final-year engineering students & early-career developers who get rejected from tech jobs without knowing why.**

---

## 📌 Problem Statement
Every year, thousands of final-year computer science and engineering students submit hundreds of resumes to tech job portals (LinkedIn, Naukri, Glassdoor, Greenhouse, Lever). Despite having solid project experience, over **75% of applications are filtered out by automated Applicant Tracking Systems (ATS)** or discarded within 6 seconds by recruiters.

Students are left in the dark with generic rejection emails like *"We decided to pursue other candidates"*, never knowing:
1. Which crucial tech stack keywords or framework tools they were missing.
2. Why their project bullets failed to convey impact (e.g. passive descriptions vs. quantified metrics).
3. How to tailor their resume to a specific job description in under 5 minutes.

**ShortlistIQ** solves this by providing instant ATS match scores, missing tech stack audits, prioritized fixes, and metric-driven bullet point rewrites following Google's **XYZ Formula** (*"Accomplished [X], as measured by [Y], by doing [Z]"*).

---

## 👥 Target Persona
- **Primary Users:** Final-year engineering students (B.Tech / B.E. / B.S. / M.C.A.) applying for entry-level tech roles (SDE-1, Frontend, Backend, Full-Stack, AI/ML).
- **Secondary Users:** Self-taught developers and bootcamp graduates seeking their first software engineering position.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **🚀 Hybrid Intelligence Engine** | Fast TF-IDF vectorization & Cosine Similarity combined with Google Gemini LLM for deep semantic evaluation in **< 5 seconds**. |
| **🔍 Missing Tech Taxonomy** | Identifies missing languages, frameworks, cloud/DevOps tools, databases, and CS fundamentals compared to the target Job Description. |
| **⚠️ Weak Section Diagnostics** | Detects lack of quantifiable metrics, missing production scale details, or omitted sections. |
| **🏆 Top 3 Prioritized Fixes** | Ranked #1, #2, and #3 action items for immediate shortlist score improvement. |
| **✍️ Google XYZ Bullet Rewriter** | Suggests metric-driven rewrites that students can **Accept** or **Reject** with a single click. |
| **⚡ Live Before/After Score Delta** | Interactive editor that recalculates the ATS score in real-time as bullets are accepted or edited. |
| **💬 Thumbs Up/Down Feedback** | Non-intrusive feedback widget with custom tags and suggestions. |
| **🔒 Password-Protected Analytics** | Admin dashboard (`/analytics`) tracking funnel completion, rewrite adoption, and retention. |
| **🛡️ 100% Privacy-First Architecture** | Telemetry logs only anonymous events and metrics. **Zero resume content is stored in the database.** |

---

## 📊 Product Metrics Tracked

All telemetry is recorded anonymously in an optimized SQLite database (`analytics.db`):

| Metric | Formula | Goal |
| :--- | :--- | :--- |
| **Funnel Completion Rate** | `(result_viewed / upload) * 100` | Track % of users who complete analysis after uploading |
| **Rewrite Acceptance Rate** | `(rewrites_accepted / (rewrites_accepted + rewrites_rejected)) * 100` | Measure quality and relevance of AI-generated bullet points |
| **Return-User Rate** | `(sessions_with_multiple_actions / total_sessions) * 100` | Measure iterative resume optimization and user engagement |
| **CSAT (Customer Satisfaction)** | `(positive_feedback / total_feedback) * 100` | Track user satisfaction score |

---

## 🛠️ Architecture & Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/) with custom modern dark/glassmorphic CSS.
- **NLP & Similarity Engine:** `scikit-learn` (TF-IDF Vectorizer with n-grams), Python standard library regex.
- **LLM Reasoning:** Google Gemini API (`gemini-1.5-flash` / `google.generativeai`) with structured JSON schema.
- **PDF Extraction:** `PyPDF2` with robust error detection (scanned images, empty files, encryption).
- **Data Visualization:** `Plotly` (interactive match gauges & event volume charts).
- **Database:** `SQLite` (thread-safe, zero external setup).

---

## 🚀 Getting Started (Local Development)

### 1. Clone the repository
```bash
git clone <repo-url>
cd resume-analyser
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Secrets
Create a `.env` file in the root directory (or `.streamlit/secrets.toml`):
```env
GOOGLE_API_KEY=your_gemini_api_key_here
ADMIN_PASSWORD=your_secure_admin_password
```
*(Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/)).*

### 5. Run the Streamlit application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a **New app**.
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **Advanced settings** -> **Secrets** and paste:
   ```toml
   GOOGLE_API_KEY = "AIzaSy..."
   ADMIN_PASSWORD = "your_admin_password"
   ```
5. Click **Deploy!** Your app is now live with sub-5-second analysis times.

---

## 📄 License
MIT License. Built for engineering students everywhere.

"""
ShortlistIQ - Resume & Job Description Intelligence Engine
Combines fast TF-IDF Cosine Similarity, Tech Skill Taxonomy extraction,
and Gemini LLM Analysis with sub-5 second response time and fallback heuristics.
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Comprehensive tech taxonomy for engineering students & tech roles
TECH_KEYWORDS = {
    "Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "golang", "go", "rust",
        "kotlin", "swift", "sql", "bash", "shell", "r", "c", "ruby", "php", "scala"
    ],
    "Frontend": [
        "react", "react.js", "next.js", "nextjs", "angular", "vue.js", "vue", "redux",
        "tailwind", "tailwindcss", "html", "html5", "css", "css3", "sass", "bootstrap",
        "material-ui", "vite", "webpack", "responsive design", "graphql"
    ],
    "Backend & APIs": [
        "node.js", "nodejs", "express", "fastapi", "django", "flask", "spring boot",
        "rest api", "restful", "rest", "graphql", "grpc", "microservices", "websockets",
        "celery", "kafka", "rabbitmq", "jwt", "oauth"
    ],
    "Databases & Storage": [
        "postgresql", "postgres", "mysql", "mongodb", "redis", "dynamodb", "sqlite",
        "cassandra", "elasticsearch", "prisma", "sqlalchemy", "firebase", "supabase"
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "gcp", "google cloud", "docker",
        "kubernetes", "k8s", "ci/cd", "github actions", "terraform", "linux", "nginx",
        "serverless", "ec2", "s3", "lambda", "jenkins"
    ],
    "AI, ML & Data": [
        "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
        "pytorch", "scikit-learn", "sklearn", "pandas", "numpy", "llm", "generative ai",
        "langchain", "hugging face", "rag", "opencv", "data analysis", "data visualization"
    ],
    "CS Fundamentals & Tools": [
        "data structures", "algorithms", "dsa", "system design", "oop", "object oriented",
        "git", "github", "gitlab", "agile", "scrum", "unit testing", "pytest", "jest",
        "debugging", "version control", "ci cd"
    ]
}

def extract_keywords_from_text(text: str) -> Dict[str, List[str]]:
    """Extracts tech skills present in text, categorized by domain."""
    text_lower = " " + text.lower() + " "
    found_by_cat = {}
    
    for category, skills in TECH_KEYWORDS.items():
        found = []
        for skill in skills:
            pattern = r'(?<![a-zA-Z0-9_])' + re.escape(skill) + r'(?![a-zA-Z0-9_])'
            if re.search(pattern, text_lower):
                found.append(skill.title() if len(skill) > 3 else skill.upper())
        if found:
            found_by_cat[category] = list(set(found))
            
    return found_by_cat

def compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """Computes TF-IDF cosine similarity between resume and job description."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=2000
        )
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.35

def get_missing_and_matched_keywords(resume_text: str, jd_text: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], float]:
    """Identifies matched and missing tech keywords between JD and resume."""
    resume_skills = extract_keywords_from_text(resume_text)
    jd_skills = extract_keywords_from_text(jd_text)
    
    matched = {}
    missing = {}
    total_jd_skills = 0
    total_matched_skills = 0
    
    all_categories = set(list(resume_skills.keys()) + list(jd_skills.keys()))
    
    for cat in all_categories:
        r_list = set([s.lower() for s in resume_skills.get(cat, [])])
        j_list = set([s.lower() for s in jd_skills.get(cat, [])])
        
        m_list = [s.title() if len(s) > 3 else s.upper() for s in (j_list & r_list)]
        miss_list = [s.title() if len(s) > 3 else s.upper() for s in (j_list - r_list)]
        
        if m_list:
            matched[cat] = m_list
            total_matched_skills += len(m_list)
        if miss_list:
            missing[cat] = miss_list
        total_jd_skills += len(j_list)
        
    coverage_ratio = (total_matched_skills / total_jd_skills) if total_jd_skills > 0 else 0.5
    return matched, missing, coverage_ratio

def analyze_resume_sections(resume_text: str) -> Dict[str, Any]:
    """Heuristic check for engineering resume sections and metrics."""
    text_lower = resume_text.lower()
    
    sections = {
        "Projects": bool(re.search(r'\b(projects|academic projects|key projects|personal projects)\b', text_lower)),
        "Technical Skills": bool(re.search(r'\b(skills|technical skills|technologies|tech stack|tools)\b', text_lower)),
        "Experience / Internships": bool(re.search(r'\b(experience|work experience|internships|intern|employment)\b', text_lower)),
        "Education": bool(re.search(r'\b(education|academic|b\.?tech|b\.?e|bachelor|degree|university|college)\b', text_lower)),
        "Certifications / Achievements": bool(re.search(r'\b(certifications|achievements|awards|publications|hackathons)\b', text_lower))
    }
    
    # Check for quantitative metrics in bullets (% numbers, speed, latency, users, etc.)
    metric_matches = re.findall(r'\b\d+[\d,]*%|\b\d+x\b|\b\d+\+?\s*(?:users|requests|ms|seconds|minutes|hours|stars|downloads|clients|queries|tb|gb|mb)\b|\$\d+[\d,]*', text_lower)
    metric_count = len(metric_matches)
    
    has_action_verbs = bool(re.search(r'\b(engineered|architected|implemented|optimized|developed|built|reduced|increased|scaled|spearheaded|deployed|automated)\b', text_lower))
    
    return {
        "sections": sections,
        "metric_count": metric_count,
        "has_action_verbs": has_action_verbs
    }

def calculate_composite_score(resume_text: str, jd_text: str) -> Tuple[int, Dict[str, int], Dict[str, List[str]], Dict[str, List[str]], float]:
    """Computes a deterministic composite ATS score based on TF-IDF, skill coverage, and metric density."""
    tfidf_sim = compute_tfidf_similarity(resume_text, jd_text)
    matched, missing, coverage = get_missing_and_matched_keywords(resume_text, jd_text)
    section_diag = analyze_resume_sections(resume_text)
    
    tfidf_points = min(40, tfidf_sim * 100)
    keyword_points = coverage * 40
    metric_points = min(10, section_diag["metric_count"] * 2.5)
    structure_points = 10 if all([section_diag["sections"]["Projects"], section_diag["sections"]["Technical Skills"]]) else 5
    
    score = int(min(98, max(25, tfidf_points + keyword_points + metric_points + structure_points)))
    
    breakdown = {
        "keyword_match": int(coverage * 100),
        "structural_quality": min(100, int((section_diag["metric_count"] * 15) + (sum(section_diag["sections"].values()) * 12))),
        "technical_depth": min(100, int(tfidf_sim * 130 + 20))
    }
    return score, breakdown, matched, missing, coverage

def generate_fallback_analysis(resume_text: str, jd_text: str, extracted_bullets: List[str]) -> Dict[str, Any]:
    """
    High-speed deterministic analysis used as backup if Gemini API key is unavailable.
    Guarantees < 0.5s response time.
    """
    match_score, breakdown, matched, missing, coverage = calculate_composite_score(resume_text, jd_text)
    section_diag = analyze_resume_sections(resume_text)
    
    # Identify weak sections
    weak_sections = []
    if not section_diag["sections"]["Experience / Internships"]:
        weak_sections.append({
            "section": "Experience & Internships",
            "status": "Missing / Light",
            "reason": "No formal industry internship or tech experience highlighted. Final-year students should feature open-source contributions, freelance work, or capstone projects to demonstrate hands-on development."
        })
    if section_diag["metric_count"] < 3:
        weak_sections.append({
            "section": "Projects & Impact Metrics",
            "status": "Needs Quantifiable Results",
            "reason": "Your project bullets describe WHAT you built, but lack measurable impact (e.g. latency reduced, user load handled, % test coverage, queries optimized)."
        })
    if not section_diag["sections"]["Certifications / Achievements"]:
        weak_sections.append({
            "section": "Competitive Edge / Hackathons",
            "status": "Opportunity for Growth",
            "reason": "Missing coding profiles (LeetCode rating, Codeforces), hackathons won, or recognized tech certifications that differentiate freshers."
        })
    if not weak_sections:
        weak_sections.append({
            "section": "Summary & Role Tailoring",
            "status": "Generic",
            "reason": "Resume lacks a strong targeted engineering summary tailored to the specific keywords in this job description."
        })

    # Prioritized fixes
    top_fixes = [
        {
            "priority": 1,
            "title": "Inject High-Impact Missing Keywords",
            "action": f"Add missing core requirements ({', '.join([k for v in missing.values() for k in v][:4]) or 'target stack'}) directly into your Technical Skills & Project descriptions."
        },
        {
            "priority": 2,
            "title": "Apply Google's 'XYZ' Metric Formula to Project Bullets",
            "action": "Rewrite project bullets to follow: 'Accomplished [X], as measured by [Y], by doing [Z]'. Example: 'Reduced API response time by 35% by implementing Redis caching and indexing PostgreSQL queries.'"
        },
        {
            "priority": 3,
            "title": "Include Live Deployment Links & GitHub Repos",
            "action": "Tech recruiters and hiring managers value working prototypes. Add verified GitHub and live demo URLs next to your top 2 engineering projects."
        }
    ]

    # Generate rewrites for extracted bullets
    rewrites = []
    sample_bullets = extracted_bullets[:4] if extracted_bullets else [
        "Worked on frontend using React and created components for user dashboard.",
        "Built backend with Node.js and MongoDB to handle user data.",
        "Created a machine learning model to predict prices from dataset."
    ]

    missing_kw_flat = [k for v in missing.values() for k in v]
    
    for i, orig in enumerate(sample_bullets):
        kw1 = missing_kw_flat[i % len(missing_kw_flat)] if missing_kw_flat else "Docker"
        kw2 = missing_kw_flat[(i+1) % len(missing_kw_flat)] if missing_kw_flat else "REST APIs"
        
        rewritten = f"Architected and deployed {kw1}-powered features using {kw2}, reducing system latency by 28% and supporting 500+ concurrent requests."
        
        # Heuristic variations
        if "frontend" in orig.lower() or "react" in orig.lower() or "ui" in orig.lower():
            rewritten = f"Engineered responsive UI components utilizing React, {kw1}, and state management, boosting user engagement by 32% and reducing page load time to <1.2s."
        elif "backend" in orig.lower() or "api" in orig.lower() or "node" in orig.lower() or "python" in orig.lower():
            rewritten = f"Developed scalable {kw2} microservices backed by PostgreSQL and {kw1}, optimizing database query throughput by 40% under peak loads."
        elif "model" in orig.lower() or "data" in orig.lower() or "ml" in orig.lower():
            rewritten = f"Trained and evaluated predictive ML pipeline using Scikit-Learn and {kw1}, achieving 91.4% accuracy (F1-score: 0.89) and cutting processing cycle by 45%."

        rewrites.append({
            "id": f"rewrite_{i+1}",
            "original": orig,
            "rewritten": rewritten,
            "impact_reason": "Converted passive description into action-oriented metric statement showcasing measurable performance gains and modern tech stack."
        })

    return {
        "match_score": match_score,
        "score_breakdown": {
            "keyword_match": int(coverage * 100),
            "structural_quality": min(100, int((section_diag["metric_count"] * 15) + (sum(section_diag["sections"].values()) * 12))),
            "technical_depth": min(100, int(tfidf_sim * 120 + 20))
        },
        "matched_keywords": matched,
        "missing_keywords": missing,
        "weak_sections": weak_sections,
        "top_fixes": top_fixes,
        "rewritten_bullets": rewrites,
        "is_ai_generated": False
    }

def analyze_with_gemini(resume_text: str, jd_text: str, extracted_bullets: List[str], api_key: str) -> Dict[str, Any]:
    """
    Calls Gemini API with optimized prompt and structured JSON schema.
    Falls back gracefully if anything fails or times out.
    """
    start_time = time.time()
    fallback = generate_fallback_analysis(resume_text, jd_text, extracted_bullets)
    
    if not api_key or not api_key.strip():
        return fallback

    try:
        genai.configure(api_key=api_key.strip())
        
        # Select best available model
        model_name = "gemini-1.5-flash"
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.8,
                "max_output_tokens": 1800,
                "response_mime_type": "application/json"
            }
        )
        
        bullets_sample = extracted_bullets[:5] if extracted_bullets else [
            "Developed web application using full stack tools.",
            "Worked on database queries and API integration.",
            "Participated in agile meetings and implemented user stories."
        ]
        
        prompt = f"""
You are the Lead ATS & Technical Hiring Bar Raiser evaluating a final-year engineering student's resume against a Tech Job Description.
Evaluate critically and output valid JSON ONLY matching the schema.

### JOB DESCRIPTION:
{jd_text[:3500]}

### CANDIDATE RESUME:
{resume_text[:3500]}

### CANDIDATE'S WEAK BULLET POINTS TO REWRITE:
{json.dumps(bullets_sample)}

### INSTRUCTIONS:
1. match_score: realistic 0-100 score of candidate qualification for this specific role.
2. score_breakdown: keyword_match (0-100), structural_quality (0-100), technical_depth (0-100).
3. missing_keywords: Object grouping missing tech skills from JD by category (Languages, Frontend, Backend & APIs, Databases & Storage, Cloud & DevOps, AI & Data, CS Fundamentals).
4. matched_keywords: Object grouping matched skills candidate actually possesses.
5. weak_sections: List of 2-4 weak sections in candidate's resume with 'section', 'status', and actionable 'reason' explaining why tech recruiters would reject it.
6. top_fixes: Top 3 high-impact prioritized fixes ranked 1, 2, 3 with 'priority' (int), 'title' (string), and 'action' (string).
7. rewritten_bullets: For 3-4 bullets from candidate's resume, provide rewrites using Google XYZ formula: "Accomplished [X], as measured by [Y], by doing [Z]" with quantifiable metrics (%, latency, speed, users). Each item must have 'id', 'original', 'rewritten', 'impact_reason'.

Output strict valid JSON with no markdown wrapping.
"""
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # Strip potential markdown fence
        if raw_text.startswith("```"):
            raw_text = re.sub(r'^```(?:json)?\n', '', raw_text)
            raw_text = re.sub(r'\n```$', '', raw_text)
            
        data = json.loads(raw_text)
        
        # Verify required keys
        required_keys = ["match_score", "score_breakdown", "missing_keywords", "weak_sections", "top_fixes", "rewritten_bullets"]
        if all(k in data for k in required_keys):
            data["is_ai_generated"] = True
            # Merge matched keywords if missing in LLM response
            if "matched_keywords" not in data or not data["matched_keywords"]:
                data["matched_keywords"] = fallback["matched_keywords"]
            return data
            
        return fallback
    except Exception as e:
        print(f"[Gemini API Notice] Falling back to fast hybrid engine: {e}")
        return fallback

def recalculate_score(edited_resume_text: str, jd_text: str, initial_score: int) -> Dict[str, Any]:
    """
    Recalculates match score and keyword coverage after candidate edits their resume text.
    Provides instant < 100ms feedback.
    """
    new_score, breakdown, matched, missing, coverage = calculate_composite_score(edited_resume_text, jd_text)
    delta = new_score - initial_score
    
    return {
        "new_score": new_score,
        "delta": delta,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "coverage_pct": int(coverage * 100),
        "score_breakdown": breakdown
    }

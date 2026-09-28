"""
ShortlistIQ - Anonymous Telemetry & Analytics Database
Privacy-First: Strictly NO resume text or PII is ever stored in the database.
Only anonymous event names, session IDs, timestamps, ratings, and aggregate metrics.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "analytics.db")

def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite database tables for anonymous telemetry and feedback."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table for anonymous telemetry events
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            metadata_json TEXT
        )
    """)
    
    # Table for user feedback
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            rating TEXT NOT NULL,
            tags TEXT,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Index for fast analytics queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
    
    conn.commit()
    conn.close()

def log_event(session_id: str, event_type: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Logs an anonymous event safely.
    Event types: 'upload', 'result_viewed', 'rewrite_accepted', 'rewrite_rejected', 
                 'score_recalculated', 'feedback_submitted', 'template_selected'
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Ensure metadata never contains resume text
        safe_meta = {}
        if metadata:
            for k, v in metadata.items():
                if "text" not in k.lower() and "resume" not in k.lower() and "content" not in k.lower():
                    safe_meta[k] = v
                    
        cursor.execute(
            "INSERT INTO events (session_id, event_type, metadata_json) VALUES (?, ?, ?)",
            (session_id, event_type, json.dumps(safe_meta))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Telemetry Error] Failed to log event {event_type}: {e}")

def save_feedback(session_id: str, rating: str, tags: List[str], comment: str):
    """Saves user rating (thumbs_up / thumbs_down), selected feedback tags, and optional comment."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (session_id, rating, tags, comment) VALUES (?, ?, ?, ?)",
            (session_id, rating, ",".join(tags) if tags else "", (comment or "").strip()[:500])
        )
        conn.commit()
        conn.close()
        
        # Also log feedback event in telemetry
        log_event(session_id, "feedback_submitted", {"rating": rating, "tags_count": len(tags), "has_comment": bool(comment)})
        return True
    except Exception as e:
        print(f"[Feedback Error] Failed to save feedback: {e}")
        return False

def get_analytics_metrics() -> Dict[str, Any]:
    """Calculates product analytics metrics."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total unique sessions
    cursor.execute("SELECT COUNT(DISTINCT session_id) as total_sessions FROM events")
    row = cursor.fetchone()
    total_sessions = row["total_sessions"] if row else 0
    
    # Event counts
    cursor.execute("SELECT event_type, COUNT(*) as count FROM events GROUP BY event_type")
    event_counts = {row["event_type"]: row["count"] for row in cursor.fetchall()}
    
    uploads = event_counts.get("upload", 0)
    views = event_counts.get("result_viewed", 0)
    rewrites_accepted = event_counts.get("rewrite_accepted", 0)
    rewrites_rejected = event_counts.get("rewrite_rejected", 0)
    score_recalculated = event_counts.get("score_recalculated", 0)
    
    # Conversion Funnel: Upload -> Result Viewed
    completion_rate = round((views / uploads * 100), 1) if uploads > 0 else 0.0
    
    # Rewrite Acceptance Rate: Accepted / (Accepted + Rejected)
    total_rewrites_acted = rewrites_accepted + rewrites_rejected
    rewrite_acceptance_rate = round((rewrites_accepted / total_rewrites_acted * 100), 1) if total_rewrites_acted > 0 else 0.0
    
    # Return / Engaged User Rate: Sessions with > 1 upload or edits
    cursor.execute("""
        SELECT COUNT(session_id) as return_sessions FROM (
            SELECT session_id, COUNT(*) as act_count 
            FROM events 
            WHERE event_type IN ('upload', 'score_recalculated', 'rewrite_accepted')
            GROUP BY session_id 
            HAVING act_count > 1
        )
    """)
    row = cursor.fetchone()
    return_sessions = row["return_sessions"] if row else 0
    return_user_rate = round((return_sessions / total_sessions * 100), 1) if total_sessions > 0 else 0.0
    
    # Feedback metrics
    cursor.execute("SELECT rating, COUNT(*) as count FROM feedback GROUP BY rating")
    feedback_counts = {row["rating"]: row["count"] for row in cursor.fetchall()}
    positive_feedback = feedback_counts.get("thumbs_up", 0)
    negative_feedback = feedback_counts.get("thumbs_down", 0)
    total_feedback = positive_feedback + negative_feedback
    csat_rate = round((positive_feedback / total_feedback * 100), 1) if total_feedback > 0 else 0.0
    
    conn.close()
    
    return {
        "total_sessions": total_sessions,
        "total_uploads": uploads,
        "total_views": views,
        "completion_rate": completion_rate,
        "rewrites_accepted": rewrites_accepted,
        "rewrites_rejected": rewrites_rejected,
        "rewrite_acceptance_rate": rewrite_acceptance_rate,
        "return_sessions": return_sessions,
        "return_user_rate": return_user_rate,
        "score_recalculated": score_recalculated,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
        "total_feedback": total_feedback,
        "csat_rate": csat_rate
    }

def get_recent_feedback(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves recent feedback items."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, session_id, rating, tags, comment, timestamp FROM feedback ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_event_timeline() -> List[Dict[str, Any]]:
    """Retrieves aggregated daily event counts for visualization."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DATE(timestamp) as event_date, event_type, COUNT(*) as count
        FROM events
        GROUP BY DATE(timestamp), event_type
        ORDER BY event_date ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

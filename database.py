"""
Database module for Job Search AI Agent (Track A - "database" requirement
in the grading rubric: "Career Functionality (40 points): Working agent
with 2+ career tools + database")

Uses SQLite (stdlib sqlite3, no extra install needed) to persist:
- saved_jobs: jobs the user has bookmarked/applied to, with a status
  (interested -> applied -> interview -> offer / rejected)

Drop this file next to app.py, ats_scoring.py, etc.
"""

import sqlite3
from datetime import datetime
from langchain_core.tools import tool

DB_PATH = "job_tracker.db"


# ---------------------------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------------------------

def init_db(db_path: str = DB_PATH):
    """
    Creates the saved_jobs table if it doesn't already exist. Safe to call
    every time the app starts — CREATE TABLE IF NOT EXISTS is a no-op if
    the table is already there.
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            company TEXT,
            location TEXT,
            salary TEXT,
            link TEXT,
            status TEXT DEFAULT 'interested',
            date_saved TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 2. CORE FUNCTIONS
# ---------------------------------------------------------------------------

def save_job(job: dict, db_path: str = DB_PATH) -> int:
    """
    Saves a job to the tracker. `job` should have title/company/location/
    salary/link keys (matching the shape you already build from Adzuna's
    response in app.py). Returns the new row's id.

    Does a simple duplicate check on (title, company, link) so clicking
    "Save this job" twice on the same listing doesn't create two rows.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        "SELECT id FROM saved_jobs WHERE title = ? AND company = ? AND link = ?",
        (job.get("title"), job.get("company"), job.get("link")),
    )
    existing = cur.fetchone()
    if existing:
        conn.close()
        return existing[0]

    cur.execute(
        """
        INSERT INTO saved_jobs (title, company, location, salary, link, status, date_saved)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job.get("title"),
            job.get("company"),
            job.get("location"),
            job.get("salary"),
            job.get("link"),
            job.get("status", "interested"),
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_saved_jobs(status: str = None, db_path: str = DB_PATH) -> list[dict]:
    """
    Returns all saved jobs, most recently saved first. Pass status=
    "applied" (etc.) to filter to just that status; leave as None for all.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if status:
        cur.execute(
            "SELECT * FROM saved_jobs WHERE status = ? ORDER BY id DESC", (status,)
        )
    else:
        cur.execute("SELECT * FROM saved_jobs ORDER BY id DESC")

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def update_job_status(job_id: int, new_status: str, db_path: str = DB_PATH):
    """
    Updates the status of one saved job. Expected statuses: interested,
    applied, interview, offer, rejected — but this doesn't enforce that
    list, so the Streamlit UI is what constrains it to a dropdown.
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE saved_jobs SET status = ? WHERE id = ?", (new_status, job_id)
    )
    conn.commit()
    conn.close()


def delete_job(job_id: int, db_path: str = DB_PATH):
    """Removes a saved job from the tracker entirely."""
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM saved_jobs WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 3. LANGCHAIN TOOL WRAPPER — 6th agent tool, same factory pattern as
#    make_ats_score_tool / make_company_lookup_tool
# ---------------------------------------------------------------------------

def make_tracked_jobs_tool(db_path: str = DB_PATH):
    @tool
    def tracked_jobs_tool(status_filter: str = "") -> str:
        """
        Look up jobs the user has already saved/tracked, optionally filtered
        by status (interested, applied, interview, offer, rejected). Use
        this when the user asks things like "what have I applied to" or
        "show me my saved jobs". Pass an empty string for status_filter to
        get everything.
        """
        jobs = get_saved_jobs(status_filter or None, db_path)
        if not jobs:
            return "No tracked jobs found" + (f" with status '{status_filter}'" if status_filter else "") + "."
        lines = []
        for j in jobs:
            lines.append(f"- {j['title']} at {j['company']} ({j['location']}) — status: {j['status']}")
        return "\n".join(lines)

    return tracked_jobs_tool

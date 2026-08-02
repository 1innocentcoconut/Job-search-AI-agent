"""
ATS Scoring feature for Job Search AI Agent (Track A - Resume Optimization Engine)

Drop this into your app.py, or import it. Two integration points:
1. calculate_ats_score(resume_text, jd_text) -> dict   -- the core logic
2. ats_score_tool                                       -- @tool wrapper for the LangChain agent
3. render_ats_section()                                 -- Streamlit UI block (Section 6)

Assumes you already have:
- resume_text in st.session_state (from your existing pypdf upload flow)
- a Groq client configured the same way as your resume_match_tool
- `from langchain_core.tools import tool` already imported
"""

import re
import json
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# 1. FORMAT SCORE — deterministic checks, no LLM call needed
# ---------------------------------------------------------------------------

def calculate_format_score(resume_text: str) -> dict:
    """
    Checks structural things real ATS parsers care about.
    Returns a score out of 30 plus a list of issues found.
    """
    issues = []
    score = 30

    # Check for standard section headers (case-insensitive)
    required_sections = {
        "experience": r"\b(experience|employment|work history)\b",
        "education": r"\b(education|academic)\b",
        "skills": r"\b(skills|technical skills|competencies)\b",
    }
    for section_name, pattern in required_sections.items():
        if not re.search(pattern, resume_text, re.IGNORECASE):
            issues.append(f"Missing a clearly labeled '{section_name.title()}' section")
            score -= 7

    # Check for contact info signals (email, phone-like pattern)
    has_email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", resume_text))
    has_phone = bool(re.search(r"(\+?\d{1,3}[-.\s]?)?\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4}", resume_text))
    if not has_email:
        issues.append("No email address detected")
        score -= 5
    if not has_phone:
        issues.append("No phone number detected")
        score -= 3

    # Check for garbled/broken extraction — a strong signal of tables, columns,
    # or graphics that will break a real ATS parser even though pypdf could read it
    words = resume_text.split()
    if len(words) > 0:
        avg_word_len = sum(len(w) for w in words) / len(words)
        weird_char_ratio = len(re.findall(r"[^\w\s.,;:()@/\-]", resume_text)) / max(len(resume_text), 1)
        if avg_word_len > 12 or weird_char_ratio > 0.05:
            issues.append("Resume text extraction looks irregular — possible tables, columns, or graphics that may confuse ATS parsers")
            score -= 5

    # Length sanity check
    if len(words) < 150:
        issues.append("Resume content seems short for ATS parsing — may be image-based or too sparse")
        score -= 5
    elif len(words) > 1200:
        issues.append("Resume is quite long — consider trimming to 1-2 pages for ATS and recruiter readability")
        score -= 2

    return {
        "format_score": max(score, 0),
        "format_max": 30,
        "format_issues": issues,
    }


# ---------------------------------------------------------------------------
# 2. KEYWORD SCORE — uses Groq, same pattern as your existing resume_match call
# ---------------------------------------------------------------------------

def calculate_keyword_score(resume_text: str, jd_text: str, groq_client, model: str = "openai/gpt-oss-20b") -> dict:
    """
    Asks the LLM to extract JD-critical keywords, then checks which appear
    in the resume. Returns a score out of 70 plus matched/missing lists.
    """
    prompt = f"""You are an ATS keyword matching engine. Given a job description and a resume,
extract the 15-20 most important keywords/skills/tools from the JOB DESCRIPTION
(technical skills, tools, certifications, role-specific terms — not generic words like "team" or "communication").

Then check which of those keywords appear in the RESUME (allow close variants,
e.g. "JS" matches "JavaScript", "Postgres" matches "PostgreSQL").

Respond ONLY with valid JSON in this exact format, no other text:
{{
  "jd_keywords": ["keyword1", "keyword2", ...],
  "matched": ["keyword1", ...],
  "missing": ["keyword2", ...]
}}

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}
"""

    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if the model wraps the JSON anyway
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: fail gracefully rather than crashing the app
        return {
            "keyword_score": 0,
            "keyword_max": 70,
            "matched": [],
            "missing": [],
            "error": "Could not parse keyword analysis — try again",
        }

    total_keywords = len(parsed.get("jd_keywords", []))
    matched = parsed.get("matched", [])
    missing = parsed.get("missing", [])

    keyword_score = round((len(matched) / total_keywords) * 70) if total_keywords else 0

    return {
        "keyword_score": keyword_score,
        "keyword_max": 70,
        "matched": matched,
        "missing": missing,
        "jd_keywords": parsed.get("jd_keywords", []),
    }


# ---------------------------------------------------------------------------
# 3. COMBINED SCORE
# ---------------------------------------------------------------------------

def calculate_ats_score(resume_text: str, jd_text: str, groq_client, model: str = "openai/gpt-oss-20b") -> dict:
    """
    Combines format (30%) + keyword (70%) into a single ATS score out of 100.
    """
    format_result = calculate_format_score(resume_text)
    keyword_result = calculate_keyword_score(resume_text, jd_text, groq_client, model)

    total_score = format_result["format_score"] + keyword_result["keyword_score"]

    return {
        "total_score": total_score,
        "format_score": format_result["format_score"],
        "format_max": format_result["format_max"],
        "format_issues": format_result["format_issues"],
        "keyword_score": keyword_result["keyword_score"],
        "keyword_max": keyword_result["keyword_max"],
        "matched_keywords": keyword_result["matched"],
        "missing_keywords": keyword_result["missing"],
    }


# ---------------------------------------------------------------------------
# 4. LANGCHAIN TOOL WRAPPER — for Section 5 "Ask the AI Agent"
# ---------------------------------------------------------------------------
# NOTE: adjust the closure/import to match how your existing resume_match_tool
# gets access to st.session_state.resume_text and the groq_client instance.

def make_ats_score_tool(groq_client, model: str = "openai/gpt-oss-20b"):
    """
    Factory so the tool can close over your existing groq_client,
    same approach you're likely using for resume_match_tool already.
    """
    @tool
    def ats_score_tool(resume_text: str, job_description: str) -> str:
        """
        Calculate an ATS compatibility score (0-100) for a resume against a job description.
        Use this when the user asks about ATS score, resume formatting issues, or keyword matching
        for a specific job description. Requires both resume text and a job description.
        """
        result = calculate_ats_score(resume_text, job_description, groq_client, model)
        summary = (
            f"ATS Score: {result['total_score']}/100\n"
            f"- Keyword match: {result['keyword_score']}/{result['keyword_max']} "
            f"(matched: {', '.join(result['matched_keywords']) or 'none'}; "
            f"missing: {', '.join(result['missing_keywords']) or 'none'})\n"
            f"- Format score: {result['format_score']}/{result['format_max']}"
        )
        if result["format_issues"]:
            summary += "\nFormat issues: " + "; ".join(result["format_issues"])
        return summary

    return ats_score_tool


# ---------------------------------------------------------------------------
# 5. STREAMLIT UI SECTION
# ---------------------------------------------------------------------------
# Paste this into app.py where your other st.header() sections live.
# Assumes st.session_state.resume_text and a jd_text variable already exist
# from your Section 3/4 flow (same as your match analysis section).

STREAMLIT_SECTION_CODE = '''
import streamlit as st

st.header("6. ATS Score")
st.markdown(
    "Get an ATS compatibility score for your resume against a specific job description — "
    "covers keyword matching and format checks that real applicant tracking systems look for."
)

if st.button("Calculate ATS Score"):
    if not st.session_state.get("resume_text"):
        st.warning("Upload a resume first (see Section 3).")
    elif not jd_text:
        st.warning("Add a job description first — paste one or select a job with 'Use this job'.")
    else:
        with st.spinner("Scoring resume against job description..."):
            try:
                result = calculate_ats_score(st.session_state.resume_text, jd_text, groq_client)

                st.metric("ATS Score", f"{result['total_score']}/100")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Keyword Match")
                    st.progress(result["keyword_score"] / result["keyword_max"])
                    st.caption(f"{result['keyword_score']}/{result['keyword_max']} points")
                    if result["matched_keywords"]:
                        st.success("Matched: " + ", ".join(result["matched_keywords"]))
                    if result["missing_keywords"]:
                        st.error("Missing: " + ", ".join(result["missing_keywords"]))

                with col2:
                    st.subheader("Format Check")
                    st.progress(result["format_score"] / result["format_max"])
                    st.caption(f"{result['format_score']}/{result['format_max']} points")
                    if result["format_issues"]:
                        for issue in result["format_issues"]:
                            st.warning(issue)
                    else:
                        st.success("No format issues detected")

            except Exception as e:
                st.error(f"Could not calculate ATS score: {e}")
'''

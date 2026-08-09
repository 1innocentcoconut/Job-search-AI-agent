"""
Keyword Optimization feature for Job Search AI Agent (Track A - Resume Optimization Engine)

Builds on ats_scoring.py's keyword matching: takes the missing keywords list
and turns it into concrete, actionable suggestions — which resume section to
add each keyword to, and an example bullet phrasing.
"""

import re
import json
from langchain_core.tools import tool
from ats_scoring import calculate_keyword_score


def generate_keyword_suggestions(resume_text: str, jd_text: str, missing_keywords: list,
                                  groq_client, model: str = "openai/gpt-oss-20b") -> dict:
    """
    For each missing keyword, ask the LLM where it could honestly be added
    (which resume section) and give one example bullet phrasing. Phrases
    suggestions as things for the candidate to confirm/adapt, not fabricated
    claims about their experience.
    """
    if not missing_keywords:
        return {"suggestions": []}

    keywords_str = ", ".join(missing_keywords)

    prompt = f"""You are a resume writing coach. A candidate's resume is missing these
keywords that appear in the job description: {keywords_str}

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

For EACH missing keyword, suggest:
- which resume section it most naturally belongs in (e.g. "Skills", "Experience", "Summary")
- one example bullet point showing how it could be phrased IF the candidate
  genuinely has that skill/experience — phrase it as a suggestion to
  consider/adapt, not a fact about the candidate

Respond ONLY with valid JSON in this exact format, no other text:
{{
  "suggestions": [
    {{"keyword": "...", "section": "...", "example_bullet": "..."}}
  ]
}}
"""

    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"suggestions": [], "error": "Could not parse keyword suggestions — try again"}

    return {"suggestions": parsed.get("suggestions", [])}


def make_keyword_optimizer_tool(groq_client, model: str = "openai/gpt-oss-20b"):
    """Factory so the tool can close over groq_client — same pattern as make_ats_score_tool."""
    @tool
    def keyword_optimizer_tool(resume_text: str, job_description: str) -> str:
        """
        Suggest how to add missing ATS keywords to a resume — which section
        to add each one to and an example bullet phrasing. Use this when the
        user asks how to improve their resume's keyword match, not just what
        the score is. Requires both resume text and job description.
        """
        keyword_result = calculate_keyword_score(resume_text, job_description, groq_client, model)
        missing = keyword_result.get("missing", [])
        if not missing:
            return "No missing keywords detected — resume already covers the job description's key terms."

        suggestions_result = generate_keyword_suggestions(
            resume_text, job_description, missing, groq_client, model
        )
        suggestions = suggestions_result.get("suggestions", [])
        if not suggestions:
            return "Missing keywords: " + ", ".join(missing) + " (could not generate detailed suggestions)"

        lines = []
        for s in suggestions:
            lines.append(f"- {s.get('keyword')}: add to {s.get('section')} — e.g. \"{s.get('example_bullet')}\"")
        return "\n".join(lines)

    return keyword_optimizer_tool 

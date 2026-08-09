import os
import streamlit as st
import requests
from ats_scoring import calculate_ats_score, make_ats_score_tool
from keyword_optimizer import generate_keyword_suggestions, make_keyword_optimizer_tool
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

import keyword_optimizer


load_dotenv()
APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# --- Session state init ---
if "job_description" not in st.session_state:
    st.session_state.job_description = ""


# --- Functions ---
def search_jobs(query, location="Bangalore", results=10):
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "what": query,
        "where": location,
        "results_per_page": results,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["results"]


def analyze_resume_match(resume_text, job_description):
    prompt = f"""You are a career advisor. Compare this resume against this job description.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Provide:
1. A match score out of 100
2. Top 3 matching strengths
3. Top 3 gaps or missing keywords
4. One specific suggestion to improve the resume for this role

Keep it concise."""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Analysis failed: {e}\n\nTry again in a moment."


# --- LangChain tools (thin wrappers around the functions above) ---
# These don't change search_jobs / analyze_resume_match at all — they just
# give an LLM agent a name + description so it can decide which one to call.

@tool
def job_search_tool(query: str, location: str = "Bangalore") -> str:
    """Search for job listings in India on Adzuna, given a job title/role
    (e.g. 'Python Developer') and a location (e.g. 'Bangalore'). Returns a
    short list of matching job titles, companies, and locations."""
    try:
        jobs = search_jobs(query, location)
    except Exception as e:
        return f"Job search failed: {e}"
    if not jobs:
        return "No jobs found for that search."
    lines = []
    for job in jobs[:10]:
        title = job.get("title", "N/A")
        company = job.get("company", {}).get("display_name", "N/A")
        loc = job.get("location", {}).get("display_name", "N/A")
        lines.append(f"- {title} at {company} ({loc})")
    return "\n".join(lines)


@tool
def resume_match_tool(resume_text: str, job_description: str) -> str:
    """Compare a candidate's resume text against a job description and
    return a match score out of 100, top matching strengths, gaps, and one
    improvement suggestion. Requires both the full resume text and the full
    job description text as input."""
    return analyze_resume_match(resume_text, job_description)


def build_agent_executor():
    """Builds a LangChain agent that owns all tools and decides which
    one(s) to call based on the user's natural-language request.
    Uses langchain 1.x's create_agent (the current API — the older
    create_tool_calling_agent + AgentExecutor pattern from 0.x LangChain
    was removed)."""
    agent_llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

    # ATS scoring tool — built via the factory so it can close over `client`
    # (the same Groq client used everywhere else in this file), same pattern
    # as how resume_match_tool reuses analyze_resume_match/client above.
    ats_score_tool = make_ats_score_tool(client)
    keyword_optimizer_tool = make_keyword_optimizer_tool(client)

    tools = [job_search_tool, resume_match_tool, ats_score_tool,keyword_optimizer_tool]
    system_prompt = (
        "You are a job search assistant for the Indian job market. "
        "You have three tools: one to search job listings, one to "
        "compare a resume against a job description, and one to calculate "
        "an ATS compatibility score for a resume against a job description. "
        "Use whichever tool(s) fit the user's request. If the user's message includes "
        "resume text and/or a job description, pass that full text through "
        "to resume_match_tool or ats_score_tool rather than summarizing it yourself."
    )
    return create_agent(agent_llm, tools, system_prompt=system_prompt)


    # --- UI starts here ---
st.set_page_config(page_title="Job Search AI Agent", page_icon="🔍")
st.title("🔍 Job Search AI Agent")
st.write("Find jobs across India.")

# --- Resume Upload ---
st.header("1. Upload Your Resume")
resume_text = ""
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        resume_text += page.extract_text() or ""

    st.success(f"Resume parsed — {len(resume_text)} characters extracted")
    with st.expander("Preview extracted text"):
        st.text(resume_text[:1000])

# --- Job Search ---
st.header("2. Search for Jobs")
col1, col2 = st.columns(2)
with col1:
    query = st.text_input("What job are you looking for?", "Python Developer")
with col2:
    location = st.text_input("Location", "Bangalore")

if st.button("Search Jobs"):
    with st.spinner("Searching..."):
        try:
            jobs = search_jobs(query, location)
            if not jobs:
                st.warning("No jobs found. Try a different search.")
            else:
                st.success(f"Found {len(jobs)} jobs")
                for job in jobs:
                    with st.container(border=True):
                        st.subheader(job["title"])
                        st.write(f"**Company:** {job.get('company', {}).get('display_name', 'N/A')}")
                        st.write(f"**Location:** {job.get('location', {}).get('display_name', 'N/A')}")
                        salary_min = job.get("salary_min")
                        salary_max = job.get("salary_max")
                        if salary_min and salary_max:
                            st.write(f"**Salary:** ₹{salary_min:,.0f} - ₹{salary_max:,.0f}")
                        st.write(job["description"][:300] + "...")
                        st.link_button("View Job", job["redirect_url"])
                        if st.button("Use this job for resume match", key=job["id"]):
                            st.session_state.job_description = job["description"]
                            st.success("Loaded below — scroll down to Resume Match section")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# --- Job Description Input ---
st.header("3. Job Description")
job_description = st.text_area(
    "Paste a job description, or click 'Use this job' above to auto-fill",
    value=st.session_state.job_description,
    height=200,
    key="jd_textarea",
)

# --- Resume Match Analysis ---
st.header("4. Resume Match Analysis")
if st.button("Analyze Match"):
    if not resume_text:
        st.warning("Upload a resume first")
    elif not job_description:
        st.warning("Add a job description first (paste or select a job above)")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_resume_match(resume_text, job_description)
        st.markdown(result)

# --- AI Agent (LangChain) ---
st.header("5. Ask the AI Agent")
st.caption(
    "Ask in plain English — e.g. \"find backend developer jobs in Pune\" or "
    "\"how well does my resume match this role\". The agent decides on its "
    "own whether to search jobs, analyze your resume, check your ATS score, or "
    "any combination of these."
)

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = build_agent_executor()

agent_query = st.text_area(
    "Your request",
    placeholder="e.g. Find me data analyst jobs in Hyderabad",
    key="agent_query_box",
)

if st.button("Ask Agent"):
    if not agent_query:
        st.warning("Type a request first")
    else:
        # Pass along whatever context is already loaded on the page, so the
        # agent can use resume_match_tool / ats_score_tool without the user
        # retyping anything.
        context_parts = [agent_query]
        if resume_text:
            context_parts.append(f"My resume text:\n{resume_text}")
        if job_description:
            context_parts.append(f"Job description in context:\n{job_description}")
        full_input = "\n\n".join(context_parts)

        with st.spinner("Agent is thinking..."):
            try:
                result = st.session_state.agent_executor.invoke(
                    {"messages": [{"role": "user", "content": full_input}]}
                )
                final_message = result["messages"][-1].content
                st.markdown(final_message)
            except Exception as e:
                st.error(f"Agent failed: {e}")

# --- ATS Score ---
st.header("6. ATS Score")
st.markdown(
    "Get an ATS compatibility score for your resume against a specific job "
    "description — covers keyword matching and format checks that real "
    "applicant tracking systems look for."
)

if st.button("Calculate ATS Score"):
    if not resume_text:
        st.warning("Upload a resume first")
    elif not job_description:
        st.warning("Add a job description first (paste or select a job above)")
    else:
        with st.spinner("Scoring resume against job description..."):
            try:
                result = calculate_ats_score(resume_text, job_description, client)
                st.session_state.ats_result = result

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

# --- Keyword Optimization Suggestions ---
st.header("7. Keyword Optimization Suggestions")
st.markdown(
    "Get concrete suggestions for where and how to add the keywords your "
    "resume is missing, based on your last ATS score run above."
)

if st.button("Get Optimization Suggestions"):
    ats_result = st.session_state.get("ats_result")
    if not ats_result:
        st.warning("Run 'Calculate ATS Score' above first")
    elif not ats_result["missing_keywords"]:
        st.success("No missing keywords — nothing to optimize!")
    else:
        with st.spinner("Generating suggestions..."):
            try:
                suggestions_result = generate_keyword_suggestions(
                    resume_text, job_description, ats_result["missing_keywords"], client
                )
                suggestions = suggestions_result.get("suggestions", [])
                if not suggestions:
                    st.warning(suggestions_result.get("error", "No suggestions generated — try again"))
                else:
                    for s in suggestions:
                        with st.container(border=True):
                            st.write(f"**{s.get('keyword')}** → add to *{s.get('section')}*")
                            st.caption(s.get("example_bullet"))
            except Exception as e:
                st.error(f"Could not generate suggestions: {e}")

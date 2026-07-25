import os
import streamlit as st
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent


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
    """Builds a LangChain agent that owns both tools and decides which
    one(s) to call based on the user's natural-language request.
    Uses langchain 1.x's create_agent (the current API — the older
    create_tool_calling_agent + AgentExecutor pattern from 0.x LangChain
    was removed)."""
    agent_llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))
    tools = [job_search_tool, resume_match_tool]
    system_prompt = (
        "You are a job search assistant for the Indian job market. "
        "You have two tools: one to search job listings, and one to "
        "compare a resume against a job description. Use whichever "
        "tool(s) fit the user's request. If the user's message includes "
        "resume text and/or a job description, pass that full text through "
        "to resume_match_tool rather than summarizing it yourself."
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
    "own whether to search jobs, analyze your resume, or both."
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
        # agent can use resume_match_tool without the user retyping anything.
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

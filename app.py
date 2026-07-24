import os
import streamlit as st
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq


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

# Job-search-AI-agent

# Job Search AI Agent

An AI-powered career assistant for the Indian job market. It searches live job listings, parses and matches resumes against job descriptions, scores resumes for ATS compatibility, suggests keyword optimizations, looks up company information, and lets users save and track job applications — all through both a guided UI and a conversational LangChain agent.

**Live app:** https://job-search-ai-agent1.streamlit.app/

## Features

- **Job Search** — live job listings via the Adzuna API, filterable by role and location
- **Resume Parsing** — upload a PDF resume, text extracted with `pypdf`
- **Resume-vs-JD Match Analysis** — compares a resume against a job description and scores the fit (Groq LLM)
- **ATS Compatibility Scoring** — scores a resume's compatibility with automated applicant tracking systems
- **Keyword Optimization** — suggests keywords to add based on a target job description
- **Company Lookup** — pulls public company information (Wikipedia API)
- **Application Tracking** — save jobs and track their status (Interested / Applied / Interview / Offer / Rejected) in a local SQLite database
- **Conversational Agent** — a LangChain agent (`create_agent`, LangChain 1.x) that owns all six tools above and picks the right one(s) based on natural-language requests, e.g. "what have I applied to?" or "score my resume against this JD"

## Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq (`openai/gpt-oss-20b` via `langchain-groq`)
- **Agent framework:** LangChain 1.x (`create_agent`)
- **Job data:** Adzuna API
- **Company data:** Wikipedia API
- **Resume parsing:** pypdf
- **Database:** SQLite
- **Deployment:** Streamlit Community Cloud

## Setup & Installation

### Prerequisites
- Python 3.13
- An [Adzuna API](https://developer.adzuna.com/) App ID and App Key (free tier)
- A [Groq API](https://console.groq.com/) key (free tier)

### 1. Clone the repository
```bash
git clone https://github.com/1innocentcoconut/Job-search-AI-agent.git
cd Job-search-AI-agent
```

### 2. Create a virtual environment and install dependencies
```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API keys
Create a `.env` file in the project root:
```
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_APP_KEY=your_adzuna_app_key
GROQ_API_KEY=your_groq_api_key
```
The app reads keys via `os.getenv()` locally, with a fallback to Streamlit's `st.secrets` when deployed on Streamlit Cloud — no code changes needed between local and deployed environments.

### 4. Run the app
```bash
streamlit run app.py
```
The SQLite database (`job_tracker.db`) is created automatically on first run — no manual setup needed.

The app will open at `http://localhost:8501`.

## Project Structure

```
.
├── app.py                 # Main Streamlit app — UI, agent setup, tool orchestration
├── database.py             # SQLite layer — saved_jobs table, tracked_jobs_tool
├── ats_scoring.py           # ATS compatibility scoring, ats_score_tool
├── keyword_optimizer.py     # Keyword suggestion logic, keyword_optimizer_tool
├── company_lookup.py        # Wikipedia-based company info, company_lookup_tool
├── requirements.txt
└── .env                    # API keys (not committed)
```

## Notes

- Deployed on Streamlit Community Cloud; the SQLite database resets on redeploy since Cloud's filesystem is ephemeral — application tracking data persists within a session/deployment but not across redeploys.
- API keys are never committed to the repository; `.env` and `job_tracker.db` are both git-ignored.

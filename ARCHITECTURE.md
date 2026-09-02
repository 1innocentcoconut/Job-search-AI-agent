# Architecture & Data Flow

This document explains how data moves through the Job Search AI Agent — separate from the README, which covers setup and usage. It's organized by feature, since the app is a single Streamlit script (`app.py`) with focused helper modules rather than a layered service architecture.

## Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   app.py     │────▶│  External    │
│  UI (9      │◀────│  (functions  │◀────│  APIs:       │
│  sections)  │     │  + tools)    │     │  Adzuna,     │
└─────────────┘     └──────┬───────┘     │  Wikipedia,  │
                            │             │  Groq        │
                     ┌──────┴───────┐     └─────────────┘
                     │  SQLite      │
                     │ (job_tracker.│
                     │     db)      │
                     └──────────────┘
```

Every feature is exposed **twice**: once as a direct UI action (a button that calls a plain Python function), and once as a `@tool`-decorated wrapper the LangChain agent can call on its own. Both paths call the *same* underlying function, so there's one source of truth for each capability — the tool wrappers don't duplicate logic, they just add a docstring the LLM uses to decide when to invoke them.

## 1. Job Search

**Flow:** UI (Section 2) → `search_jobs()` → Adzuna API → results rendered as cards

- `search_jobs(query, location, results)` in `app.py` builds a GET request to `https://api.adzuna.com/v1/api/jobs/in/search/1` with `app_id`/`app_key` (loaded via `get_secret()`, see Configuration below) and returns the `results` list from the JSON response.
- The **UI path**: clicking "Search Jobs" calls `search_jobs()` directly and stores the list in `st.session_state.job_results`, so results survive Streamlit's rerun-on-interaction behavior (e.g. clicking "Save this job" on one card won't wipe the others).
- The **agent path**: `job_search_tool` (a thin `@tool`-wrapped function) calls the same `search_jobs()` and formats the results as a short text list for the LLM to read and relay conversationally.

## 2. Resume Parsing

**Flow:** PDF upload (Section 1) → `pypdf.PdfReader` → plain text in local `resume_text` variable

- Runs once per page load, not stored in `st.session_state` — so it doesn't persist if the user navigates away, but that's fine since it's re-extracted instantly from the uploaded file on each rerun while the file stays attached.
- The extracted text feeds directly into Resume Match (Section 4), ATS Scoring (Section 6), and is also attached to the agent's context whenever the user asks it something (Section 5) — see "Agent context" below.

## 3. Resume-vs-JD Match Analysis

**Flow:** `resume_text` + `job_description` → `analyze_resume_match()` → Groq LLM call → markdown result

- A single prompt asks the model for a match score, top strengths, gaps, and one improvement suggestion.
- Same dual-path pattern: the UI's "Analyze Match" button and the agent's `resume_match_tool` both call `analyze_resume_match()` directly.

## 4. ATS Compatibility Scoring

**Flow:** `resume_text` + `job_description` → `calculate_ats_score()` (in `ats_scoring.py`) → combined score

This is the most involved feature — it's two independent checks combined into one score out of 100:

- **Format score (30 pts, deterministic, no LLM call)** — `calculate_format_score()` uses regex to check for standard section headers (Experience/Education/Skills), an email, a phone number, and flags signs of broken text extraction (e.g. tables/columns that confuse both `pypdf` and real ATS parsers).
- **Keyword score (70 pts, uses Groq)** — `calculate_keyword_score()` asks the LLM to extract 15-20 JD-critical keywords, then checks which appear in the resume (allowing close variants like "JS" / "JavaScript"). Score is proportional to the fraction matched.
- `calculate_ats_score()` just adds the two together and returns matched/missing keyword lists alongside the totals.
- The result is cached in `st.session_state.ats_result` after a UI run — this matters because Section 7 (Keyword Optimization) depends on it existing rather than re-running the scoring itself.

## 5. Keyword Optimization

**Flow:** `st.session_state.ats_result["missing_keywords"]` → `generate_keyword_suggestions()` (in `keyword_optimizer.py`) → per-keyword suggestions

- Deliberately **depends on ATS Scoring having already run** — the UI button reads the missing-keyword list out of session state rather than recomputing it, so if a user hasn't run "Calculate ATS Score" yet, they're prompted to do that first.
- For each missing keyword, one Groq call asks: which resume section it belongs in, and an example bullet — explicitly phrased as a suggestion for the candidate to adapt, not a fabricated claim about their experience (see the prompt's instructions in `generate_keyword_suggestions()`).
- The agent's `keyword_optimizer_tool` is independent of this session-state dependency — since the agent has no UI state to read from, it recomputes the keyword score itself via `calculate_keyword_score()` before generating suggestions, so it works standalone from a single conversational request.

## 6. Company Lookup

**Flow:** company name → `get_company_info()` (in `company_lookup.py`) → Wikipedia REST API → summary + link

- No API key required — hits Wikipedia's public `page/summary` endpoint directly.
- Returns a structured `{"found": bool, ...}` dict so both the UI and the agent tool can handle "not found" cleanly without exceptions.

## 7. Application Tracking (Database)

**Flow:** "Save this job" (Section 2) → `save_job()` → SQLite `saved_jobs` table → read back in Section 9 and by the agent's `tracked_jobs_tool`

- `database.py` owns a single table, `saved_jobs` (title, company, location, salary, link, status, date_saved), created on startup by `init_db()` (called once at the top of `app.py`).
- `save_job()` checks for an existing row matching (title, company, link) before inserting, so re-clicking "Save this job" on the same listing doesn't create duplicates.
- Section 9 ("Tracked Applications") reads all saved jobs via `get_saved_jobs()`, renders each as a card with a status dropdown (`interested → applied → interview → offer/rejected`) wired to `update_job_status()`, and a remove button wired to `delete_job()`.
- The agent's `tracked_jobs_tool` (built via `make_tracked_jobs_tool()`, a factory that closes over the DB path) wraps `get_saved_jobs()` so a user can ask the agent "what have I applied to?" and get a plain-text summary, optionally filtered by status.
- **Note:** on Streamlit Community Cloud, the filesystem is ephemeral, so `job_tracker.db` resets on redeploy — saved jobs persist within a deployment but not across one (documented in the README).

## 8. The Conversational Agent

**Flow:** free-text request (Section 5) → `build_agent_executor()` → LangChain `create_agent` → LLM tool-selection → tool call(s) → final message

- Built once per session and cached in `st.session_state.agent_executor` (so the agent isn't rebuilt on every rerun).
- Uses LangChain 1.x's `create_agent(agent_llm, tools, system_prompt=...)` — the current API, replacing the older `create_tool_calling_agent` + `AgentExecutor` pattern from LangChain 0.x.
- **Tool selection is pure LLM judgment** — there's no manual routing/if-else logic in this app. The system prompt lists all six tools and their purposes; the underlying Groq model (`openai/gpt-oss-20b`) decides which tool(s) to call based on the user's message, using each tool's docstring as its description.
- **The six tools**, all thin wrappers around the same functions used by the UI: `job_search_tool`, `resume_match_tool`, `ats_score_tool`, `keyword_optimizer_tool`, `company_lookup_tool`, `tracked_jobs_tool`.
- **Agent context injection:** when the user submits a request, `app.py` prepends whatever's already loaded on the page — the parsed `resume_text` and any `job_description` currently in the text area — to the user's message before sending it to the agent. This means a user can upload a resume, paste a JD, then just ask "how do I improve my ATS score for this?" without retyping either — the agent receives both automatically as part of its input.

## Configuration & Secrets

- `get_secret(key)` in `app.py` tries `os.getenv(key)` first (for local development via a `.env` file), then falls back to `st.secrets.get(key)` (for Streamlit Cloud), wrapped in a try/except since `st.secrets` raises `StreamlitSecretNotFoundError` rather than returning `None` when no secrets file exists at all.
- This lets the exact same code run unmodified locally and on Streamlit Cloud.

## Error Handling

Error handling is currently local/ad-hoc rather than centralized: each Streamlit button handler wraps its API/LLM call in its own `try/except` and surfaces failures via `st.error()` or `st.warning()` (e.g. job search failures, agent failures, ATS scoring failures each have their own catch block). There's no shared error-handling layer — this is a known simplification, not an oversight, given the project's scope and timeline.

import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

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

# --- UI starts here ---
st.set_page_config(page_title="Job Search AI Agent", page_icon="🔍")
st.title("🔍 Job Search AI Agent")
st.write("Find jobs across India, powered by Adzuna.")

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
        except Exception as e:
            st.error(f"Something went wrong: {e}")

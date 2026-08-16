"""
Company Lookup feature for Job Search AI Agent (Track A - "company research" requirement)

Uses Wikipedia's public REST API — no API key needed, no signup, no rate-limit hassle.
"""

import requests
from langchain_core.tools import tool


def get_company_info(company_name: str) -> dict:
    """
    Looks up a company on Wikipedia and returns a short summary + link.
    Returns {"found": False, "reason": ...} if nothing matches.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{company_name}"
    headers = {
        "User-Agent": "JobSearchAIAgent/1.0 (student project; contact: your-email@example.com)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 404:
            return {"found": False, "reason": f"No Wikipedia page found for '{company_name}'"}
        response.raise_for_status()
        data = response.json()
        return {
            "found": True,
            "title": data.get("title", company_name),
            "extract": data.get("extract", "No summary available."),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "thumbnail": data.get("thumbnail", {}).get("source"),
        }
    except requests.exceptions.RequestException as e:
        return {"found": False, "reason": f"Lookup failed: {e}"}


def make_company_lookup_tool():
    @tool
    def company_lookup_tool(company_name: str) -> str:
        """
        Look up basic public information about a company by name (what they
        do, industry, background). Use this when the user asks about a
        company they found in job listings or wants to research an employer.
        """
        result = get_company_info(company_name)
        if not result["found"]:
            return result["reason"]
        return f"{result['title']}: {result['extract']}\nMore: {result['url']}"

    return company_lookup_tool

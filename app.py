"""
ScholarTrack - AI financial-aid research advisor
================================================

How a request flows:

  Browser -> POST /api/plan
          -> step 1: search the web ourselves (DuckDuckGo, no API key needed)
          -> step 2: Gemini reads those real pages and writes the research
          -> step 3: Gemini turns that research into structured JSON
          -> dashboard data back to the browser

Why we search ourselves instead of using Google's built-in grounding: grounding has
a separate quota that free accounts often can't use. Doing the search in our own
server removes that dependency, and it means every citation is a URL we actually
found rather than one the model remembered.

Everything degrades safely. If search fails, the app falls back to LIMITED MODE,
which forbids amounts, deadlines and links rather than guessing at them.
"""

import os
import json
import time
from typing import Optional, List, Dict

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# The search library is optional. If it isn't installed or fails to import,
# the app still runs - it just stays in limited mode.
try:
    from ddgs import DDGS
    SEARCH_AVAILABLE = True
except Exception as _import_error:      # pragma: no cover
    print(f"Search library unavailable: {_import_error}")
    DDGS = None
    SEARCH_AVAILABLE = False

app = FastAPI(title="ScholarTrack")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Sources we trust most. Results from these get shown to the AI first.
PREFERRED_DOMAINS = (
    ".gov", ".edu",
    "studentaid.gov", "fafsa.gov", "ed.gov",
    "collegeboard.org", "nasfaa.org", "ncaa.org",
)


# ----------------------------------------------------------------------------
# What the browser sends us
# ----------------------------------------------------------------------------

class RequestData(BaseModel):
    mode: str
    query: str
    state: Optional[str] = None
    school: Optional[str] = None
    major: Optional[str] = None
    gpa: Optional[str] = None
    income_range: Optional[str] = None
    household_size: Optional[int] = None
    student_athlete: bool = False
    sport: Optional[str] = None
    recruiting_status: Optional[str] = None
    notes: Optional[str] = None


# ----------------------------------------------------------------------------
# Instructions for the AI
# ----------------------------------------------------------------------------

SYSTEM_RESEARCH = """
You are ScholarTrack, a college financial-aid research assistant.

You are given SEARCH RESULTS retrieved from the live web moments ago. Base your
answer on those results.

SOURCE RULES:
- You may ONLY cite URLs that appear in the SEARCH RESULTS block. Never write a URL
  that is not in that block, even if you believe it exists.
- If the search results do not answer something, say so plainly. Do not fill the gap
  from memory.
- Prefer official sources: .gov, the college's own financial-aid pages, and the
  scholarship provider's own site, over blogs and listing sites.
- Search results contain short snippets, not full pages. Treat every amount and date
  as needing confirmation by the student.

CONFIDENCE - label every opportunity as exactly one of:
- confirmed  : a source in the results explicitly states this student meets the criteria
- potential  : the student's profile suggests it may apply, but it needs confirming
- unverified : the results do not contain enough information to judge

Never say "you qualify". Say "you may be eligible based on the published criteria -
verify with the provider".

Never invent a scholarship, amount, deadline, requirement or source. Never recommend
falsifying or hiding FAFSA information. Never request SSNs, FAFSA passwords,
tax-account credentials, or bank credentials.

Cover, where relevant: FAFSA, federal aid, state aid, institutional/university aid,
outside scholarships, athletic aid and NIL. For each opportunity give the provider,
amount, deadline, eligibility, required documents, application steps, why it may apply
to this student, what they must verify, and the source URL.

Do not rank opportunities as "best". Do not promise a total award. Do not calculate an
official Student Aid Index.
"""

SYSTEM_NO_SEARCH = """
You are ScholarTrack running in LIMITED MODE.

You have NO web access for this request. You cannot verify anything, and financial-aid
details change every year, so you must not present remembered details as fact.

HARD RULES - these override any other instruction:
1. Do NOT output any URL, web address, or domain name.
2. Do NOT use citation markers such as [1], (source), or "Official Source:".
3. Do NOT state any specific dollar amount, award size, deadline date, GPA cutoff,
   income threshold, or number of recipients.
4. Do NOT write "based on official sources" or "confirmed". Nothing here is confirmed.
5. Do NOT invent program names. Name only large, long-running programs you are
   confident exist, and only in general terms.

INSTEAD: explain what category of aid applies and how it generally works, what
generally determines eligibility (in words, no numbers), what documents are usually
needed, which official organizations the student should look up themselves, and what
questions to ask their financial-aid office.

Never recommend falsifying or hiding FAFSA information. Never request SSNs, FAFSA
passwords, tax-account credentials, or bank credentials.
"""

SYSTEM_STRUCTURE = """
You convert a financial-aid research report into structured JSON.

CRITICAL: You are a formatter, not a researcher. Use ONLY information present in the
report you are given. Never add a program, amount, deadline, URL or requirement that
is not already in that text. If a field is not stated, use null or an empty list.

Return ONLY valid JSON. No markdown fences, no commentary.

Shape:

{
  "summary": "2-3 plain-language sentences on what this student should focus on",
  "opportunities": [
    {
      "id": "short-slug",
      "name": "Program name exactly as written in the report",
      "category": "federal" | "state" | "university" | "scholarship" | "athletic",
      "provider": "Who gives the money, or null",
      "amount": "Award amount as written, or null",
      "deadline": "YYYY-MM-DD if an exact date is given, else null",
      "deadline_text": "Deadline as written, or null",
      "confidence": "confirmed" | "potential" | "unverified",
      "why_it_may_apply": "Why this student should look at it",
      "eligibility": ["criterion"],
      "required_documents": ["document"],
      "application_steps": ["step"],
      "verify": "What the student must confirm themselves, or null",
      "source_name": "Name of the source, or null",
      "source_url": "URL only if one appears in the report, else null"
    }
  ],
  "action_plan": [
    { "order": 1, "task": "What to do", "due": "YYYY-MM-DD or null", "related_to": "opportunity id or null" }
  ],
  "verify_notes": ["Things the student must confirm before relying on any of this"]
}

If the report is limited-mode general guidance with no specific programs, return an
empty "opportunities" list, put the guidance in "summary", and put the suggested
actions in "action_plan". Never fabricate opportunities to fill the list.
"""


# ----------------------------------------------------------------------------
# Step 1: search the web ourselves
# ----------------------------------------------------------------------------

def build_queries(d: RequestData) -> List[str]:
    """Turn the student's request into a handful of targeted searches."""
    queries: List[str] = []
    q = d.query.strip()

    if d.mode == "scholarship":
        queries += [f"{q} official scholarship eligibility deadline",
                    f"{q} how to apply official site"]
    elif d.mode == "find":
        queries += [f"{q} scholarships official application",
                    f"{q} scholarship deadline eligibility"]
    elif d.mode == "college":
        queries += [f"{q} official financial aid office scholarships grants",
                    f"{q} institutional aid priority deadline FAFSA"]
    elif d.mode == "athlete":
        queries += [f"{q} athletic scholarship official NCAA eligibility",
                    f"{q} NIL policy official"]
    else:
        queries += [f"{q} financial aid scholarships grants official"]

    if d.school:
        queries.append(f"{d.school} financial aid office official FAFSA priority deadline")
    if d.state:
        queries.append(f"{d.state} state grant program official .gov financial aid")

    queries.append("FAFSA federal student aid grants studentaid.gov official")

    # Keep it to 5 searches so the request doesn't take forever.
    seen, unique = set(), []
    for item in queries:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique.append(item)
    return unique[:5]


def rank(result: Dict) -> int:
    """Lower sorts first. Official domains win."""
    url = (result.get("url") or "").lower()
    for i, domain in enumerate(PREFERRED_DOMAINS):
        if domain in url:
            return i
    return len(PREFERRED_DOMAINS)


def web_search(queries: List[str], per_query: int = 5) -> List[Dict]:
    """
    Run the searches. Returns [] on any failure - the caller treats that as
    'no search available' and drops to limited mode.
    """
    if not SEARCH_AVAILABLE:
        return []

    found: Dict[str, Dict] = {}
    for query in queries:
        try:
            with DDGS() as ddgs:
                for row in ddgs.text(query, max_results=per_query):
                    url = row.get("href") or row.get("url") or ""
                    if not url.startswith("http") or url in found:
                        continue
                    found[url] = {
                        "title":   (row.get("title") or "").strip()[:200],
                        "url":     url,
                        "snippet": (row.get("body") or "").strip()[:500],
                    }
        except Exception as e:
            print(f"Search failed for '{query}': {type(e).__name__}: {str(e)[:160]}")
            continue

    results = sorted(found.values(), key=rank)
    return results[:24]


def format_results(results: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Step 2 & 3: talk to Gemini, with retries
# ----------------------------------------------------------------------------

def call_model(client, model, contents, system, want_json=False, attempts=3):
    """
    One logical call to Gemini, retried on temporary server errors.
    503 (busy) and 429 (rate limited) usually clear in seconds, so those retry.
    Anything else is a real error and raises immediately.
    """
    config_args = {"system_instruction": system}
    if want_json:
        config_args["response_mime_type"] = "application/json"

    for attempt in range(attempts):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_args),
            )
        except Exception as e:
            text = str(e)
            temporary = ("503" in text or "UNAVAILABLE" in text
                         or "429" in text or "RESOURCE_EXHAUSTED" in text)
            if not temporary or attempt == attempts - 1:
                raise
            wait = 2 * (attempt + 1)      # 2s, then 4s
            print(f"Temporary error (attempt {attempt + 1}), retrying in {wait}s: {text[:160]}")
            time.sleep(wait)


def parse_json(text: str):
    """Gemini usually returns clean JSON, but strip code fences just in case."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) > 1:
            cleaned = parts[1]
            if cleaned.lstrip().lower().startswith("json"):
                cleaned = cleaned.lstrip()[4:]
    try:
        return json.loads(cleaned.strip())
    except Exception as e:
        print(f"Could not parse JSON: {e}")
        return None


def build_profile(d: RequestData) -> str:
    bits = [
        ("State", d.state), ("College", d.school), ("Major", d.major),
        ("GPA", d.gpa), ("Household income range", d.income_range),
        ("Household size", d.household_size), ("Sport", d.sport),
        ("Recruiting status", d.recruiting_status), ("Other notes", d.notes),
    ]
    lines = [f"{k}: {v}" for k, v in bits if v not in (None, "", 0)]
    if d.student_athlete:
        lines.append("Student athlete: yes")
    return "\n".join(lines) if lines else "(No profile details provided.)"


def build_task(d: RequestData) -> str:
    tasks = {
        "scholarship": f"Analyze this specific scholarship or funding opportunity: {d.query}",
        "find":        f"Find current scholarships matching this request: {d.query}",
        "college":     f"Analyze current financial aid and funding opportunities at this college: {d.query}",
        "athlete":     f"Find and explain legitimate athletic scholarship and NIL funding opportunities relevant to: {d.query}",
    }
    return tasks.get(d.mode, f"Build a personalized college funding plan for: {d.query}")


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/api/health")
def health():
    """Is the server awake and configured? Safe to open in a browser - shows no key."""
    return {
        "status": "ok",
        "model": os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        "key_configured": bool(os.getenv("GOOGLE_API_KEY")),
        "search_library_installed": SEARCH_AVAILABLE,
    }


@app.get("/api/search-check")
def search_check():
    """
    Open this in a browser to find out, in a few seconds, whether live web search
    works from this server. Use it before a demo.
    """
    if not SEARCH_AVAILABLE:
        return {"search_working": False, "reason": "ddgs library is not installed"}
    results = web_search(["FAFSA official site studentaid.gov"], per_query=3)
    return {
        "search_working": len(results) > 0,
        "results_found": len(results),
        "sample": results[:3],
    }


@app.post("/api/plan")
def plan(data: RequestData):
    """Main endpoint: returns a structured financial-aid plan."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return JSONResponse(status_code=500, content={"error": "GOOGLE_API_KEY is not set."})

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    task = build_task(data)
    profile = build_profile(data)

    try:
        client = genai.Client(api_key=key)

        # --- step 1: search --------------------------------------------------
        results = web_search(build_queries(data))
        used_search = len(results) > 0

        # --- step 2: research ------------------------------------------------
        if used_search:
            contents = (
                f"{task}\n\n"
                f"Student profile:\n{profile}\n\n"
                f"SEARCH RESULTS (retrieved just now - these are your only allowed sources):\n\n"
                f"{format_results(results)}"
            )
            research = call_model(client, model, contents, SYSTEM_RESEARCH)
        else:
            print("No search results. Using limited mode.")
            contents = f"{task}\n\nStudent profile:\n{profile}"
            research = call_model(client, model, contents, SYSTEM_NO_SEARCH)

        report = (research.text or "").strip()
        if not report:
            return JSONResponse(status_code=502, content={"error": "The AI returned no text."})

        # --- step 3: structure -----------------------------------------------
        structured = None
        try:
            shaped = call_model(
                client, model,
                f"Convert this research report into JSON.\n\nREPORT:\n{report}",
                SYSTEM_STRUCTURE,
                want_json=True,
            )
            structured = parse_json(shaped.text)
        except Exception as e:
            print(f"Structuring pass failed: {e}")
        # If structuring fails the dashboard still renders `report`, so this
        # degrades gracefully rather than breaking the page.

        return {
            "used_search": used_search,
            "sources_searched": len(results),
            "plan": structured,
            "report": report,
            "student": {
                "query": data.query,
                "mode": data.mode,
                "college": data.school,
                "state": data.state,
            },
        }

    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"{type(e).__name__}: {str(e)}"}
        )


@app.post("/api/research")
def research_endpoint(data: RequestData):
    """Older plain-text endpoint, kept as a fallback."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return JSONResponse(status_code=500, content={"error": "GOOGLE_API_KEY is not set."})

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    task = build_task(data)
    profile = build_profile(data)

    try:
        client = genai.Client(api_key=key)
        results = web_search(build_queries(data))
        used_search = len(results) > 0

        if used_search:
            contents = (f"{task}\n\nStudent profile:\n{profile}\n\n"
                        f"SEARCH RESULTS (your only allowed sources):\n\n{format_results(results)}")
            response = call_model(client, model, contents, SYSTEM_RESEARCH)
        else:
            response = call_model(client, model, f"{task}\n\nStudent profile:\n{profile}",
                                  SYSTEM_NO_SEARCH)

        report = (response.text or "").strip()
        if not report:
            return JSONResponse(status_code=502, content={"error": "The AI returned no text."})
        return {"report": report, "used_search": used_search}

    except Exception as e:
        return JSONResponse(status_code=502,
                            content={"error": f"{type(e).__name__}: {str(e)}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

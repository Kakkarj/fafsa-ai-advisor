import os
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="ScholarTrack")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

SYSTEM = """
You are a current-information college funding research assistant.

You research scholarships, financial aid, grants, athletic aid, and NIL opportunities.
Use web search and prioritize official/primary sources. For scholarships, prefer the
official scholarship provider's website. For college aid, prefer the college's official
financial-aid pages and StudentAid.gov. For state programs, use the official state agency.

Never invent a scholarship, amount, deadline, eligibility requirement, NIL opportunity,
or source. Never recommend falsifying or hiding FAFSA information. Never request SSNs,
FAFSA passwords, tax-account credentials, or bank credentials.

Clearly distinguish:
- Confirmed official facts
- Possible eligibility
- Estimated/variable amounts
- NIL opportunities, which are not guaranteed financial aid

Every important current claim must have a source URL.

When analyzing one scholarship, return:
NAME, PROVIDER, AWARD, DEADLINE, ELIGIBILITY, REQUIREMENTS, APPLICATION STEPS,
FIT FOR THIS STUDENT, IMPORTANT NOTES, CONFIDENCE, OFFICIAL SOURCE.

When finding scholarships, return a table/list with:
NAME, AWARD, DEADLINE, ELIGIBILITY, WHY IT MATCHES, ACTION, OFFICIAL SOURCE.
Do not rank scholarships as "best"; organize them by match/category instead.

When analyzing a college, cover:
FAFSA/SAI, federal aid, state aid, institutional grants, merit scholarships,
outside scholarships, athletic aid, and NIL if relevant.

When creating a funding plan, organize:
KNOWN/PUBLISHED AID, POTENTIAL AID, ACTION ITEMS, DEADLINES, and SOURCES.
Do not promise a total award or calculate an official SAI.
"""

NO_SEARCH_WARNING = (
    "NOTE: Live web search was unavailable for this request, so nothing below "
    "was verified against current official sources. Treat every program name, "
    "amount and deadline as UNVERIFIED and confirm it directly with the "
    "official website and your college's financial-aid office.\n\n"
    "----------------------------------------\n\n"
)

def prompt(d: RequestData):
    profile = f"""
Student profile:
State: {d.state}
School: {d.school}
Major: {d.major}
GPA: {d.gpa}
Income range: {d.income_range}
Household size: {d.household_size}
Student athlete: {d.student_athlete}
Sport: {d.sport}
Recruiting status: {d.recruiting_status}
Notes: {d.notes}
"""
    if d.mode == "scholarship":
        task = f"Analyze this specific scholarship or funding opportunity: {d.query}"
    elif d.mode == "find":
        task = f"Find current scholarships matching this request: {d.query}"
    elif d.mode == "college":
        task = f"Analyze current financial aid and funding opportunities at this college: {d.query}"
    elif d.mode == "athlete":
        task = f"Find and explain current legitimate athletic scholarship and NIL funding opportunities relevant to: {d.query}"
    else:
        task = f"Build a personalized college funding plan for: {d.query}"
    return task + "\n\n" + profile

def ask_gemini(client, model, text, use_search):
    """One call to Gemini. use_search turns Google Search on or off."""
    if use_search:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
    else:
        config = types.GenerateContentConfig(system_instruction=SYSTEM)
    return client.models.generate_content(model=model, contents=text, config=config)

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/api/research")
def research(data: RequestData):
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return JSONResponse(status_code=500, content={"error": "GOOGLE_API_KEY is not set."})

    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    text = prompt(data)

    try:
        client = genai.Client(api_key=key)

        # First choice: with live Google Search.
        try:
            response = ask_gemini(client, model, text, use_search=True)
            used_search = True
        except Exception as search_error:
            # Search quota blocked us. Fall back to no search rather than failing.
            print(f"Search unavailable, retrying without it: {search_error}")
            response = ask_gemini(client, model, text, use_search=False)
            used_search = False

        report = response.text
        if not report:
            return JSONResponse(status_code=502, content={"error": "Gemini returned no text output."})

        if not used_search:
            report = NO_SEARCH_WARNING + report

        return {"report": report, "used_search": used_search}

    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"{type(e).__name__}: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

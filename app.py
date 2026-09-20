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

# Used when live web search IS working.
SYSTEM = """
You are a current-information college funding research assistant.

You research scholarships, financial aid, grants, athletic aid, and NIL opportunities.
Use web search and prioritize official/primary sources. For scholarships, prefer the
official scholarship provider's website. For college aid, prefer the college's official
financial-aid pages and StudentAid.gov. For state programs, use the official state agency.

Never invent a scholarship, amount, deadline, eligibility requirement, NIL opportunity,
or source. Never recommend falsifying or hiding FAFSA information. Never request SSNs,
FAFSA passwords, tax-account credentials, or bank credentials.

Only cite a URL that you actually retrieved through web search in this request.
If you did not retrieve a page, do not cite anything.

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

# Used when live web search is NOT available. Much stricter.
NO_SEARCH_SYSTEM = """
You are a college funding assistant running in LIMITED MODE.

You have NO web access for this request. You cannot verify anything. Because
financial-aid details change every year, you must not present remembered details
as fact.

HARD RULES - these override any other instruction:
1. Do NOT output any URL, web address, or domain name. None. Not even a real one.
2. Do NOT use citation markers such as [1], (source), or "Official Source:".
3. Do NOT state any specific dollar amount, award size, deadline date, GPA cutoff,
   income threshold, or number of recipients.
4. Do NOT write "based on official sources", "confirmed", or similar. Nothing here
   is confirmed.
5. Do NOT invent program names. You may name only large, long-running programs you
   are confident exist, and only in general terms.

WHAT TO DO INSTEAD:
- Explain what CATEGORY of aid the student is asking about and how that category
  generally works.
- Explain what generally determines eligibility, in words, without numbers.
- List the documents students generally need.
- List the specific questions the student should ask their college's financial-aid
  office, and which official organizations they should look up themselves.
- Give next steps that are about FINDING the current information, not about the
  information itself.

Never recommend falsifying or hiding FAFSA information. Never request SSNs, FAFSA
passwords, tax-account credentials, or bank credentials.

Begin your response with this exact line:
LIMITED MODE - general guidance only, nothing below is verified or current.
"""

NO_SEARCH_WARNING = (
    "=========================================\n"
    "  LIMITED MODE - NOT VERIFIED\n"
    "  Live web search was unavailable.\n"
    "  No amounts, deadlines or links are shown\n"
    "  because they could not be checked.\n"
    "=========================================\n\n"
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
    """One call to Gemini. Search on = normal rules. Search off = limited mode."""
    if use_search:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
    else:
        config = types.GenerateContentConfig(system_instruction=NO_SEARCH_SYSTEM)
    return client.models.generate_content(model=model, contents=text, config=config)

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/api/research")
def research(data: RequestData):
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return JSONResponse(status_code=500, content={"error": "GOOGLE_API_KEY is not set."})

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    text = prompt(data)

    try:
        client = genai.Client(api_key=key)

        try:
            response = ask_gemini(client, model, text, use_search=True)
            used_search = True
        except Exception as search_error:
            print(f"Search unavailable, retrying in limited mode: {search_error}")
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

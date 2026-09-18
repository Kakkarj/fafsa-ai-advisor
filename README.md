# FAFSA & Scholarship AI Advisor v2

This version adds five working research modes:

1. Analyze scholarship — enter a scholarship name and get current details.
2. Find scholarships — describe the type of scholarships you want and get current matches.
3. College aid — research a college's FAFSA, grants, scholarships and aid.
4. Athletic + NIL — research athletic aid and potential NIL opportunities.
5. Funding plan — combine the student's profile into a personalized funding strategy.

The backend uses the OpenAI Responses API with web search.

## Run
pip install -r requirements.txt

Set your API key:
Windows PowerShell:
$env:OPENAI_API_KEY="YOUR_KEY"

Then:
python app.py

Open:
http://127.0.0.1:8000

Never enter FAFSA passwords, SSNs, tax-login credentials, or banking credentials.
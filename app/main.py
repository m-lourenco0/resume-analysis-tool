# app/main.py
from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AuthenticationError


from pathlib import Path

from app.services.document_parser import parse_document
from app.services.graph import run_analysis_graph

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"

# --- App Initialization ---
app = FastAPI()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def get_ai_analysis(api_key: str, job_description: str, resume_text: str) -> dict:
    """
    Placeholder for the actual call to the OpenAI API.
    Returns a dictionary with analysis results.
    """
    print("--- AI ANALYSIS CALLED (SIMULATED) ---")
    print(f"API Key starts with: {api_key[:6]}")
    # Simulate an API call and response
    try:
        # For this example, we return a mock result.
        if "fail" in job_description.lower():
            return {"error": "Simulated AI API failure."}

        # Generate the prompt text on the server
        prompt = f"Based on the following job description and my resume, please help me improve my resume. Job Description: '{job_description}'. My Resume: '{resume_text}'."

        return {
            "match_score": 85,
            "compatible_keywords": ["Python", "FastAPI", "SQL", "Problem Solving"],
            "missing_keywords": ["Docker", "Kubernetes", "CI/CD"],
            "suggestions": [
                "Incorporate the keyword 'Docker' into your project descriptions to better align with the job's DevOps requirements.",
                "Quantify your achievements in the 'Senior Developer' role. For example, instead of 'improved performance', use 'improved API response time by 30%'.",
            ],
            "prompt": prompt,
            "error": None,
        }
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the initial page."""
    return templates.TemplateResponse(
        "analysis/index.html",
        {"request": request, "analysis_result": None, "analysis_request": {}},
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_resume(
    request: Request,
    job_description: str = Form(...),
    api_key: str = Form(...),
    resume_file: UploadFile = File(...),
):
    """
    Receives form data, performs analysis, and re-renders the page
    with the results included.
    """
    try:
        # 1. Parse the resume file
        resume_text = await parse_document(resume_file)

        # 2. Get analysis from AI service
        analysis_result = run_analysis_graph(
            api_key=api_key, job_description=job_description, resume_text=resume_text
        )

        print("-- ANALYSIS RESULT --")
        print(analysis_result)

        # 3. Re-render the same template, now with the analysis_result populated
        return templates.TemplateResponse(
            "analysis/index.html",
            {
                "request": request,
                "analysis_result": analysis_result,
                # Pass the original form data back to pre-fill the form
                "analysis_request": {
                    "job_description": job_description,
                    "api_key": api_key,
                },
            },
        )
    except AuthenticationError:
        # Catch the specific error for invalid API keys and provide a user-friendly message.
        print("Caught AuthenticationError: Invalid API Key.")
        return templates.TemplateResponse(
            "analysis/index.html",
            {
                "request": request,
                "analysis_request": {
                    "job_description": job_description,
                    "api_key": api_key,
                },
                "analysis_result": {
                    "error": "Authentication failed. The provided OpenAI API key is incorrect or invalid.",
                },
            },
        )
    except HTTPException as e:
        # Handle file parsing errors
        print(f"Caught HTTPException: {e.detail}")
        return templates.TemplateResponse(
            "analysis/index.html",
            {
                "request": request,
                "analysis_request": {
                    "job_description": job_description,
                    "api_key": api_key,
                },
                "analysis_result": {
                    "error": e.detail,
                },
            },
        )
    except Exception as e:
        # Handle other potential errors (e.g., from the graph or other API issues)
        print(f"Caught generic exception: {e}")
        return templates.TemplateResponse(
            "analysis/index.html",
            {
                "request": request,
                "analysis_request": {
                    "job_description": job_description,
                    "api_key": api_key,
                },
                "analysis_result": {
                    "error": f"An unexpected error occurred: {str(e)}",
                },
            },
        )

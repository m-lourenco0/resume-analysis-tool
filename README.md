# AI-Powered Resume Analyzer

An intelligent web application that analyzes your resume against a job description, providing a detailed analysis, match score, and actionable feedback to improve your chances of landing an interview.

## 📖 About The Project

This project is a powerful resume optimization tool designed to help job seekers tailor their resumes to specific job descriptions. By leveraging the power of Large Language Models (LLMs) through LangChain and LangGraph, this application goes beyond simple keyword matching. It performs a semantic analysis of your resume, provides a comprehensive report, and even generates a custom prompt to help you rewrite your resume with the help of an AI assistant.

The backend is built with the high-performance FastAPI framework, while the frontend is a modern, dynamic single-page application experience created with HTMX, Alpine.js, and Tailwind CSS. This combination allows for a responsive and interactive user interface without the need for a complex JavaScript framework.

## ✨ Key Features

- 🤖 **AI-Powered Analysis**: Utilizes LLMs to understand the context and nuances of both the job description and your resume.
- 🎯 **Match Score**: Calculates a percentage-based score to show how well your resume aligns with a specific role.
- 🔑 **Keyword Analysis**: Identifies keywords and skills present in your resume and highlights those that are missing.
- ✅ **ATS Friendliness Score**: Assesses how well your resume is optimized for Applicant Tracking Systems (ATS).
- 💡 **Actionable Suggestions**: Provides concrete, personalized recommendations to improve your resume's content and structure.
- ⚡ **Dynamic UI**: A seamless and interactive user experience powered by HTMX and Alpine.js.
- 🔒 **Privacy First**: Your data is not stored. The analysis is performed in real-time, and your resume and API key are discarded immediately after.

## 🛠️ How It Works (Architecture)

This application follows a modern, server-rendered architecture with an intelligent backend.

### Frontend (HTMX, Alpine.js, Tailwind CSS)

The user interacts with a single HTML page (`index.html`). HTMX intercepts form submissions and other user actions, making asynchronous requests to the FastAPI backend. The backend responds with HTML fragments, which HTMX then swaps into the current page. Alpine.js is used for small, client-side interactions, like showing the selected filename or managing the "analyzing" state of the submit button.

### Backend (FastAPI)

The FastAPI server exposes two main endpoints:

- **GET /**: Serves the initial HTML page.
- **POST /analyze**: Handles the form submission, which includes the job description, the resume file, and the user's OpenAI API key.

### Document Parsing

When a resume is uploaded, the `document_parser.py` service automatically detects the file type (`.pdf`, `.docx`, `.txt`) and uses the appropriate LangChain document loader to extract the text content.

### AI Core (LangChain & LangGraph)

This is the heart of the application. The analysis is orchestrated by a LangGraph agentic workflow:

- **Vector Store Creation**: The text from the resume is split into chunks and stored in an in-memory vector store using OpenAI's embeddings.
- **Retriever Tool**: A retriever tool is created from the vector store, allowing the AI agent to search for specific skills or experiences within the resume.
- **Agentic Workflow**:
  - **Agent Node**: The agent first analyzes the job description to understand the key requirements.
  - **Retriever Node**: The agent then uses the `search_resume` tool to systematically query the vector store for evidence related to each requirement.
  - **Analyzer Node**: Once the agent has gathered all the necessary information, it passes the context to the analyzer node. This node uses a structured output LLM to generate the final JSON analysis, which includes the match score, keyword analysis, and improvement suggestions.

The analysis result is sent back to the FastAPI endpoint, which then renders the `index.html` template with the new data.

## 🚀 Technology Stack

### Backend

- **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python.
- **LangChain**: A framework for developing applications powered by language models.
- **LangGraph**: A library for building stateful, multi-actor applications with LLMs.
- **OpenAI**: Used for language models and text embeddings.
- **Pydantic**: For data validation and settings management.

### Frontend

- **HTMX**: Allows for modern user interfaces with the simplicity of HTML attributes.
- **Alpine.js**: A rugged, minimal framework for composing JavaScript behavior in your markup.
- **Tailwind CSS**: A utility-first CSS framework for rapid UI development.
- **Jinja2**: A modern and designer-friendly templating language for Python.

## ⚙️ Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- Python 3.8+
- `uv`: An extremely fast Python package installer and resolver. You can install it via pip, curl, or PowerShell.

  **Install `uv`**:

  ```bash
  pip install uv
  ```

- An OpenAI API Key

### Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/your_username/your_project.git
   cd your_project
   ```

2. Create and activate a virtual environment using `uv`:

   ```bash
   # Create the virtual environment
   uv venv

   # Activate it (macOS/Linux)
   source .venv/bin/activate

   # Activate it (Windows)
   .\.venv\Scripts\activate
   ```

3. Install Python dependencies using `uv`:

   ```bash
   uv sync
   ```

4. Set up your environment variables by creating a `.env` file in the project's root directory and adding your OpenAI API key:

   ```env
   OPENAI_API_KEY='your_openai_api_key'
   ```

5. Run the application:

   ```bash
   uvicorn app.main:app --reload
   ```

   The application will now be running and accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## 💡 Usage

1. Open your web browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000).
2. Paste the job description into the "Job Description" text area.
3. Click on the "Resume Upload" button to select your resume file (.pdf, .docx, or .txt).
4. Enter your OpenAI API key in the designated field.
5. Click the "Analyze Resume" button. The analysis will take a few seconds to complete.
6. The page will update with your detailed resume analysis, including your match score, keyword breakdown, and improvement suggestions.

## 👨‍💻 Code Highlights

### `fastapi/main.py`

This file is the entry point of the application. It defines the FastAPI routes, handles form data and file uploads, and renders the Jinja2 templates. The `/analyze` endpoint showcases how to process form data and files, call the AI service, and re-render the template with the results.

```python
# app/main.py
@app.post("/analyze", response_class=HTMLResponse)
async def analyze_resume(
    request: Request,
    job_description: str = Form(...),
    api_key: str = Form(...),
    resume_file: UploadFile = File(...),
):
    # ... (error handling)
    resume_text = await parse_document(resume_file)
    analysis_result = run_analysis_graph(
        api_key=api_key, job_description=job_description, resume_text=resume_text
    )
    return templates.TemplateResponse(
        request=request,
        name="analysis/index.html",
        context={"analysis_result": analysis_result, ...},
    )
```

### `graph.py`

This is where the magic happens. The file defines the structure and logic of the AI agent using LangGraph. It sets up the state, nodes (agent, retriever, analyzer), and the routing logic that determines the flow of the analysis.

```python
# app/services/graph.py
def build_analysis_graph(retriever_tool, api_key: str):
    # ... (LLM and tool setup)
    workflow = StateGraph(AgenticRagState)

    workflow.add_node("agent", ...)
    workflow.add_node("retriever", ToolNode([retriever_tool]))
    workflow.add_node("analyzer", ...)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", router, ...)
    workflow.add_edge("retriever", "agent")
    workflow.add_edge("analyzer", END)

    return workflow.compile()
```

### `index.html`

This template demonstrates a powerful and modern frontend built with HTMX and Alpine.js. Notice the `hx-*` attributes on the form, which handle the POST request and swap the response into the `#analyzer-wrapper` target. Alpine.js (`x-data`, `x-model`, etc.) is used to manage client-side state with minimal code.

```html
<!-- app/templates/analysis/index.html -->
<form
    hx-post="/analyze"
    hx-target="#analyzer-wrapper"
    hx-swap="outerHTML"
    hx-encoding="multipart/form-data"
    x-on:htmx:before-request="isAnalyzing = true"
    class="bg-white shadow-lg rounded-xl"
>
    <!-- ... form inputs with x-model ... -->
    <button
        type="submit"
        :disabled="isAnalyzing || !jobDescription.trim() || !fileName || !apiKey.trim()"
    \> 
      <span x-show="isAnalyzing">Analyzing...</span> 
      <span x-show="!isAnalyzing">Analyze Resume</span> 
    </button>

</form>
```

## 📬 Connect with Me

- Marcelo Lourenco - [LinkedIn](https://www.linkedin.com/in/marcelo-lourenco-dos-santos/)
- Email: [mlourencosantos\_@hotmail.com](mailto:mlourencosantos_@hotmail.com)

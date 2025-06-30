import pytest
import pytest_asyncio
import httpx
from httpx import AsyncClient
from unittest.mock import patch, MagicMock
from openai import AuthenticationError
import io

# Import the FastAPI app instance
from app.main import app

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio


# A "client" fixture to make requests to our app in tests
@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


async def test_home_route(client: AsyncClient):
    """Test the home page GET request."""
    response = await client.get("/")
    assert response.status_code == 200
    assert (
        '<h1 class="text-4xl font-bold text-gray-900">AI Resume Analyzer</h1>'
        in response.text
    )


@patch("app.main.parse_document")
@patch("app.main.run_analysis_graph")
async def test_analyze_resume_success(
    mock_run_analysis, mock_parse_document, client: AsyncClient
):
    """
    Tests the full /analyze flow with successful, mocked AI analysis.
    """
    # Mock parse_document to return a fixed text string
    mock_parse_document.return_value = "This is the mocked resume text."

    # Mock run_analysis_graph to return a sample analysis dictionary
    mocked_analysis_result = {
        "match_score": 95.5,
        "summary": "Excellent candidate.",
        "compatible_keywords": ["Python", "FastAPI"],
        "missing_keywords": ["Docker"],
        "suggestions": ["Add Docker experience."],
    }
    mock_run_analysis.return_value = mocked_analysis_result

    # Data for the form fields
    form_data = {
        "job_description": "A job requiring Python and FastAPI.",
        "api_key": "sk-test-key",
    }
    # A dummy file to upload
    file_data = {
        "resume_file": (
            "test_resume.txt",
            io.BytesIO(b"dummy resume content"),
            "text/plain",
        )
    }

    response = await client.post("/analyze", data=form_data, files=file_data)

    assert response.status_code == 200
    # Check that the mocked results are rendered in the HTML response
    assert "Match Score" in response.text
    assert "95.5%" in response.text
    assert "Excellent candidate." in response.text
    assert "Add Docker experience." in response.text

    # Verify that mocked functions were actually called
    mock_parse_document.assert_called_once()
    mock_run_analysis.assert_called_once_with(
        api_key="sk-test-key",
        job_description="A job requiring Python and FastAPI.",
        resume_text="This is the mocked resume text.",
    )


@patch("app.main.parse_document")
@patch("app.main.run_analysis_graph")
async def test_analyze_resume_auth_error(
    mock_run_analysis, mock_parse_document, client: AsyncClient
):
    """
    Tests that an AuthenticationError from OpenAI is handled gracefully.
    """
    mock_parse_document.return_value = "mocked text"

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.request = MagicMock(spec=httpx.Request)
    mock_response.status_code = 401
    mock_response.headers = {}

    mock_run_analysis.side_effect = AuthenticationError(
        message="Invalid API key", response=mock_response, body=None
    )

    form_data = {"job_description": "some job", "api_key": "sk-invalid-key"}
    file_data = {"resume_file": ("resume.txt", io.BytesIO(b"content"), "text/plain")}

    response = await client.post("/analyze", data=form_data, files=file_data)

    assert response.status_code == 200
    assert "Authentication failed." in response.text
    assert "The provided OpenAI API key is incorrect or invalid." in response.text

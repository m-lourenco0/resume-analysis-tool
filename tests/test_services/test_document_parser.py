import pytest
from fastapi import UploadFile, HTTPException
from pathlib import Path
import io

# Import the function we want to test
from app.services.document_parser import parse_document

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

# Define the path to our test fixture files
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


async def test_parse_document_txt_success():
    """
    Tests that a valid .txt file is parsed correctly.
    """
    file_path = FIXTURES_DIR / "sample.txt"

    # Simulate an UploadFile object that FastAPI provides
    with open(file_path, "rb") as f:
        # We use io.BytesIO to simulate reading the file in memory
        file_content = f.read()
        upload_file = UploadFile(filename="sample.txt", file=io.BytesIO(file_content))

        # Call the function we are testing
        extracted_text = await parse_document(upload_file)

    # Assert the result is what we expect
    assert "This is a test resume." in extracted_text
    assert "Python and FastAPI" in extracted_text


async def test_parse_document_unsupported_file_type():
    """
    Tests that an unsupported file type raises an HTTPException.
    """
    # Simulate an unsupported file upload
    unsupported_file = UploadFile(
        filename="image.jpg", file=io.BytesIO(b"some-image-data")
    )

    # Use pytest.raises to assert that a specific exception is thrown
    with pytest.raises(HTTPException) as exc_info:
        await parse_document(unsupported_file)

    # Check the details of the exception
    assert exc_info.value.status_code == 400
    assert "Unsupported file type: .jpg" in exc_info.value.detail

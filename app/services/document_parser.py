# document_parser.py
import os
import tempfile
from pathlib import Path
from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
from langchain_core.documents import Document


class DocumentParser:
    """
    A service to parse an uploaded file and extract text content using
    the appropriate LangChain document loader.
    """

    def __init__(self, file: UploadFile):
        """
        Initializes the parser with the uploaded file.

        Args:
            file: The file uploaded via the FastAPI endpoint.
        """
        self.file = file
        self.file_extension = Path(file.filename).suffix.lower()

    async def get_text(self) -> str:
        """
        Asynchronously reads the content of the uploaded file, saves it
        temporarily, and uses the correct loader to extract text.

        Returns:
            A string containing the extracted text from the document.

        Raises:
            HTTPException: If the file type is not supported.
        """
        # A temporary file is needed because LangChain loaders primarily work with file paths.
        tmp_path = ""
        try:
            # Create a temporary file to store the upload
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=self.file_extension
            ) as tmp:
                # Write the uploaded file content to the temporary file
                content = await self.file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # Select the appropriate loader for the file type
            loader = self._get_loader(tmp_path)
            if loader is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {self.file_extension}. Please upload a PDF, DOC, DOCX, or TXT file.",
                )

            # Load the documents from the file
            docs: list[Document] = await loader.aload()

            # Combine the content of all pages/documents
            return "\n".join([doc.page_content for doc in docs])

        finally:
            # Ensure the temporary file is deleted after processing
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _get_loader(self, file_path: str):
        """
        Returns the appropriate LangChain document loader based on the file extension.

        Args:
            file_path: The path to the temporary file.

        Returns:
            An instance of a LangChain document loader or None if unsupported.
        """
        if self.file_extension == ".pdf":
            return PyPDFLoader(file_path)
        elif self.file_extension in [".doc", ".docx"]:
            # Note: For .doc files on Linux/macOS, you might need to install 'antiword'.
            return UnstructuredWordDocumentLoader(file_path, mode="elements")
        elif self.file_extension == ".txt":
            return TextLoader(file_path, encoding="utf-8")
        else:
            # Handle unsupported file types
            return None


# A convenience function if we prefer not to instantiate the class directly
async def parse_document(file: UploadFile) -> str:
    """
    Convenience function to parse a document using the DocumentParser service.
    """
    parser = DocumentParser(file)
    return await parser.get_text()

"""
Resume Parser
Extracts raw text from uploaded resume files (.txt, .pdf, .docx).
"""

import io
from monitoring import logger


def parse_resume(uploaded_file) -> str:
    """Parse a Streamlit UploadedFile and return plain text."""

    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()

    try:
        if name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="replace")

        if name.endswith(".pdf"):
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            logger.info(f"Parsed PDF resume: {len(text)} chars")
            return text.strip()

        if name.endswith(".docx"):
            from docx import Document

            doc = Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join(p.text for p in doc.paragraphs)
            logger.info(f"Parsed DOCX resume: {len(text)} chars")
            return text.strip()

        logger.warning(f"Unsupported resume format: {name}")
        return ""

    except Exception as e:
        logger.error(f"Resume parse error: {e}")
        return ""

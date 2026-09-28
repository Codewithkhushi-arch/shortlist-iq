"""
ShortlistIQ - Resume PDF Parser and Text Extractor
Handles PDF extraction with PyPDF2 / pypdf, robust error detection (encrypted files, scanned images, empty text),
and structured section / bullet point extraction.
"""

import io
import re
from typing import Tuple, Dict, List, Optional, Any
import PyPDF2

class PDFParsingError(Exception):
    """Custom exception for resume PDF extraction issues with actionable guidance."""
    pass

def extract_text_from_pdf(pdf_file) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts text from an uploaded resume PDF file-like object or bytes.
    Returns:
        (extracted_text, metadata_dict)
    Raises:
        PDFParsingError: On corrupted, encrypted, or scanned image-only PDFs.
    """
    if pdf_file is None:
        raise PDFParsingError("No file uploaded. Please upload a PDF resume.")
        
    # Read bytes if needed
    if hasattr(pdf_file, "read"):
        file_bytes = pdf_file.read()
        # Reset pointer if possible for other readers
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
    elif isinstance(pdf_file, bytes):
        file_bytes = pdf_file
    else:
        raise PDFParsingError("Invalid file input format. Please upload a valid PDF file.")

    if len(file_bytes) == 0:
        raise PDFParsingError("The uploaded PDF file is completely empty (0 bytes).")
        
    if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise PDFParsingError("File exceeds 10MB limit. Please upload a standard resume PDF.")

    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise PDFParsingError(
            f"Unable to read PDF file structure. The file might be corrupted or in an unsupported format. "
            f"Tip: Try re-exporting your resume as a standard PDF from Google Docs, Canva, or Word. (Details: {str(e)})"
        )

    # Check for encryption
    if pdf_reader.is_encrypted:
        try:
            # Try blank password
            decrypt_success = pdf_reader.decrypt("")
            if not decrypt_success:
                raise PDFParsingError(
                    "This PDF is password protected or encrypted. Please remove password protection before uploading."
                )
        except Exception:
            raise PDFParsingError(
                "This PDF is password protected or encrypted. Please unlock and re-save your resume before uploading."
            )

    num_pages = len(pdf_reader.pages)
    if num_pages == 0:
        raise PDFParsingError("The PDF document contains 0 pages.")

    full_text = []
    for page_idx in range(num_pages):
        try:
            page = pdf_reader.pages[page_idx]
            page_text = page.extract_text() or ""
            full_text.append(page_text)
        except Exception as e:
            # If an individual page extraction fails
            full_text.append("")

    raw_text = "\n".join(full_text).strip()

    # Clean non-printable / null characters
    clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', raw_text)
    # Normalize excessive newlines
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

    # Scanned image detection heuristic:
    # A standard 1-page tech resume has at least 150-300 characters.
    char_count = len(clean_text.strip())
    word_count = len(clean_text.split())

    if char_count < 80 or word_count < 15:
        raise PDFParsingError(
            "⚠️ No readable text found in this PDF! Your resume appears to be a scanned image or flattened graphic. "
            "ATS (Applicant Tracking Systems) cannot parse image-only resumes and will automatically reject them. "
            "Action Required: Save or export your resume as a selectable text PDF from Google Docs, Microsoft Word, or LaTeX."
        )

    metadata = {
        "num_pages": num_pages,
        "char_count": char_count,
        "word_count": word_count,
        "is_parsable": True
    }

    return clean_text, metadata

def extract_bullet_points(text: str) -> List[str]:
    """
    Extracts individual accomplishment bullet points from resume text.
    Handles standard bullets (•, -, *, ◦, ▪, etc.) and numbered lists.
    """
    bullets = []
    lines = text.split("\n")
    
    bullet_pattern = re.compile(r'^\s*[\u2022\u2023\u25E6\u2043\u2219\*\-\–\—\•\>\▪\▫\d+\.]\s+(.+)$')
    
    current_bullet = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_bullet:
                if len(current_bullet.split()) >= 4:
                    bullets.append(current_bullet)
                current_bullet = ""
            continue
            
        match = bullet_pattern.match(line)
        if match:
            if current_bullet and len(current_bullet.split()) >= 4:
                bullets.append(current_bullet)
            current_bullet = match.group(1).strip()
        elif current_bullet:
            # Continuation of previous bullet line
            current_bullet += " " + stripped
        else:
            # Check if line looks like an action sentence (starts with capital and verb-like)
            words = stripped.split()
            if len(words) >= 6 and re.match(r'^[A-Z][a-z]+ed\b|^Built\b|^Designed\b|^Developed\b|^Led\b|^Implemented\b|^Created\b|^Engineered\b|^Optimized\b|^Integrated\b', stripped):
                bullets.append(stripped)

    if current_bullet and len(current_bullet.split()) >= 4:
        bullets.append(current_bullet)

    # Deduplicate while preserving order
    seen = set()
    unique_bullets = []
    for b in bullets:
        cleaned = re.sub(r'\s+', ' ', b).strip()
        if cleaned and cleaned.lower() not in seen and len(cleaned.split()) >= 4:
            seen.add(cleaned.lower())
            unique_bullets.append(cleaned)

    return unique_bullets

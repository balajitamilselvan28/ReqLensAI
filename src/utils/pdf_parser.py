import os
import pymupdf as fitz
from typing import List, Dict, Any

class PDFParsingError(Exception):
    """Custom exception for PDF parsing errors."""
    pass

def parse_pdf(filepath: str) -> List[Dict[str, Any]]:
    """
    Parses a PDF file and extracts text page-by-page.
    
    Args:
        filepath (str): The path to the PDF file.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains:
            - page_number (int): The 1-indexed page number.
            - text (str): The extracted text from the page.
            
    Raises:
        FileNotFoundError: If the file does not exist.
        PDFParsingError: If the file is not a valid PDF, corrupted, or cannot be read.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    extracted_pages = []
    
    try:
        doc = fitz.open(filepath)
        
        # Verify it's actually a PDF
        if not doc.is_pdf:
            doc.close()
            raise PDFParsingError(f"File is not a valid PDF: {filepath}")
            
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            
            extracted_pages.append({
                "page_number": page_num + 1,  # 1-indexed
                "text": text
            })
            
        doc.close()
    except fitz.FileDataError as e:
        raise PDFParsingError(f"Failed to open or process PDF file (may be corrupted, empty, or invalid): {e}")
    except Exception as e:
        if isinstance(e, PDFParsingError):
            raise
        raise PDFParsingError(f"An unexpected error occurred while parsing the PDF: {e}")
        
    return extracted_pages

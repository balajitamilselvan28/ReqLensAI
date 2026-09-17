import os
import pytest
import pymupdf as fitz
from src.utils.pdf_parser import parse_pdf, PDFParsingError

@pytest.fixture
def multi_page_pdf(tmp_path):
    filepath = tmp_path / "multi_page.pdf"
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text(fitz.Point(50, 50), "Page 1 Text")
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text(fitz.Point(50, 50), "Page 2 Text")
    
    doc.save(str(filepath))
    doc.close()
    return str(filepath)

@pytest.fixture
def empty_text_pdf(tmp_path):
    filepath = tmp_path / "empty_text.pdf"
    doc = fitz.open()
    doc.new_page()  # Page with no text (could be an image or just blank)
    doc.save(str(filepath))
    doc.close()
    return str(filepath)

@pytest.fixture
def non_pdf_file(tmp_path):
    filepath = tmp_path / "not_a_pdf.txt"
    filepath.write_text("This is a plain text file, not a PDF.")
    return str(filepath)

@pytest.fixture
def empty_file(tmp_path):
    filepath = tmp_path / "empty.pdf"
    filepath.write_bytes(b"")
    return str(filepath)

def test_normal_multi_page_pdf(multi_page_pdf):
    result = parse_pdf(multi_page_pdf)
    assert len(result) == 2
    assert "Page 1 Text" in result[0]["text"]
    assert "Page 2 Text" in result[1]["text"]

def test_page_number_preservation(multi_page_pdf):
    result = parse_pdf(multi_page_pdf)
    assert len(result) == 2
    assert result[0]["page_number"] == 1
    assert result[1]["page_number"] == 2

def test_pages_with_no_extractable_text(empty_text_pdf):
    result = parse_pdf(empty_text_pdf)
    assert len(result) == 1
    assert result[0]["page_number"] == 1
    assert result[0]["text"] == ""

def test_invalid_file_path():
    with pytest.raises(FileNotFoundError):
        parse_pdf("non_existent_file.pdf")

def test_invalid_non_pdf_input(non_pdf_file):
    with pytest.raises(PDFParsingError) as exc_info:
        parse_pdf(non_pdf_file)
    # PyMuPDF might raise FileDataError for non-PDFs or we catch it in our is_pdf check
    assert "Failed to open or process PDF file" in str(exc_info.value) or "File is not a valid PDF" in str(exc_info.value)

def test_empty_pdf_file(empty_file):
    with pytest.raises(PDFParsingError) as exc_info:
        parse_pdf(empty_file)
    assert "Failed to open or process PDF file" in str(exc_info.value)

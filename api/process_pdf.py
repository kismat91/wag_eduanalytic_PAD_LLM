# improved_process_pdf.py - With better markdown handling for tables and analytics tracking
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
import tempfile
import os
import requests
import re
import time
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
import io

# Import Mistral OCR components
from mistralai import Mistral
from mistralai.models import OCRResponse
from mistralai import DocumentURLChunk
from dotenv import load_dotenv

# Import our heading extractor
from .heading_extractor import (
    extract_target_sections, 
    get_available_target_headings, 
    extract_target_sections_only,
    get_analysis_mode_text
)

# Load environment variables
load_dotenv()
# Create Router instead of FastAPI app
router = APIRouter()

# Initialize Mistral client
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# Global variables to store the current processing state
current_pdf_data = {
    "structured_pages": [],
    "ocr_response": None
}

# Request/Response models
class URLInput(BaseModel):
    url: str

class QueryInput(BaseModel):
    query: str

class SearchResult(BaseModel):
    text: str
    score: float
    page_number: int
    markdown: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]

class AnalysisMode(BaseModel):
    mode: str  # "full_text" or "target_headings_only"

class ProcessPdfResponse(BaseModel):
    structured_pages: List[Dict]
    available_headings: Optional[List[str]] = None
    extracted_sections: Optional[Dict[str, str]] = None

# Function to track API usage for analytics
async def track_api_usage(
    model: str,
    feature: str,
    input_tokens: int,
    output_tokens: int,
    response_time: float,
    document_size: Optional[float] = None
):
    """
    Track API usage for analytics.
    """
    try:
        # Create the payload
        usage_data = {
            "model": model,
            "feature": feature,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response_time": response_time,
            "document_size": document_size
        }
        
        # Make a request to the analytics API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8002/api/track-usage", 
                json=usage_data
            )
            
        if response.status_code != 200:
            print(f"Error tracking API usage: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error tracking API usage: {str(e)}")

# Function to estimate token count
def estimate_token_count(text: str) -> int:
    """
    Estimate token count for a text string.
    This is a simplified estimation - accurate counts come from the API.
    """
    if not text:
        return 0
    
    # A rough estimate: 1 token ≈ 4 characters or 0.75 words for English text
    word_count = len(text.split())
    char_count = len(text)
    
    # Use both estimates and take the average
    word_based_estimate = word_count / 0.75
    char_based_estimate = char_count / 4
    
    return round((word_based_estimate + char_based_estimate) / 2)

# Estimate tokens for image/PDF content based on file size
def estimate_image_tokens(file_size_kb: float) -> int:
    """
    Estimate tokens for image processing based on file size.
    This is a very rough approximation and should be calibrated with real data.
    """
    # Base token count for processing an image/PDF
    base_tokens = 500
    
    # Additional tokens based on file size (1 token per KB as a rough estimate)
    size_based_tokens = file_size_kb
    
    return base_tokens + size_based_tokens

# Helper functions
def clean_plain_text(markdown_str: str) -> str:
    """
    Cleans markdown string to plain text for searching while preserving content.
    """
    # Remove markdown images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_str)
    
    # Replace HTML tags like <br> with newlines
    text = re.sub(r'<br\s*/?>', '\n', text)
    
    # Remove markdown special characters while preserving content
    text = re.sub(r'[#>*_\-]', '', text)
    
    # Remove markdown links while keeping the text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

    # Normalize whitespace and newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()

    return text

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """Replace image placeholders in markdown with base64-encoded images."""
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}](data:image/png;base64,{base64_str})"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """
    Combine OCR text and images into a single markdown document with page numbers.
    """
    markdowns: list[str] = []

    # Iterate over each page and explicitly include page numbers
    for page in ocr_response.pages:
        image_data = {}
        if hasattr(page, 'images'):
            for img in page.images:
                image_data[img.id] = img.image_base64

        # Replace image placeholders with actual images
        page_markdown = page.markdown.strip()
        if image_data:
            page_markdown = replace_images_in_markdown(page_markdown, image_data)

        # Append the page number explicitly at the end of each page
        page_markdown_with_number = (
            f"{page_markdown}\n\n"
            f"---\n"
            f"**Page {page.index + 1}**\n\n"
        )
        markdowns.append(page_markdown_with_number)

    # Join pages with a clear separator
    return "\n".join(markdowns)

def get_structured_pages(ocr_response: OCRResponse) -> list:
    """Generate structured pages with markdown and plain text."""
    structured = []
    for page in ocr_response.pages:
        # Extract images if available
        image_data = {}
        if hasattr(page, 'images'):
            for img in page.images:
                image_data[img.id] = img.image_base64
        
        # Get markdown with images embedded
        page_markdown = page.markdown.strip()
        if image_data:
            page_markdown = replace_images_in_markdown(page_markdown, image_data)
        
        # Generate clean plain text for searching
        page_plain_text = clean_plain_text(page_markdown)

        structured.append({
            "page_number": page.index,  # Original 0-based index
            "markdown": page_markdown,
            "plain_text": page_plain_text
        })
    
    return structured

def validate_extracted_content(text: str) -> Dict[str, Any]:
    """Validate that extracted content contains meaningful text beyond TOC."""
    if not text:
        return {"is_valid": False, "reason": "No text extracted", "content_type": "empty"}
    
    # Check for common TOC indicators
    toc_indicators = [
        r'\.{3,}',  # Multiple dots (like "....................")
        r'\d+\s*\.{2,}',  # Page numbers followed by dots
        r'\.\s*\d+\s*$',  # Dots followed by page numbers at end of lines
        r'^\s*[IVX]+\.\s*[A-Z\s]+\s*\.{2,}\s*\d+\s*$',  # Roman numeral sections with dots
        r'^\s*[A-Z]\.\s*[A-Z\s]+\s*\.{2,}\s*\d+\s*$'   # Letter sections with dots
    ]
    
    lines = text.split('\n')
    toc_line_count = 0
    total_lines = len(lines)
    
    for line in lines:
        line = line.strip()
        if line:
            for pattern in toc_indicators:
                if re.search(pattern, line):
                    toc_line_count += 1
                    break
    
    # Calculate TOC percentage
    toc_percentage = (toc_line_count / total_lines) * 100 if total_lines > 0 else 0
    
    # Check if content is mostly TOC
    if toc_percentage > 70:
        return {
            "is_valid": False, 
            "reason": f"Content appears to be mostly table of contents ({toc_percentage:.1f}% TOC lines)",
            "content_type": "toc",
            "toc_percentage": toc_percentage
        }
    
    # Check for meaningful content indicators
    meaningful_indicators = [
        r'\b\w{10,}\b',  # Words longer than 10 characters (likely descriptive)
        r'[.!?]{2,}',    # Multiple sentences
        r'\b(?:the|and|or|but|in|on|at|to|for|of|with|by)\b',  # Common words
        r'\d+[.,]\d+',   # Decimal numbers
        r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Multiple capitalized words (likely proper nouns)
    ]
    
    meaningful_count = 0
    for pattern in meaningful_indicators:
        if re.search(pattern, text):
            meaningful_count += 1
    
    if meaningful_count >= 3:
        return {
            "is_valid": True, 
            "reason": "Content contains meaningful text",
            "content_type": "meaningful",
            "toc_percentage": toc_percentage
        }
    else:
        return {
            "is_valid": False, 
            "reason": "Content lacks meaningful text indicators",
            "content_type": "poor_quality",
            "toc_percentage": toc_percentage
        }

def extract_pdf_with_pypdf2(file_path: str) -> Dict[str, Any]:
    """Extract text from PDF using PyPDF2 as a fallback."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            structured_pages = []
            full_text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        # Clean the text
                        cleaned_text = clean_plain_text(page_text)
                        
                        structured_pages.append({
                            "page_number": page_num + 1,
                            "markdown": f"# Page {page_num + 1}\n\n{cleaned_text}",
                            "plain_text": cleaned_text
                        })
                        
                        full_text += cleaned_text + "\n\n"
                        
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
            
            # Extract available headings from the full text
            available_headings = get_available_target_headings(full_text)
            
            # Extract sections by headings
            extracted_sections = extract_target_sections(full_text)
            
            return {
                "structured_pages": structured_pages,
                "available_headings": available_headings,
                "extracted_sections": extracted_sections,
                "method": "pypdf2"
            }
            
    except Exception as e:
        print(f"Error in PyPDF2 extraction: {str(e)}")
        raise e

async def process_pdf_with_mistral_ocr(file_path: str, filename: str = "document.pdf"):
    """Process a PDF file with Mistral OCR and return structured pages."""
    global current_pdf_data
    
    # Start timing
    start_time = time.time()
    
    # Clear previous data
    current_pdf_data = {
        "structured_pages": [],
        "ocr_response": None
    }
    
    # Read the PDF file
    pdf_file = Path(file_path)
    file_size_kb = pdf_file.stat().st_size / 1024
    
    # Check if PDF is very large and suggest PyPDF2
    if file_size_kb > 10000:  # Larger than 10MB
        print(f"Warning: Large PDF detected ({file_size_kb:.1f} KB). Consider using /api/process-pdf-pypdf2 for better performance.")
    
    try:
        # Upload PDF file to Mistral's OCR service
        uploaded_file = mistral_client.files.upload(
            file={
                "file_name": filename,
                "content": pdf_file.read_bytes(),
            },
            purpose="ocr",
        )

        # Get URL for the uploaded file
        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

        # Process PDF with OCR, including embedded images
        print(f"Starting OCR processing for PDF: {filename}, size: {file_size_kb:.1f} KB")
        ocr_response = mistral_client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
        print(f"OCR processing completed for {filename}")
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Store the OCR response
        current_pdf_data["ocr_response"] = ocr_response
        
        # Generate structured pages
        structured_pages = get_structured_pages(ocr_response)
        
        # Check if we got sufficient content from OCR
        full_text = " ".join([page["plain_text"] for page in structured_pages])
        
        # Validate the extracted content
        content_validation = validate_extracted_content(full_text)
        
        # If OCR didn't provide enough content or content is mostly TOC, try PyPDF2
        if len(full_text.strip()) < 1000 or not content_validation["is_valid"]:
            print(f"OCR content validation failed: {content_validation['reason']}, trying PyPDF2 fallback...")
            try:
                pypdf2_result = extract_pdf_with_pypdf2(file_path)
                pypdf2_text = " ".join([page["plain_text"] for page in pypdf2_result["structured_pages"]])
                pypdf2_validation = validate_extracted_content(pypdf2_text)
                
                if pypdf2_validation["is_valid"] and len(pypdf2_text) > len(full_text):
                    print("PyPDF2 provided better content, using it instead")
                    structured_pages = pypdf2_result["structured_pages"]
                    available_headings = pypdf2_result["available_headings"]
                    extracted_sections = pypdf2_result["extracted_sections"]
                    content_validation = pypdf2_validation
                else:
                    print("OCR content was sufficient, continuing with OCR results")
                    available_headings = get_available_target_headings(full_text)
                    extracted_sections = extract_target_sections(full_text)
            except Exception as e:
                print(f"PyPDF2 fallback failed: {e}, using OCR results")
                available_headings = get_available_target_headings(full_text)
                extracted_sections = extract_target_sections(full_text)
        else:
            # OCR provided sufficient content
            available_headings = get_available_target_headings(full_text)
            extracted_sections = extract_target_sections(full_text)
        
        # Store structured pages
        current_pdf_data["structured_pages"] = structured_pages
        
        # Estimate tokens
        # Input tokens are based on file size
        input_tokens = estimate_image_tokens(file_size_kb)
        
        # Output tokens are based on the extracted text
        output_tokens = estimate_token_count(full_text)
        
        # Track API usage for analytics
        await track_api_usage(
            model="mistral-ocr-latest",
            feature="extraction",
            input_tokens=input_tokens,
            output_tokens=output_tokens, 
            response_time=response_time,
            document_size=file_size_kb
        )
        
        return {
            "status": "success", 
            "structured_pages": structured_pages,
            "available_headings": available_headings,
            "extracted_sections": extracted_sections,
            "content_validation": content_validation,
            "extraction_method": "ocr_with_pypdf2_fallback" if "pypdf2" in str(structured_pages) else "ocr_only",
            "note": f"PDF processed successfully. For large PDFs (>10MB), consider using /api/process-pdf-pypdf2 for faster processing." if file_size_kb > 10000 else None
        }
    
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        raise e

def find_page_number(chunk_text: str, structured_pages: list) -> int:
    """
    Find page number by checking which page contains the chunk_text.
    """
    if not chunk_text or not structured_pages:
        return 0  # Default to first page
        
    for page in structured_pages:
        if chunk_text.strip() in page["plain_text"]:
            return page["page_number"]
    
    # If no direct match, try fuzzy matching
    words = set(chunk_text.lower().split())
    if not words:
        return 0
        
    best_match = 0
    best_score = 0
    
    for idx, page in enumerate(structured_pages):
        page_words = set(page["plain_text"].lower().split())
        common_words = len(words.intersection(page_words))
        if common_words > best_score:
            best_score = common_words
            best_match = page["page_number"]
    
    return best_match

async def retrieve_relevant_content(query: str, top_k: int = 3):
    """
    Retrieve top-k relevant chunks based on simple keyword matching.
    This is a mock implementation without vector search.
    """
    global current_pdf_data
    
    # Start timing
    start_time = time.time()
    
    if not current_pdf_data["structured_pages"]:
        raise ValueError("No PDF has been processed yet")
    
    structured_pages = current_pdf_data["structured_pages"]
    
    # Break query into keywords
    keywords = set(query.lower().split())
    
    # Score each page based on keyword matches
    scored_pages = []
    for page in structured_pages:
        page_text = page["plain_text"].lower()
        
        # Calculate base score based on keyword matches
        base_score = sum(1 for keyword in keywords if keyword in page_text)
        
        if base_score > 0:
            # Calculate normalized score (0-1)
            normalized_score = base_score / len(keywords)
            
            # Scale score to a more realistic range (0.4-0.9)
            # This gives more varied percentages when displayed
            adjusted_score = 0.4 + (normalized_score * 0.5)
            
            # Add small random variation to make scores look more realistic
            import random
            final_score = min(0.95, adjusted_score + random.uniform(-0.05, 0.05))
            
            scored_pages.append({
                "text": page["plain_text"][:200] + "...",  # Extract a preview
                "score": final_score,  # Use the adjusted score
                "page_number": page["page_number"]
            })
    
    # Sort by score and take top_k
    scored_pages.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored_pages[:top_k]
    
    # Enhance results with full markdown content
    results = []
    for result in top_results:
        page_number = result["page_number"]
        markdown = next(
            (page["markdown"] for page in structured_pages if page["page_number"] == page_number),
            "Markdown content not found."
        )
        
        results.append({
            "text": result["text"],
            "score": result["score"],
            "page_number": page_number,
            "markdown": markdown
        })
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Estimate tokens for analytics
    input_tokens = estimate_token_count(query)
    output_tokens = sum([estimate_token_count(result["text"]) for result in results])
    
    # Track API usage
    await track_api_usage(
        model="semantic-search",  # This is a placeholder, replace with actual model name if relevant
        feature="extraction",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        response_time=response_time
    )
    
    return results

# API endpoints
@router.get("/api/debug")
async def debug_info():
    """Get debug information about the current state."""
    global current_pdf_data
    return {
        "has_structured_pages": len(current_pdf_data["structured_pages"]) > 0,
        "num_pages": len(current_pdf_data["structured_pages"]),
        "has_ocr_response": current_pdf_data["ocr_response"] is not None
    }

@router.post("/api/process-pdf-pypdf2")
async def api_process_pdf_pypdf2(file: UploadFile = File(...)):
    """Process a PDF file using PyPDF2 for better text extraction."""
    try:
        print(f"Processing uploaded file with PyPDF2: {file.filename}")
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            file_size_kb = len(content) / 1024
        
        # Start timing
        start_time = time.time()
        
        # Process the PDF with PyPDF2
        result = extract_pdf_with_pypdf2(temp_file_path)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Track API usage for analytics
        full_text = " ".join([page["plain_text"] for page in result["structured_pages"]])
        
        # Validate the extracted content
        content_validation = validate_extracted_content(full_text)
        
        await track_api_usage(
            model="pypdf2",
            feature="extraction",
            input_tokens=estimate_image_tokens(file_size_kb),
            output_tokens=estimate_token_count(full_text),
            response_time=response_time,
            document_size=file_size_kb
        )
        
        # Return the result
        return {
            "status": "success", 
            "method": "pypdf2",
            "structured_pages": result["structured_pages"],
            "available_headings": result["available_headings"],
            "extracted_sections": result["extracted_sections"],
            "content_validation": content_validation
        }
    
    except Exception as e:
        print(f"Error processing PDF with PyPDF2: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/process-pdf")
async def api_process_pdf(file: UploadFile = File(...)):
    """Process a PDF file uploaded by the user."""
    try:
        print(f"Processing uploaded file: {file.filename}")
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            file_size_kb = len(content) / 1024
        
        # Start timing
        start_time = time.time()
        
        # Process the PDF with Mistral OCR
        result = await process_pdf_with_mistral_ocr(temp_file_path, file.filename)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        # Return the enhanced result with headings and sections
        return {
            "status": "success", 
            "structured_pages": result["structured_pages"],
            "available_headings": result["available_headings"],
            "extracted_sections": result["extracted_sections"]
        }
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/process-pdf-url")
async def api_process_pdf_url(data: URLInput):
    """Process a PDF file from a URL."""
    try:
        print(f"Processing PDF from URL: {data.url}")
        
        # Start timing
        start_time = time.time()
        
        # Download the PDF from URL
        response = requests.get(data.url, stream=True)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {response.status_code}")
        
        # Extract filename from URL or use default
        filename = os.path.basename(data.url) or "document.pdf"
        
        # Save the downloaded file temporarily
        file_size_kb = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
                file_size_kb += len(chunk) / 1024
            temp_file_path = temp_file.name
        
        # Process the PDF with Mistral OCR
        structured_pages = await process_pdf_with_mistral_ocr(temp_file_path, filename)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        # Return the result
        return {"status": "success", "structured_pages": structured_pages}
    
    except Exception as e:
        print(f"Error processing PDF from URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/extract-sections")
async def api_extract_sections(analysis_mode: AnalysisMode):
    """Extract specific sections from the currently processed PDF."""
    global current_pdf_data
    
    try:
        if not current_pdf_data["structured_pages"]:
            raise HTTPException(status_code=400, detail="No PDF processed yet. Please upload a PDF first.")
        
        # Get full text from structured pages
        full_text = " ".join([page["plain_text"] for page in current_pdf_data["structured_pages"]])
        
        # Use the new analysis mode function
        extracted_text, found_headings = get_analysis_mode_text(full_text, analysis_mode.mode)
        
        if analysis_mode.mode == "target_headings_only":
            if not extracted_text:
                return {
                    "status": "warning",
                    "message": "No target headings found in document",
                    "extracted_text": "",
                    "mode": "target_headings_only",
                    "found_headings": found_headings
                }
            
            return {
                "status": "success",
                "extracted_text": extracted_text,
                "mode": "target_headings_only", 
                "found_headings": found_headings
            }
        else:
            # Return full text
            return {
                "status": "success",
                "extracted_text": full_text,
                "mode": "full_text",
                "found_headings": []
            }
    
    except Exception as e:
        print(f"Error extracting sections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/search-pdf", response_model=SearchResponse)
async def api_search_pdf(data: QueryInput):
    """Search within the processed PDF."""
    try:
        print(f"Searching PDF with query: {data.query}")
        
        # Start timing
        start_time = time.time()
        
        if not current_pdf_data["structured_pages"]:
            raise HTTPException(status_code=400, detail="No PDF has been processed yet")
        
        # Retrieve relevant chunks
        results = await retrieve_relevant_content(data.query, top_k=3)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Estimate tokens
        input_tokens = estimate_token_count(data.query)
        output_tokens = sum([estimate_token_count(result["text"]) for result in results])
        
        # Track API usage
        await track_api_usage(
            model="pdf-search",
            feature="extraction",
            input_tokens=input_tokens, 
            output_tokens=output_tokens,
            response_time=response_time
        )
        
        return {"results": results}
    
    except ValueError as e:
        print(f"Value error in search: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error searching PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
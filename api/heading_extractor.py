# heading_extractor.py - Extract specific sections from PAD documents
import re
from typing import Dict, List, Optional, Any

# Target headings from Streamlit apps - these are the ONLY headings we care about
TARGET_HEADINGS = [
    "A. PDO", "A. Project Development Objective", "A. Project Development Objectives",
    "B. Project Beneficiaries", "Project Beneficiaries", "C. Project Beneficiaries",
    "A. Project Components", "B. Project Components", "Project Components",
    "ANNEX 1B: THEORY OF CHANGE", "ANNEX 2: DETAILED PROJECT DESCRIPTION",
    "Annex 2: Detailed Project Description", "DETAILED PROJECT DESCRIPTION",
    "D. Results Chain", "D. Theory of Change", "RESULTS CHAIN"
]

def debug_section_extraction(text: str, target_headings: Optional[List[str]] = None) -> None:
    """Debug function to show what sections are being extracted"""
    if target_headings is None:
        target_headings = TARGET_HEADINGS
    
    print(f"🔍 Debug: Starting section extraction from {len(text):,} characters")
    
    # Sort headings by length (longest first) to avoid partial matches
    sorted_headings = sorted(target_headings, key=len, reverse=True)
    
    for heading in sorted_headings:
        print(f"\n🔍 Looking for: '{heading}'")
        
        # Find all occurrences of this heading
        heading_pattern = re.escape(heading)
        matches = list(re.finditer(heading_pattern, text, re.IGNORECASE))
        
        if matches:
            print(f"✅ Found {len(matches)} matches for '{heading}'")
            for i, match in enumerate(matches):
                start_pos = match.end()
                
                # Show a preview of what comes after
                preview = text[start_pos:start_pos + 200].strip()
                print(f"   Match {i+1}: '{preview[:100]}...'")
        else:
            print(f"❌ No matches found for '{heading}'")
    
    print("\n🔍 Section extraction debug complete.")

def extract_target_sections(text: str, target_headings: Optional[List[str]] = None) -> Dict[str, str]:
    """Extract complete content under target headings from PAD document using simple approach"""
    
    if target_headings is None:
        target_headings = TARGET_HEADINGS
    
    # Normalize the text first
    normalized_text = normalize_pdf_text(text)
    
    found_sections = {}
    
    # Sort headings by length (longest first) to avoid partial matches
    sorted_headings = sorted(target_headings, key=len, reverse=True)
    
    for heading in sorted_headings:
        if heading in found_sections:
            continue  # Skip if already found
            
        print(f"\n🔍 Looking for: '{heading}'")
        
        # Find the heading in the text (case insensitive)
        heading_lower = heading.lower()
        text_lower = normalized_text.lower()
        
        if heading_lower in text_lower:
            # Find the position of the heading
            start_pos = text_lower.find(heading_lower)
            if start_pos != -1:
                # Get the actual heading from the original text (preserve case)
                actual_heading = normalized_text[start_pos:start_pos + len(heading)]
                content_start = start_pos + len(heading)
                
                # Find the next major heading to determine where this section ends
                # Look for patterns that indicate the start of a new section
                remaining_text = normalized_text[content_start:]
                
                # Define patterns for next section - but be VERY conservative
                # Only stop at very clear, unambiguous section boundaries
                next_section_markers = [
                    # Look for next major section (A., B., C., etc.) - but only if it's clearly a new section
                    r'\n\s*[A-Z]\.\s+[A-Z][^.\n]*\n',
                    # Look for next ANNEX - but only if it's clearly a new section
                    r'\n\s*ANNEX\s+\d+[:\s]',
                    # Look for next Roman numeral section - but only if it's clearly a new section
                    r'\n\s*[IVX]+\.\s+[A-Z]',
                    # Look for next major heading (all caps) - but only if it's clearly a new section
                    r'\n\s*[A-Z][A-Z\s]+[A-Z]\n',
                    # Look for page breaks
                    r'\n\s*---\s*PAGE\s*\d+\s*---\s*\n'
                ]
                
                # Find the earliest occurrence of any next section marker
                end_pos = len(remaining_text)
                for pattern in next_section_markers:
                    matches = list(re.finditer(pattern, remaining_text, re.IGNORECASE))
                    if matches:
                        match_pos = matches[0].start()
                        # Only use this marker if it's not too close to the start (avoid false positives)
                        # AND if it's clearly a new section (not just a subheading)
                        if match_pos > 500 and match_pos < end_pos:  # Increased minimum distance
                            # Additional check: make sure this is really a new section, not a subheading
                            match_text = remaining_text[match_pos:match_pos + 100]
                            if any(keyword in match_text.lower() for keyword in ['component', 'annex', 'section', 'chapter']):
                                end_pos = match_pos
                
                # Extract the content
                section_content = remaining_text[:end_pos].strip()
                
                # Clean the content
                if len(section_content) > 100:  # Ensure we have substantial content
                    cleaned_content = clean_section_content(section_content)
                    if cleaned_content:
                        found_sections[actual_heading] = cleaned_content
                        print(f"✅ Found: {actual_heading} ({len(cleaned_content):,} chars)")
                        
                        # Show a preview
                        preview = cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content
                        print(f"   Preview: {preview}")
                        
                        # Also show what comes after to help debug
                        if end_pos < len(remaining_text):
                            after_preview = remaining_text[end_pos:end_pos + 200].strip()
                            print(f"   After section: {after_preview}")
                else:
                    print(f"❌ Content too short for {heading}: {len(section_content)} chars")
            else:
                print(f"❌ Could not find exact position for {heading}")
        else:
            print(f"❌ Heading '{heading}' not found in text")
    
    print(f"\n🔍 Section extraction complete. Found {len(found_sections)} sections")
    return found_sections

def clean_section_content(content: str) -> str:
    """Clean extracted section content while preserving structure"""
    if not content:
        return ""
    
    # Remove page references and numbers
    content = re.sub(r'Page \d+\s+of\s+\d+', '', content, flags=re.I)
    content = re.sub(r'\n\d+\s*\n', '\n', content)
    
    # Remove excessive whitespace but preserve paragraph structure
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    content = re.sub(r'[ \t]+', ' ', content)
    
    # Clean up line breaks within sentences (common in PDFs)
    content = re.sub(r'([a-z])\n([a-z])', r'\1 \2', content)
    
    # Remove standalone page numbers
    content = re.sub(r'\n\s*\d+\s*\n', '\n', content)
    
    # Clean up excessive dots and formatting artifacts
    content = re.sub(r'\.{3,}', '...', content)
    content = re.sub(r'\.\s*\.', '.', content)
    
    # Remove footer/header artifacts
    content = re.sub(r'^\s*[A-Z\s]+\s*\n', '', content, flags=re.MULTILINE)
    
    return content.strip()

def normalize_pdf_text(text: str) -> str:
    """Normalize PDF text to handle common formatting issues"""
    if not text:
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with single newline
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Handle common PDF formatting issues
    text = re.sub(r'\.{2,}', '.', text)  # Replace multiple dots with single dot
    text = re.sub(r'\.\s*\.', '.', text)  # Replace dot-space-dot with single dot
    
    # Normalize spacing around headings
    text = re.sub(r'([A-Z]\.)\s+([A-Z])', r'\1 \2', text)  # Fix "A. PDO" spacing
    
    # Handle Roman numerals
    text = re.sub(r'([IVX]+\.)\s+([A-Z])', r'\1 \2', text)  # Fix "II. PROJECT" spacing
    
    # Handle potential PDF OCR issues
    text = re.sub(r'([A-Z])\s*\.\s*([A-Z])', r'\1. \2', text)  # Fix "A . PDO" -> "A. PDO"
    text = re.sub(r'([A-Z])\s*\.\s*([A-Z])', r'\1. \2', text)  # Fix "A . PDO" -> "A. PDO"
    
    # Handle potential line breaks in headings
    text = re.sub(r'([A-Z]\.)\s*\n\s*([A-Z])', r'\1 \2', text)  # Fix "A.\nPDO" -> "A. PDO"
    
    # Handle potential extra spaces around colons
    text = re.sub(r'\s*:\s*', ': ', text)
    
    # Clean up excessive whitespace at the beginning and end
    text = text.strip()
    
    return text

def get_available_target_headings(text: str) -> List[str]:
    """Find which TARGET_HEADINGS are actually present in the document"""
    # Normalize the text first
    normalized_text = normalize_pdf_text(text)
    
    available = []
    
    for heading in TARGET_HEADINGS:
        # Check if heading exists in text using more flexible patterns for PAD documents
        patterns = [
            # Pattern 1: Basic heading followed by newline
            rf"(?:^|\n)\s*({re.escape(heading)})\s*\n",
            # Pattern 2: Heading with colon or dash
            rf"(?:^|\n)\s*({re.escape(heading)})\s*[:\-]\s*",
            # Pattern 3: Heading with optional punctuation
            rf"(?:^|\n)\s*({re.escape(heading)})(?:\s|\.|\:|\-)*\n",
            # Pattern 4: Very flexible - just look for heading anywhere
            rf"({re.escape(heading)})",
            # Pattern 5: Case-insensitive search
            rf"(?i)({re.escape(heading)})",
            # Pattern 6: Handle headings with dots and spaces (like "A. PDO")
            rf"(?:^|\n)\s*[A-Z]\.\s*({re.escape(heading.replace('A. ', '').replace('B. ', '').replace('C. ', '').replace('D. ', ''))})\s*",
            # Pattern 7: Handle headings with multiple dots and formatting
            rf"(?:^|\n)\s*[A-Z]\.\s*({re.escape(heading.replace('A. ', '').replace('B. ', '').replace('C. ', '').replace('D. ', ''))})\s*\.+\s*\d+",
            # Pattern 8: Handle headings that might be followed by page numbers
            rf"({re.escape(heading)})\s*\d+",
            # Pattern 9: Handle headings with potential formatting variations
            rf"({re.escape(heading)})\s*(?:\n|\.|:|-|$)",
            # Pattern 10: Handle headings with potential spacing issues
            rf"\s*({re.escape(heading)})\s*"
        ]
        
        found = False
        for pattern_idx, pattern in enumerate(patterns):
            if re.search(pattern, normalized_text, re.MULTILINE):
                found = True
                break
        
        if found:
            available.append(heading)
    
    return available

def extract_target_sections_only(text: str) -> str:
    """Extract and combine text from TARGET_HEADINGS only, automatically detecting which ones exist"""
    
    # Find which target headings are present
    available_headings = get_available_target_headings(text)
    
    if not available_headings:
        return ""
    
    # Extract sections for available headings
    sections = extract_target_sections(text, available_headings)
    
    if not sections:
        return ""
    
    # Combine sections with clear separation
    combined_text = "\n\n--- SECTION BREAK ---\n\n".join([
        f"## {heading}\n{content}" 
        for heading, content in sections.items()
    ])
    
    return combined_text

def get_analysis_mode_text(text: str, analysis_mode: str) -> tuple[str, List[str]]:
    """
    Get text based on analysis mode.
    Returns: (extracted_text, found_headings)
    """
    if analysis_mode == "target_headings_only":
        # Extract only TARGET_HEADINGS content
        available_headings = get_available_target_headings(text)
        extracted_text = extract_target_sections_only(text)
        return extracted_text, available_headings
    else:
        # Return full text
        return text, []

# Legacy function for backward compatibility (remove later)
def get_available_headings(text: str) -> List[str]:
    """Legacy function - use get_available_target_headings instead"""
    return get_available_target_headings(text)

# Legacy function for backward compatibility (remove later)  
def extract_sections_by_headings(text: str, selected_headings: List[str]) -> str:
    """Legacy function - use extract_target_sections_only for automatic detection"""
    sections = extract_target_sections(text, selected_headings)
    
    if not sections:
        return ""
    
    combined_text = "\n\n".join([
        f"SECTION: {heading}\n{content}" 
        for heading, content in sections.items()
    ])
    
    return combined_text

# Test function for debugging
def test_heading_detection(text: str) -> Dict[str, Any]:
    """Test function to debug heading detection issues"""
    print("=" * 50)
    print("DEBUGGING HEADING DETECTION")
    print("=" * 50)
    
    # Get available headings
    available = get_available_target_headings(text)
    
    # Try to extract sections
    sections = extract_target_sections(text, available)
    
    # Return debug info
    return {
        "text_length": len(text),
        "text_preview": text[:1000],
        "available_headings": available,
        "sections_found": len(sections),
        "section_names": list(sections.keys()) if sections else []
    }

def debug_heading_detection(text: str) -> Dict[str, Any]:
    """Debug function to help understand heading detection issues"""
    normalized_text = normalize_pdf_text(text)
    
    debug_info = {
        "text_length": len(text),
        "normalized_length": len(normalized_text),
        "target_headings": TARGET_HEADINGS,
        "found_headings": [],
        "text_samples": {},
        "detection_details": {}
    }
    
    for heading in TARGET_HEADINGS:
        # Check if heading exists in text
        found = False
        patterns_tried = []
        
        # Try different patterns
        patterns = [
            rf"(?:^|\n)\s*({re.escape(heading)})\s*\n",
            rf"({re.escape(heading)})",
            rf"(?i)({re.escape(heading)})",
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, normalized_text, re.MULTILINE))
            if matches:
                found = True
                patterns_tried.append(pattern)
                
                # Get context around the match
                for match in matches:
                    start = max(0, match.start() - 100)
                    end = min(len(normalized_text), match.end() + 200)
                    context = normalized_text[start:end]
                    debug_info["text_samples"][heading] = context
                break
        
        if found:
            debug_info["found_headings"].append(heading)
            debug_info["detection_details"][heading] = {
                "found": True,
                "patterns_worked": patterns_tried
            }
        else:
            debug_info["detection_details"][heading] = {
                "found": False,
                "patterns_worked": []
            }
    
    return debug_info

def debug_pdf_text_structure(text: str) -> Dict[str, Any]:
    """Debug function to understand PDF text structure and find potential headings"""
    normalized_text = normalize_pdf_text(text)
    
    debug_info = {
        "text_length": len(text),
        "normalized_length": len(normalized_text),
        "first_500_chars": text[:500] if text else "",
        "last_500_chars": text[-500:] if text else "",
        "potential_headings": [],
        "text_structure": {}
    }
    
    # Look for potential heading patterns in the text
    lines = normalized_text.split('\n')
    debug_info["total_lines"] = len(lines)
    
    # Find lines that might be headings
    potential_headings = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            # Look for lines that start with letter + dot pattern
            if re.match(r'^[A-Z]\.\s+[A-Z]', line):
                potential_headings.append({
                    "line_number": i,
                    "text": line,
                    "context_before": lines[max(0, i-2):i],
                    "context_after": lines[i+1:min(len(lines), i+4)]
                })
            # Look for lines that are all caps and might be headings
            elif re.match(r'^[A-Z\s]+$', line) and len(line) > 5 and len(line) < 100:
                potential_headings.append({
                    "line_number": i,
                    "text": line,
                    "context_before": lines[max(0, i-2):i],
                    "context_after": lines[i+1:min(len(lines), i+4)]
                })
            # Look for lines that contain "ANNEX" or similar
            elif re.search(r'ANNEX|APPENDIX|SECTION|CHAPTER', line, re.IGNORECASE):
                potential_headings.append({
                    "line_number": i,
                    "text": line,
                    "context_before": lines[max(0, i-2):i],
                    "context_after": lines[i+1:min(len(lines), i+4)]
                })
    
    debug_info["potential_headings"] = potential_headings[:10]  # Limit to first 10
    
    # Show text structure around line numbers
    debug_info["text_structure"] = {
        "lines_1_10": lines[:10],
        "lines_50_60": lines[50:60] if len(lines) > 60 else lines[50:],
        "lines_100_110": lines[100:110] if len(lines) > 110 else lines[100:],
        "lines_200_210": lines[200:210] if len(lines) > 210 else lines[200:]
    }
    
    return debug_info

# Initialize analytics data from file if exists

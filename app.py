# streamlit_pad_semantic_matcher_enhanced.py
"""
Enhanced PAD Document Semantic Matcher with LLM Validation
- OCR with Mistral (fallback to PyPDF2)
- Extract specific sections only
- Multiple chunking methods (sentences/paragraphs/fixed)
- True semantic matching (not just embedding similarity)
- LLM validation for contextual meaning
- Show ALL scores including negatives
- No duplicates, proper validation
"""

import os
import re
import io
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import PyPDF2
from typing import List, Dict, Tuple

# Optional imports for AI services - gracefully handle missing packages
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain.chat_models import init_chat_model
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_mistralai import ChatMistralAI
    MISTRAL_LANGCHAIN_AVAILABLE = True
except ImportError:
    MISTRAL_LANGCHAIN_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────
# Load environment variables from .env file
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use environment variables instead of hardcoded keys
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

# Check if API keys are loaded
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
    st.info("Create a .env file in the project root with: OPENAI_API_KEY=your_key_here")

if not os.getenv("MISTRAL_API_KEY"):
    st.warning("⚠️ Mistral API key not found. OCR will use PyPDF2 only")
    st.info("Add to .env file: MISTRAL_API_KEY=your_key_here")

if not os.getenv("HUGGINGFACE_API_KEY"):
    st.warning("⚠️ HuggingFace API key not found")
    st.info("Add to .env file: HUGGINGFACE_API_KEY=your_key_here")

TARGET_HEADINGS = [
    "A. PDO", "A. Project Development Objective", "A. Project Development Objectives",
    "B. Project Beneficiaries", "Project Beneficiaries", "C. Project Beneficiaries", 
    "A. Project Components", "B. Project Components", "Project Components",
    "ANNEX 1B: THEORY OF CHANGE", "ANNEX 2: DETAILED PROJECT DESCRIPTION",
    "Annex 2: Detailed Project Description", "DETAILED PROJECT DESCRIPTION",
    "D. Results Chain", "D. Theory of Change", "RESULTS CHAIN"
]

# ── LLM Model Initialization ──────────────────────────────────────────
def init_llm_model(model_choice: str):
    """Initialize the selected LLM model"""
    if not LANGCHAIN_AVAILABLE:
        st.error("❌ LangChain not available. Please install langchain packages.")
        return None
        
    if model_choice == "OpenAI GPT-4o-mini":
        return init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
    elif model_choice == "DeepSeek LLaMA-70B" and GROQ_AVAILABLE:
        return ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2
        )
    elif model_choice == "Mistral Large" and MISTRAL_LANGCHAIN_AVAILABLE:
        return ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2
        )
    else:
        st.error(f"❌ {model_choice} not available. Missing dependencies.")
        return None

# ── LLM Contextual Validation ─────────────────────────────────────────
def validate_contextual_meaning(activity_name: str, definition: str, chunk: str, model) -> Dict:
    """Use LLM to validate if chunk has same contextual meaning as definition"""
    
    system_template = """
You are a World Bank education policy expert specializing in semantic analysis.

Your task is to determine if a text chunk has the SAME CONTEXTUAL MEANING as an activity definition.

INSTRUCTIONS:
1. Analyze if the chunk describes the EXACT SAME ACTION/PURPOSE as the definition
2. Look for semantic equivalence, not just keyword overlap
3. Consider synonyms and paraphrased expressions
4. Provide your analysis in this EXACT format:

DECISION: [CONTEXTUAL MATCH or NO CONTEXTUAL MATCH]
REASONING: [Your detailed explanation of why you made this decision]

EXAMPLES:

Definition: "Textbook production and distribution"
Chunk: "The project will finance printing and delivery of educational materials"
DECISION: CONTEXTUAL MATCH
REASONING: Both the definition and chunk describe the same core activities - creating/producing educational materials (textbooks vs educational materials) and distributing/delivering them to end users. The actions are semantically equivalent.

Definition: "Teacher training programs"
Chunk: "Professional development workshops for educators will be conducted"
DECISION: CONTEXTUAL MATCH
REASONING: Both describe training/developing teachers. "Professional development workshops" is synonymous with "training programs" and "educators" means "teachers".

Definition: "School infrastructure construction"
Chunk: "The curriculum will be updated to include new subjects"
DECISION: NO CONTEXTUAL MATCH
REASONING: The definition is about physical construction of school buildings/facilities, while the chunk is about curriculum content development. These are completely different activities with different purposes.

Definition: "Technology procurement for schools"
Chunk: "The project discusses the importance of educational technology"
DECISION: NO CONTEXTUAL MATCH
REASONING: The definition requires actual procurement/purchasing of technology, while the chunk only discusses or mentions technology. Discussion is not the same action as procurement.

Be strict: Only match if the core action and purpose are semantically equivalent.
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", """
Activity Name: {activity_name}
Definition: {definition}
Text Chunk: {chunk}

Analyze if this text chunk has the same contextual meaning as the definition. Provide your response in the exact format specified.
""")
    ])

    try:
        chain = prompt_template | model | StrOutputParser()
        result = chain.invoke({
            "activity_name": activity_name,
            "definition": definition,
            "chunk": chunk
        })
        
        # Parse the structured response
        lines = result.strip().split('\n')
        decision = "NO CONTEXTUAL MATCH"  # Default
        reasoning = result.strip()  # Fallback to full response
        
        for line in lines:
            if line.strip().startswith("DECISION:"):
                decision_text = line.replace("DECISION:", "").strip().upper()
                if "CONTEXTUAL MATCH" in decision_text and "NO CONTEXTUAL MATCH" not in decision_text:
                    decision = "CONTEXTUAL MATCH"
                else:
                    decision = "NO CONTEXTUAL MATCH"
            elif line.strip().startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        return {
            "status": decision,
            "reasoning": reasoning,
            "full_response": result.strip()
        }
            
    except Exception as e:
        return {
            "status": "LLM ERROR",
            "reasoning": f"Error occurred during LLM processing: {str(e)}",
            "full_response": f"Error: {str(e)}"
        }
# ── OCR Functions ──────────────────────────────────────────────────────
def extract_pdf_with_ocr(pdf_bytes: bytes) -> str:
    """Try Mistral OCR, fallback to PyPDF2"""
    if not MISTRAL_AVAILABLE:
        st.info("ℹ️ Mistral OCR not available. Using PyPDF2...")
        return extract_pdf_with_pypdf2(pdf_bytes)
        
    try:
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_file.name = "document.pdf"
        
        uploaded_file = client.files.upload(file=pdf_file)
        ocr_response = client.ocr.process(document=uploaded_file.url, model="mistral-ocr-latest")
        
        full_text = ""
        for page_num, page in enumerate(ocr_response.pages, 1):
            full_text += f"\n--- PAGE {page_num} ---\n"
            if hasattr(page, 'markdown') and page.markdown:
                full_text += page.markdown.strip()
            elif hasattr(page, 'text') and page.text:
                full_text += page.text.strip()
            full_text += "\n"
        
        return full_text
        
    except Exception as e:
        st.warning(f"OCR failed: {e}. Using PyPDF2...")
        return extract_pdf_with_pypdf2(pdf_bytes)

def extract_pdf_with_pypdf2(pdf_bytes: bytes) -> str:
    """Fallback PDF extraction"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- PAGE {page_num} ---\n" + page_text
        return text
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return ""

@st.cache_data(show_spinner="🔍 Processing PDF...")
def cached_pdf_extraction(pdf_bytes: bytes) -> str:
    return extract_pdf_with_ocr(pdf_bytes)

# ── Section Extraction ─────────────────────────────────────────────────
def extract_target_sections(text: str) -> Dict[str, str]:
    """Extract complete content under target headings with ULTRA-GREEDY extraction"""
    global re  # Ensure re module is accessible
    found_sections = {}
    
    st.info(f"🔍 Debug: Starting section extraction from {len(text):,} characters")
    
    for heading in TARGET_HEADINGS:
        st.info(f"🔍 Searching for: '{heading}'")
        
        # Enhanced pattern matching - more flexible approach
        heading_lower = heading.lower()
        text_lower = text.lower()
        
        all_positions = []
        
        # Method 1: Exact match (current approach)
        start_pos = 0
        while True:
            pos = text_lower.find(heading_lower, start_pos)
            if pos == -1:
                break
            all_positions.append(pos)
            start_pos = pos + 1
        
        # Method 2: More flexible pattern matching for common variations
        if not all_positions:
            # Try with different spacing and punctuation
            flexible_patterns = [
                heading_lower.replace('. ', '.'),  # Remove space after period
                heading_lower.replace('.', '. '),  # Add space after period  
                heading_lower.replace('  ', ' '),  # Normalize double spaces
                heading_lower.replace(' ', '\\s*'),  # Allow flexible whitespace
            ]
            
            for pattern in flexible_patterns:
                if pattern != heading_lower:  # Don't repeat exact match
                    matches = list(re.finditer(re.escape(pattern).replace('\\\\s\\*', '\\s*'), text_lower))
                    for match in matches:
                        if match.start() not in all_positions:
                            all_positions.append(match.start())
        
        # Method 3: Try word boundary matching for short headings like "A. PDO"
        if not all_positions and len(heading.split()) <= 3:
            # Create a pattern that looks for the heading at word boundaries
            escaped_heading = re.escape(heading_lower)
            word_boundary_pattern = r'\b' + escaped_heading + r'\b'
            matches = list(re.finditer(word_boundary_pattern, text_lower))
            for match in matches:
                all_positions.append(match.start())
        
        if not all_positions:
            st.warning(f"❌ Could not find: {heading}")
            
            # Debug: Show nearby text that might contain the heading
            if heading.lower() in ["a. pdo", "a. project development objective"]:
                # Look for similar patterns in the text
                import re
                pdo_patterns = [
                    r'[^\n]*pdo[^\n]*',
                    r'[^\n]*project development objective[^\n]*',
                    r'[^\n]*a\.\s*pdo[^\n]*',
                    r'28\.[^\n]*proposed project development[^\n]*'
                ]
                
                st.info(f"🔍 Debug: Looking for PDO-related text patterns...")
                for pattern in pdo_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        st.info(f"🔍 Found PDO-related text: {matches[:3]}")  # Show first 3 matches
                        # Try to find the actual position
                        for match in matches[:3]:
                            match_pos = text.lower().find(match.lower())
                            if match_pos != -1:
                                # Show context around this match
                                context_start = max(0, match_pos - 200)
                                context_end = min(len(text), match_pos + 500)
                                context = text[context_start:context_end]
                                st.info(f"🔍 Context around '{match[:50]}...': {context[:300]}...")
            continue
            
        st.info(f"🔍 Found {len(all_positions)} occurrences of '{heading}' at positions: {all_positions}")
        
        # Try each occurrence to find the one with actual content (not TOC)
        best_content = None
        best_length = 0
        best_pos = None
        
        for pos in all_positions:
            st.info(f"🔍 Analyzing occurrence at position {pos}")
            
            # Start extracting from after the heading
            start_pos = pos + len(heading)
            
            # For headings that might be followed by a number (like "A. PDO" followed by "28."), 
            # we need to be more careful about where to start
            remaining_after_heading = text[start_pos:start_pos + 50]
            
            # Skip any immediate punctuation/whitespace after heading
            while start_pos < len(text) and text[start_pos] in ':\n\t -':
                start_pos += 1
            
            # Special handling for numbered paragraphs (like "28. The proposed...")
            # If we see a number followed by a period right after the heading, include it
            number_match = re.match(r'\s*(\d+)\.\s*', text[start_pos:start_pos + 20])
            if number_match:
                st.info(f"🔍 Position {pos}: Found numbered paragraph starting with '{number_match.group(1)}.'")
            
            # Get a preview of what follows to check if it's TOC or actual content
            preview_text = text[start_pos:start_pos + 500]
            
            # Check if this looks like table of contents (has lots of dots and page numbers)
            dot_pattern_count = len(re.findall(r'\.{3,}', preview_text))
            page_number_pattern_count = len(re.findall(r'\d+\s*$', preview_text.split('\n')[0]))
            
            # Additional check: PDO content should contain key phrases
            is_pdo_content = False
            if "pdo" in heading.lower():
                pdo_indicators = [
                    "project development objective",
                    "proposed project development",
                    "objectives are to",
                    "improve",
                    "strengthen",
                    "facilitate"
                ]
                is_pdo_content = any(indicator in preview_text.lower() for indicator in pdo_indicators)
                if is_pdo_content:
                    st.info(f"🔍 Position {pos}: Detected PDO content with key indicators")
            
            # Skip if this looks like table of contents (unless it's confirmed PDO content)
            if (dot_pattern_count > 2 or 'Page' in preview_text[:100]) and not is_pdo_content:
                st.info(f"🔍 Position {pos}: Skipping - looks like table of contents (dots: {dot_pattern_count})")
                continue
            
            # Now find where this section should end
            remaining_text = text[start_pos:]
            
            # Look for the NEXT major section heading (be very conservative)
            next_section_candidates = []
            
            # Check for other target headings that might come after this one
            for other_heading in TARGET_HEADINGS:
                if other_heading != heading:
                    other_pos = remaining_text.lower().find(other_heading.lower())
                    if other_pos != -1:
                        # Make sure it's not in TOC either
                        other_preview = remaining_text[other_pos:other_pos + 100]
                        if not re.search(r'\.{3,}', other_preview):
                            next_section_candidates.append(other_pos)
            
            # Also look for major document structures that would definitely end this section
            major_terminators = [
                r'\n\s*(?:ANNEX|Annex)\s+\d+[:\s]',  # Next ANNEX
                r'\n\s*[IVX]+\.\s+[A-Z][a-z]+',      # Roman numeral sections
                r'\n\s*REFERENCES?\s*$',              # References section
                r'\n\s*BIBLIOGRAPHY\s*$',             # Bibliography 
                r'\n\s*APPENDIX\s+[A-Z\d]',          # Appendix
            ]
            
            for pattern in major_terminators:
                match = re.search(pattern, remaining_text, re.MULTILINE | re.IGNORECASE)
                if match:
                    next_section_candidates.append(match.start())
            
            # Use the earliest boundary found, or go to end of document
            if next_section_candidates:
                end_pos = min(next_section_candidates)
                content = remaining_text[:end_pos].strip()
            else:
                # No clear boundary found - take everything to end of document
                content = remaining_text.strip()
            
            st.info(f"🔍 Position {pos}: Extracted {len(content):,} characters")
            
            # Show preview
            preview = content[:300] + "..." if len(content) > 300 else content
            st.info(f"🔍 Position {pos} Preview: {preview}")
            
            # Quality check - make sure we got substantial content (not just TOC)
            if len(content) > 200:  # Increased minimum length
                words = content.split()
                
                # Check for actual content indicators
                has_sentences = bool(re.search(r'[.!?]\s+[A-Z]', content))
                has_substantial_words = len(words) > 50
                low_dot_density = content.count('.') / len(content) < 0.1 if content else False
                
                if has_substantial_words and (has_sentences or low_dot_density):
                    if len(content) > best_length:
                        best_content = content
                        best_length = len(content)
                        best_pos = pos
                        st.info(f"🔍 Position {pos}: NEW BEST MATCH - {len(content):,} chars")
                else:
                    st.info(f"🔍 Position {pos}: Quality check failed - sentences: {has_sentences}, words: {len(words)}, dots: {content.count('.')}")
            else:
                st.info(f"🔍 Position {pos}: Too short ({len(content)} chars)")
        
        # Use the best content found
        if best_content:
            cleaned = clean_section_content(best_content)
            if cleaned:
                found_sections[heading] = cleaned
                st.success(f"✅ ULTRA-GREEDY extraction: {heading} ({len(cleaned):,} chars) from position {best_pos}")
                
                # Show stats about extracted content
                lines = cleaned.split('\n')
                paragraphs = [p for p in cleaned.split('\n\n') if p.strip()]
                words = cleaned.split()
                st.info(f"📊 Content stats: {len(lines)} lines, {len(paragraphs)} paragraphs, {len(words)} words")
                
                # Show first few sentences for verification
                sentences = re.split(r'[.!?]+', cleaned)[:3]
                first_sentences = [s.strip() for s in sentences if s.strip()]
                if first_sentences:
                    st.info(f"📖 First sentences: {'. '.join(first_sentences[:2])}...")
            else:
                st.warning(f"⚠️ Content cleaned to empty for: {heading}")
        else:
            st.warning(f"⚠️ No valid content found for: {heading}")
    
    st.info(f"🔍 Debug: ULTRA-GREEDY extraction complete. Found {len(found_sections)} sections")
    
    # Show final summary
    total_extracted_chars = sum(len(content) for content in found_sections.values())
    st.info(f"📊 Total extracted: {total_extracted_chars:,} characters from {len(found_sections)} sections")
    
    return found_sections

def clean_section_content(content: str) -> str:
    """Clean extracted section content while preserving structure"""
    # Remove page footers/headers but keep content
    content = re.sub(r'Page \d+\s+of\s+\d+', '', content, flags=re.I)
    
    # Remove isolated page numbers but keep numbered items
    content = re.sub(r'\n\s*\d+\s*\n(?=\s*\n)', '\n', content)
    
    # Reduce excessive whitespace but preserve paragraph structure
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # Normalize spaces within lines but preserve line breaks
    content = re.sub(r'[ \t]+', ' ', content)
    
    # Remove trailing footnote numbers at the end (like "25" at the end)
    content = re.sub(r'\s+\d+\s*$', '', content)
    
    # Clean up any leading/trailing whitespace
    content = content.strip()
    
    return content

# ── Text Chunking ──────────────────────────────────────────────────────
def chunk_text(text: str, method: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Tuple[int, str]]:
    """Split text using different methods"""
    
    # Clean text first
    text = clean_text_for_chunking(text)
    
    if method == "Sentences (till fullstop)":
        return split_into_sentences(text)
    elif method == "Paragraphs":
        return split_into_paragraphs(text)
    else:  # Fixed size chunks
        return split_into_fixed_chunks(text, chunk_size, chunk_overlap)

def clean_text_for_chunking(text: str) -> str:
    """Clean text before chunking - preserve more content"""
    original_length = len(text)
    
    # Remove page markers but keep content
    text = re.sub(r'--- PAGE \d+ ---\s*', '', text)
    
    # Remove section headers but keep content 
    text = re.sub(r'SECTION:\s*[^\n]*\n', '', text)
    
    # Remove excessive lines but preserve paragraphs
    text = re.sub(r'={20,}', '', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Normalize spaces but preserve structure
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove page footers/headers (less aggressive)
    text = re.sub(r'Page \d+\s+of\s+\d+', '', text, flags=re.I)
    
    # Only remove isolated single numbers, not all numbers
    text = re.sub(r'\n\s*\d{1,2}\s*\n(?=\s*\n)', '\n', text)
    
    cleaned = text.strip()
    loss_percentage = ((original_length - len(cleaned)) / original_length * 100) if original_length > 0 else 0
    
    st.info(f"🔍 Debug: Text cleaned from {original_length:,} to {len(cleaned):,} characters ({loss_percentage:.1f}% loss)")
    
    return cleaned

def split_into_sentences(text: str) -> List[Tuple[int, str]]:
    """Split into sentences with quality filtering"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except:
            pass
    
    # Debug: Show text length before processing
    st.info(f"🔍 Debug: Processing {len(text):,} characters for sentence splitting")
    
    try:
        sentences = nltk.sent_tokenize(text)
    except:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    st.info(f"🔍 Debug: Found {len(sentences)} raw sentences before filtering")
    
    unique_sentences = set()
    result = []
    sid = 1
    filtered_out = 0
    
    for sentence in sentences:
        cleaned = clean_chunk(sentence)
        if is_valid_sentence(cleaned):
            if cleaned.lower() not in unique_sentences:
                unique_sentences.add(cleaned.lower())
                result.append((sid, cleaned))
                sid += 1
            else:
                filtered_out += 1
        else:
            filtered_out += 1
    
    st.info(f"🔍 Debug: Kept {len(result)} sentences, filtered out {filtered_out} (duplicates or invalid)")
    
    return result

def split_into_paragraphs(text: str) -> List[Tuple[int, str]]:
    """Split into paragraphs"""
    paragraphs = re.split(r'\n\s*\n', text)
    
    unique_paragraphs = set()
    result = []
    pid = 1
    
    for paragraph in paragraphs:
        cleaned = clean_chunk(paragraph)
        if is_valid_paragraph(cleaned) and cleaned.lower() not in unique_paragraphs:
            unique_paragraphs.add(cleaned.lower())
            result.append((pid, cleaned))
            pid += 1
    
    return result

def split_into_fixed_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, str]]:
    """Split into fixed-size chunks with overlap"""
    chunks = []
    cid = 1
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            last_sentence = max(
                text.rfind('.', start, end),
                text.rfind('!', start, end),
                text.rfind('?', start, end)
            )
            if last_sentence > start + chunk_size // 2:
                end = last_sentence + 1
        
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append((cid, chunk))
            cid += 1
        
        start = max(start + chunk_size - chunk_overlap, end)
    
    return chunks

def clean_chunk(chunk: str) -> str:
    """Clean individual chunk"""
    chunk = chunk.strip()
    chunk = re.sub(r'^[^\w\s]+', '', chunk)
    chunk = re.sub(r'[^\w\s.!?]+$', '', chunk)
    chunk = re.sub(r'\s+', ' ', chunk)
    return chunk.strip()

def is_valid_sentence(sentence: str) -> bool:
    """Check if text is a valid sentence with relaxed criteria"""
    if not sentence or len(sentence) < 10:  # Further reduced from 15 to 10
        return False
    
    words = sentence.split()
    if len(words) < 2:  # Further reduced from 3 to 2
        return False
    
    # Skip obvious fragments like "43 Project Components 32."
    if re.match(r'^\d+\s+[A-Z][^.]*\s+\d+\.?\s*$', sentence):
        return False
    
    # Skip pure page numbers or section headers
    if re.match(r'^Page \d+|^\d+\s*$|^[IVX]+\.\s*$', sentence.strip()):
        return False
    
    # Much more relaxed meaningful content check - just need some letters
    if not re.search(r'[a-zA-Z]', sentence):
        return False
    
    return True

def is_valid_paragraph(paragraph: str) -> bool:
    """Check if text is a valid paragraph"""
    if not paragraph or len(paragraph) < 20:  # Reduced from 50 to 20
        return False
    
    words = paragraph.split()
    if len(words) < 4:  # Reduced from 8 to 4
        return False
    
    return True

# ── Semantic Similarity ────────────────────────────────────────────────
def compute_semantic_similarity(definition: str, chunk: str, activity_name: str) -> Dict:
    """Compute semantic similarity with detailed analysis"""
    
    if not LANGCHAIN_AVAILABLE:
        st.error("❌ Cannot compute semantic similarity - LangChain not available")
        return {
            'cosine_similarity': 0.0,
            'semantic_score': 0.0,
            'final_similarity': 0.0,
            'matching_reasons': ['LangChain not available'],
            'semantic_issues': ['Missing dependencies'],
            'word_overlap': 0.0
        }
    
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Get embeddings and compute cosine similarity
    def_embedding = np.array(embeddings_model.embed_query(definition))
    chunk_embedding = np.array(embeddings_model.embed_query(chunk))
    
    def_norm = def_embedding / np.linalg.norm(def_embedding)
    chunk_norm = chunk_embedding / np.linalg.norm(chunk_embedding)
    
    cosine_sim = float(np.dot(def_norm, chunk_norm))
    
    # Semantic validation
    semantic_score = 0.0
    matching_reasons = []
    semantic_issues = []
    
    # Word overlap analysis
    def_words = set(definition.lower().split())
    chunk_words = set(chunk.lower().split())
    stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are'}
    
    def_words_clean = def_words - stop_words
    chunk_words_clean = chunk_words - stop_words
    
    word_overlap = 0.0
    if def_words_clean:
        word_overlap = len(def_words_clean.intersection(chunk_words_clean)) / len(def_words_clean)
        if word_overlap > 0.3:
            semantic_score += 0.3
            matching_reasons.append(f"High word overlap: {word_overlap:.2%}")
        elif word_overlap > 0.1:
            semantic_score += 0.1
            matching_reasons.append(f"Moderate word overlap: {word_overlap:.2%}")
        else:
            semantic_issues.append(f"Low word overlap: {word_overlap:.2%}")
    
    # Topic alignment
    topic_keywords = {
        'teacher': ['teacher', 'instructor', 'educator', 'faculty'],
        'student': ['student', 'pupil', 'learner', 'children'],
        'curriculum': ['curriculum', 'syllabus', 'course', 'program'],
        'infrastructure': ['infrastructure', 'building', 'construction', 'facility'],
        'technology': ['technology', 'digital', 'computer', 'software'],
        'training': ['training', 'development', 'capacity', 'workshop'],
        'assessment': ['assessment', 'evaluation', 'test', 'exam'],
        'textbook': ['textbook', 'book', 'material', 'resource']
    }
    
    activity_lower = activity_name.lower()
    definition_lower = definition.lower()
    chunk_lower = chunk.lower()
    
    for topic, keywords in topic_keywords.items():
        if topic in activity_lower or topic in definition_lower:
            if any(keyword in chunk_lower for keyword in keywords):
                semantic_score += 0.2
                matching_reasons.append(f"Topic alignment: {topic}")
                break
    else:
        semantic_issues.append("No clear topic alignment")
    
    # Action alignment
    action_words = {
        'provide': ['provide', 'supply', 'deliver', 'offer'],
        'develop': ['develop', 'create', 'build', 'establish'],
        'improve': ['improve', 'enhance', 'strengthen', 'upgrade'],
        'support': ['support', 'assist', 'help', 'facilitate'],
        'train': ['train', 'educate', 'teach', 'instruct']
    }
    
    for action, synonyms in action_words.items():
        if any(word in definition_lower for word in synonyms):
            if any(word in chunk_lower for word in synonyms):
                semantic_score += 0.2
                matching_reasons.append(f"Action alignment: {action}")
                break
    else:
        semantic_issues.append("No clear action alignment")
    
    # Activity indicators
    activity_indicators = ['will', 'project', 'component', 'support', 'provide', 'include']
    if any(indicator in chunk_lower for indicator in activity_indicators):
        semantic_score += 0.1
        matching_reasons.append("Contains activity indicators")
    
    # Final scoring
    semantic_score = max(0, min(1, semantic_score))
    final_score = 0.7 * cosine_sim + 0.3 * (semantic_score * 2 - 1)
    
    return {
        'cosine_similarity': round(cosine_sim, 6),
        'semantic_score': round(semantic_score, 6),
        'final_similarity': round(final_score, 6),
        'matching_reasons': matching_reasons,
        'semantic_issues': semantic_issues,
        'word_overlap': round(word_overlap, 3)
    }

def compute_all_similarities_with_llm_fast(chunks_with_ids: List[Tuple[int, str]], 
                                          activities_df: pd.DataFrame,
                                          chunk_type: str,
                                          use_semantic: bool = True,
                                          use_llm_validation: bool = True,
                                          model = None,
                                          llm_threshold: float = 0.3) -> List[Dict]:
    """ULTRA-FAST computation with aggressive pre-filtering"""
    
    # Determine column names
    if chunk_type == "sentences":
        id_col, content_col = "Sentence_ID", "Sentence"
    elif chunk_type == "paragraphs":
        id_col, content_col = "Paragraph_ID", "Paragraph"
    else:
        id_col, content_col = "Chunk_ID", "Chunk"
    
    st.info("🚀 **PROCESSING MODE**: Computing basic text similarities...")
    
    # If no advanced features are available, use basic text matching
    if not LANGCHAIN_AVAILABLE or not use_semantic:
        st.info("⚡ Using basic text similarity (keyword matching)...")
        
        all_results = []
        
        progress_bar = st.progress(0)
        total_pairs = len(activities_df) * len(chunks_with_ids)
        
        for activity_idx, (_, activity_row) in enumerate(activities_df.iterrows()):
            for chunk_idx, (chunk_id, chunk) in enumerate(chunks_with_ids):
                # Basic keyword similarity
                definition_words = set(activity_row['Definition'].lower().split())
                chunk_words = set(chunk.lower().split())
                
                # Remove common stop words
                stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'will', 'be', 'this', 'that'}
                definition_words_clean = definition_words - stop_words
                chunk_words_clean = chunk_words - stop_words
                
                # Calculate Jaccard similarity
                intersection = len(definition_words_clean.intersection(chunk_words_clean))
                union = len(definition_words_clean.union(chunk_words_clean))
                
                jaccard_similarity = intersection / union if union > 0 else 0.0
                
                # Basic result
                result = {
                    'Activity Name': activity_row['Activity Name'],
                    'Definition': activity_row['Definition'],
                    id_col: chunk_id,
                    content_col: chunk,
                    'FAISS_Similarity': round(jaccard_similarity, 6),
                    'LLM_Validation': "NOT PROCESSED",
                    'LLM_Output': "Basic text matching only",
                    'LLM_Matched_Sentence': "NOT PROCESSED"
                }
                
                all_results.append(result)
                
                # Update progress
                current_pair = activity_idx * len(chunks_with_ids) + chunk_idx + 1
                progress_bar.progress(current_pair / total_pairs)
        
        st.success(f"✅ Computed {len(all_results):,} basic similarity pairs")
        return all_results
    
    # Original advanced processing if dependencies are available
    st.info("🚀 **ULTRA-FAST MODE**: Using advanced semantic processing...")
    
    # STEP 1: Quick keyword-based pre-filtering (VERY FAST)
    st.info("⚡ Step 1: Keyword-based pre-filtering...")
    
    if not LANGCHAIN_AVAILABLE:
        st.error("❌ Cannot compute similarities - LangChain not available")
        return []
    
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Pre-compute all embeddings in batches (faster)
    st.info("🔄 Pre-computing embeddings...")
    
    # Activity embeddings
    activity_queries = []
    for _, row in activities_df.iterrows():
        activity_queries.append(f"{row['Activity Name']}: {row['Definition']}")
    
    # Chunk embeddings  
    chunk_texts = [chunk for _, chunk in chunks_with_ids]
    
    # Batch embedding computation
    progress_bar = st.progress(0)
    progress_bar.progress(0.1)
    
    activity_embeddings = embeddings_model.embed_documents(activity_queries)
    progress_bar.progress(0.5)
    
    chunk_embeddings = embeddings_model.embed_documents(chunk_texts)
    progress_bar.progress(0.8)
    
    # Convert to numpy arrays
    activity_embeddings = np.array(activity_embeddings)
    chunk_embeddings = np.array(chunk_embeddings)
    
    # Normalize embeddings
    activity_embeddings = activity_embeddings / np.linalg.norm(activity_embeddings, axis=1, keepdims=True)
    chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    
    progress_bar.progress(1.0)
    st.success("✅ Embeddings computed!")
    
    # STEP 2: Vectorized similarity computation (ULTRA FAST)
    st.info("⚡ Step 2: Vectorized similarity computation...")
    
    # Compute all similarities at once using matrix multiplication
    similarity_matrix = np.dot(activity_embeddings, chunk_embeddings.T)
    
    # STEP 3: Extract only high-similarity pairs
    st.info(f"⚡ Step 3: Extracting pairs above threshold {llm_threshold}...")
    
    high_sim_pairs = []
    all_similarities = []
    
    for activity_idx in range(len(activities_df)):
        for chunk_idx in range(len(chunks_with_ids)):
            similarity = similarity_matrix[activity_idx, chunk_idx]
            all_similarities.append(similarity)
            
            # Only process high-similarity pairs for detailed analysis
            if similarity > llm_threshold:
                activity_row = activities_df.iloc[activity_idx]
                chunk_id, chunk = chunks_with_ids[chunk_idx]
                
                high_sim_pairs.append({
                    'activity_idx': activity_idx,
                    'chunk_idx': chunk_idx,
                    'activity_name': activity_row['Activity Name'],
                    'definition': activity_row['Definition'],
                    'chunk_id': chunk_id,
                    'chunk': chunk,
                    'cosine_similarity': float(similarity)
                })
    
    st.info(f"🎯 Found {len(high_sim_pairs):,} high-similarity pairs from {len(all_similarities):,} total")
    
    # STEP 4: Detailed semantic analysis only for high-similarity pairs
    detailed_results = []
    
    if len(high_sim_pairs) > 0:
        st.info("⚡ Step 4: Detailed semantic analysis for high-similarity pairs...")
        progress_bar = st.progress(0)
        
        for idx, pair in enumerate(high_sim_pairs):
            progress_bar.progress((idx + 1) / len(high_sim_pairs))
            
            if use_semantic:
                # Detailed semantic analysis
                sim_result = compute_semantic_similarity(
                    pair['definition'], 
                    pair['chunk'], 
                    pair['activity_name']
                )
                final_similarity = sim_result['final_similarity']
                
                result = {
                    'Activity Name': pair['activity_name'],
                    'Definition': pair['definition'],
                    id_col: pair['chunk_id'],
                    content_col: pair['chunk'],
                    'FAISS_Similarity': final_similarity,
                    'Cosine_Similarity': sim_result['cosine_similarity'],
                    'Semantic_Score': sim_result['semantic_score'],
                    'Word_Overlap': sim_result['word_overlap'],
                    'Matching_Reasons': "; ".join(sim_result['matching_reasons']) if sim_result['matching_reasons'] else "No clear reasons",
                    'Semantic_Issues': "; ".join(sim_result['semantic_issues']) if sim_result['semantic_issues'] else "None"
                }
            else:
                # Simple cosine similarity
                result = {
                    'Activity Name': pair['activity_name'],
                    'Definition': pair['definition'],
                    id_col: pair['chunk_id'],
                    content_col: pair['chunk'],
                    'FAISS_Similarity': round(pair['cosine_similarity'], 6)
                }
            
            detailed_results.append(result)
    
    # STEP 5: LLM validation for top candidates only
    if use_llm_validation and model and len(detailed_results) > 0:
        # Further filter for LLM validation - only top 20% of high-similarity pairs
        df_temp = pd.DataFrame(detailed_results)
        if len(df_temp) > 0:
            top_percentile = df_temp['FAISS_Similarity'].quantile(0.8)  # Top 20%
            llm_candidates = df_temp[df_temp['FAISS_Similarity'] >= top_percentile]
        else:
            llm_candidates = pd.DataFrame()
        
        st.info(f"🤖 Step 5: LLM validation for top {len(llm_candidates):,} candidates...")
        
        llm_results = {}
        if len(llm_candidates) > 0:
            llm_progress = st.progress(0)
            
            for idx, (_, row) in enumerate(llm_candidates.iterrows()):
                # Ensure progress doesn't exceed 1.0
                progress_value = min(1.0, (idx + 1) / len(llm_candidates))
                llm_progress.progress(progress_value)
                
                key = f"{row['Activity Name']}|||{row[content_col]}"
                
                if key not in llm_results:
                    try:
                        llm_result = validate_contextual_meaning(
                            row['Activity Name'], 
                            row['Definition'], 
                            row[content_col], 
                            model
                        )
                        llm_results[key] = llm_result
                    except Exception as e:
                        llm_results[key] = {
                            "status": "LLM ERROR", 
                            "reasoning": f"Error occurred during LLM processing: {str(e)}",
                            "full_response": f"Error: {str(e)}"
                        }
        
        # Apply LLM results
        for result in detailed_results:
            key = f"{result['Activity Name']}|||{result[content_col]}"
            
            if key in llm_results:
                llm_result = llm_results[key]
                result['LLM_Validation'] = llm_result['status']
                result['LLM_Output'] = llm_result['reasoning']
                
                # Set matched sentence based on validation result
                if llm_result['status'] == "CONTEXTUAL MATCH":
                    result['LLM_Matched_Sentence'] = result[content_col]
                else:
                    result['LLM_Matched_Sentence'] = "NO CONTEXTUAL SAME MEANING"
            else:
                result['LLM_Validation'] = "SKIPPED_LOWER_PRIORITY"
                result['LLM_Output'] = "Skipped - not in top 20% of high-similarity pairs"
                result['LLM_Matched_Sentence'] = "NOT PROCESSED"
    else:
        # No LLM validation
        for result in detailed_results:
            result['LLM_Validation'] = "NOT PROCESSED"
            result['LLM_Output'] = "LLM validation disabled"
            result['LLM_Matched_Sentence'] = "NOT PROCESSED"
    
    # STEP 6: Add low-similarity placeholder results for completeness
    all_results = detailed_results.copy()
    
    # Add placeholder entries for low-similarity pairs (for download completeness)
    processed_pairs = set()
    for result in detailed_results:
        key = f"{result['Activity Name']}|||{result[content_col]}"
        processed_pairs.add(key)
    
    # Add remaining low-similarity pairs as placeholders
    low_sim_count = 0
    for activity_idx in range(len(activities_df)):
        for chunk_idx in range(len(chunks_with_ids)):
            activity_row = activities_df.iloc[activity_idx]
            chunk_id, chunk = chunks_with_ids[chunk_idx]
            key = f"{activity_row['Activity Name']}|||{chunk}"
            
            if key not in processed_pairs:
                similarity = similarity_matrix[activity_idx, chunk_idx]
                
                placeholder_result = {
                    'Activity Name': activity_row['Activity Name'],
                    'Definition': activity_row['Definition'],
                    id_col: chunk_id,
                    content_col: chunk,
                    'FAISS_Similarity': round(float(similarity), 6),
                    'LLM_Validation': "SKIPPED_LOW_SIMILARITY",
                    'LLM_Output': f"Skipped due to low similarity score ({similarity:.3f} < {llm_threshold})",
                    'LLM_Matched_Sentence': "NOT PROCESSED"
                }
                
                if use_semantic:
                    placeholder_result.update({
                        'Cosine_Similarity': round(float(similarity), 6),
                        'Semantic_Score': 0,
                        'Word_Overlap': 0,
                        'Matching_Reasons': "Not analyzed - low similarity",
                        'Semantic_Issues': "Skipped due to low similarity"
                    })
                
                all_results.append(placeholder_result)
                low_sim_count += 1
    
    # Show similarity distribution
    if all_similarities:
        min_sim = min(all_similarities)
        max_sim = max(all_similarities)
        avg_sim = sum(all_similarities) / len(all_similarities)
        negative_count = sum(1 for sim in all_similarities if sim < 0)
        positive_count = sum(1 for sim in all_similarities if sim > 0)
        
        st.info(f"""
        📊 **ULTRA-FAST Processing Results:**
        - **Total pairs**: {len(all_similarities):,}
        - **High-similarity pairs** (>{llm_threshold}): {len(high_sim_pairs):,}
        - **Detailed analysis**: {len(detailed_results):,}
        - **LLM validated**: {len([r for r in all_results if r['LLM_Validation'] not in ['NOT PROCESSED', 'SKIPPED_LOW_SIMILARITY', 'SKIPPED_LOWER_PRIORITY']]):,}
        - **Similarity range**: {min_sim:.3f} to {max_sim:.3f}
        - **Average similarity**: {avg_sim:.3f}
        """)
    
    return all_results

# ── Streamlit UI ───────────────────────────────────────────────────────
st.title("🔍 Enhanced PAD Document Semantic Matcher with LLM Validation")
st.markdown("**Extract sections from PAD documents and find semantic matches with activities using LLM validation**")

# Debug: Check if environment variables are loaded
openai_key = os.getenv("OPENAI_API_KEY")
mistral_key = os.getenv("MISTRAL_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

# Debug prints (will show in terminal)
print(f"Debug - OpenAI key: {'FOUND' if openai_key else 'NOT FOUND'}")
print(f"Debug - Mistral key: {'FOUND' if mistral_key else 'NOT FOUND'}")
print(f"Debug - Groq key: {'FOUND' if groq_key else 'NOT FOUND'}")

# Show API key status in sidebar
with st.sidebar:
    st.subheader("🔑 API Key Status")
    if openai_key:
        st.success(f"✅ OpenAI API Key loaded (ends with: ...{openai_key[-8:]})")
    else:
        st.error("❌ OpenAI API Key not found")
        st.info("💡 Make sure .env file is in the project root directory")
    
    if mistral_key:
        st.success(f"✅ Mistral API Key loaded (ends with: ...{mistral_key[-8:]})")
    else:
        st.warning("⚠️ Mistral API Key not found")
    
    if groq_key:
        st.success(f"✅ Groq API Key loaded (ends with: ...{groq_key[-8:]})")
    else:
        st.warning("⚠️ Groq API Key not found")

# Show warnings for missing packages
if not MISTRAL_AVAILABLE:
    st.warning("⚠️ Mistral AI not available. OCR will use PyPDF2 only.")

if not LANGCHAIN_AVAILABLE:
    st.warning("⚠️ LangChain packages not available. Advanced features will be disabled.")

# Model selection
st.subheader("🤖 Model Configuration")

# Only show available models
available_models = []
if LANGCHAIN_AVAILABLE:
    available_models.append("OpenAI GPT-4o-mini")
if GROQ_AVAILABLE:
    available_models.append("DeepSeek LLaMA-70B") 
if MISTRAL_LANGCHAIN_AVAILABLE:
    available_models.append("Mistral Large")

if available_models:
    model_choice = st.selectbox("Choose a model to run:", available_models)
else:
    st.info("ℹ️ No AI models available. The app will work with basic text matching only.")
    model_choice = None

# Configuration
st.subheader("⚙️ Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    min_similarity = st.slider("Minimum Similarity", -1.0, 1.0, -1.0, 0.01)
with col2:
    max_results = st.number_input("Max results per activity", 10, 500, 100)
with col3:
    chunking_method = st.selectbox("Chunking Method", 
                                  ["Sentences (till fullstop)", "Paragraphs", "Fixed size chunks"])

# Options
col1, col2, col3 = st.columns(3)
with col1:
    use_semantic = st.checkbox("Enable Semantic Validation", value=False, disabled=not LANGCHAIN_AVAILABLE, 
                               help="Requires OpenAI API key and LangChain packages")
with col2:
    use_llm_validation = st.checkbox("Enable LLM Contextual Validation", value=False, disabled=not available_models,
                                     help="Requires AI model access and API keys")
with col3:
    if use_llm_validation and available_models:
        llm_threshold = st.slider("LLM Validation Threshold", 0.1, 0.8, 0.3, 0.1, 
                                 help="Only validate pairs above this similarity score")
    else:
        llm_threshold = 0.3

# Show status of features
st.info(f"""
**Feature Status:**
- 📄 PDF Processing: {'✅ Available (PyPDF2)' if True else '❌ Not available'}
- 🧠 Semantic Validation: {'✅ Available' if LANGCHAIN_AVAILABLE and use_semantic else '❌ Disabled'}
- 🤖 LLM Validation: {'✅ Available' if available_models and use_llm_validation else '❌ Disabled'}
- 🔍 Basic Text Matching: ✅ Always available
""")

# Reset options if dependencies not available
if use_semantic and not LANGCHAIN_AVAILABLE:
    use_semantic = False
    
if use_llm_validation and not available_models:
    use_llm_validation = False

# Additional options for fixed chunks
if chunking_method == "Fixed size chunks":
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk size", 100, 2000, 500)
    with col2:
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 100)

# File uploads
st.subheader("📁 File Upload")
pdf_file = st.file_uploader("📄 Upload PAD PDF", type=["pdf"])
activities_file = st.file_uploader("📑 Upload Activities (CSV/Excel)", type=["xlsx", "xls", "csv"])

if pdf_file and activities_file:
    
    # Initialize LLM model if validation is enabled and models are available
    model = None
    if use_llm_validation and available_models and model_choice:
        try:
            with st.spinner(f"Initializing {model_choice}..."):
                model = init_llm_model(model_choice)
            if model:
                st.success(f"✅ {model_choice} initialized successfully")
            else:
                st.warning(f"⚠️ Failed to initialize {model_choice}. LLM validation will be disabled.")
                use_llm_validation = False
        except Exception as e:
            st.warning(f"⚠️ Failed to initialize {model_choice}: {e}. LLM validation will be disabled.")
            use_llm_validation = False
    elif use_llm_validation:
        st.info("ℹ️ LLM validation requested but no models available. Using basic text matching.")
        use_llm_validation = False
    
    # Get PDF filename for download
    pdf_filename = pdf_file.name.replace('.pdf', '') if pdf_file.name.endswith('.pdf') else pdf_file.name
    
    # Step 1: Extract PDF
    st.subheader("📖 Step 1: PDF Processing")
    raw_text = cached_pdf_extraction(pdf_file.getvalue())
    
    if not raw_text.strip():
        st.error("❌ No text extracted from PDF")
        st.stop()
    
    st.success(f"✅ Extracted {len(raw_text):,} characters")
    
    # Step 2: Extract sections
    st.subheader("🎯 Step 2: Section Extraction")
    
    # Debug: Show what we're looking for
    with st.expander("🔍 Debug: Target Headings Search", expanded=True):
        st.info("Searching for the following target headings:")
        for i, heading in enumerate(TARGET_HEADINGS):
            st.write(f"{i+1}. {heading}")
    
    sections = extract_target_sections(raw_text)
    
    if not sections:
        st.error("❌ No target sections found")
        with st.expander("🔍 Debug Info"):
            st.text_area("Raw text preview", raw_text[:3000], height=300)
            for heading in TARGET_HEADINGS:
                if heading.lower() in raw_text.lower():
                    st.success(f"✅ Found text: {heading}")
                else:
                    st.error(f"❌ Missing: {heading}")
        st.stop()
    
    # Combine sections
    combined_text = "\n\n".join([f"SECTION: {h}\n{content}" for h, content in sections.items()])
    st.success(f"✅ Extracted {len(sections)} sections ({len(combined_text):,} chars)")
    
    # NEW: Section Summary
    st.subheader("📊 Section Extraction Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sections", len(sections))
    with col2:
        st.metric("Total Characters", f"{len(combined_text):,}")
    with col3:
        st.metric("Total Words", f"{len(combined_text.split()):,}")
    with col4:
        # Count paragraphs by splitting on double newlines
        paragraph_count = len([p for p in combined_text.split('\n\n') if p.strip()])
        st.metric("Total Paragraphs", f"{paragraph_count:,}")
    
    # Show what was found vs what was searched for
    st.info("**Target Headings Search Results:**")
    found_headings = list(sections.keys())
    missing_headings = [h for h in TARGET_HEADINGS if h not in found_headings]
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ **Found ({len(found_headings)}):**")
        for heading in found_headings:
            content_length = len(sections[heading])
            st.write(f"• {heading} ({content_length:,} chars)")
    
    with col2:
        if missing_headings:
            st.warning(f"❌ **Missing ({len(missing_headings)}):**")
            for heading in missing_headings:
                st.write(f"• {heading}")
        else:
            st.success("✅ **All target headings found!**")
    
    # NEW: Show complete extracted content for each section
    st.subheader("📖 Complete Extracted Sections")
    st.info("Below is the COMPLETE text extracted under each heading:")
    
    for heading, content in sections.items():
        with st.expander(f"📖 {heading} - {len(content):,} characters", expanded=True):
            st.markdown(f"**Complete Content:**")
            # Show the FULL content, not truncated
            st.text_area(
                f"Full text for {heading}", 
                content,  # Show complete content, not truncated
                height=600,  # Increased height to show more content
                key=f"complete_{heading}",
                help=f"Complete extracted content under {heading}"
            )
            
            # Show content statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", f"{len(content):,}")
            with col2:
                st.metric("Words", f"{len(content.split()):,}")
            with col3:
                # Count paragraphs by splitting on double newlines
                paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
                st.metric("Paragraphs", f"{paragraph_count:,}")
            
            # Show a preview of what comes after this section to help verify boundaries
            st.markdown("**Content Analysis:**")
            st.write(f"• **First 200 chars:** {content[:200]}...")
            st.write(f"• **Last 200 chars:** ...{content[-200:]}")
            st.write(f"• **Total lines:** {len(content.split(chr(10)))}")
    
    # NEW: Section Review Step
    st.subheader("📋 Step 2.5: Review Extracted Sections")
    st.info("Please review the complete extracted content under each heading before proceeding to analysis.")
    
    # Show complete sections for review
    section_approval = {}
    for heading, content in sections.items():
        with st.expander(f"📖 {heading} - {len(content):,} characters", expanded=True):
            st.markdown(f"**Complete Content:**")
            st.text_area(
                f"Full text for {heading}", 
                content,  # Show complete content
                height=600,  # Increased height
                key=f"review_{heading}",
                help=f"Complete extracted content under {heading}"
            )
            
            # Approval checkbox
            approved = st.checkbox(
                f"✅ Approve content for {heading}", 
                value=True, 
                key=f"approve_{heading}"
            )
            section_approval[heading] = approved
            
            # Show content statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", f"{len(content):,}")
            with col2:
                st.metric("Words", f"{len(content.split()):,}")
            with col3:
                # Count paragraphs by splitting on double newlines
                paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
                st.metric("Paragraphs", f"{paragraph_count:,}")
    
    # Check if all sections are approved
    all_approved = all(section_approval.values())
    if not all_approved:
        st.warning("⚠️ Please approve all sections before proceeding to analysis.")
        st.stop()
    
    st.success("✅ All sections approved! Proceeding to text chunking...")
    
    # Debug: Show section sizes
    with st.expander("📊 Section Sizes Debug"):
        for heading, content in sections.items():
            st.write(f"**{heading}**: {len(content):,} characters")
        st.write(f"**Total Combined**: {len(combined_text):,} characters")
    
    # Remove the old "View Sections" expander since we now show complete content above
    # with st.expander("📋 View Sections"):
    #     for heading, content in sections.items():
    #         st.subheader(heading)
    #         st.text_area(f"Content for {heading}", content[:1000] + "..." if len(content) > 1000 else content, 
    #                     height=200, key=f"section_{heading}")
    #         st.write(f"Full length: {len(content):,} characters")
    
    # Step 3: Text chunking
    st.subheader("✂️ Step 3: Text Chunking")
    st.info("Creating text chunks from the approved sections for analysis...")
    
    # Debug: Show what text is being chunked
    with st.expander("🔍 Debug: Text Being Chunked", expanded=True):
        st.write(f"**Total text length:** {len(combined_text):,} characters")
        st.write(f"**Number of sections:** {len(sections)}")
        st.write(f"**Combined text preview (first 1000 chars):**")
        st.text_area("Combined text preview", combined_text[:1000] + "..." if len(combined_text) > 1000 else combined_text, height=200)
        st.write(f"**Combined text preview (last 1000 chars):**")
        st.text_area("Combined text end preview", combined_text[-1000:] if len(combined_text) > 1000 else combined_text, height=200)
    
    if chunking_method == "Fixed size chunks":
        chunks = chunk_text(combined_text, chunking_method, chunk_size, chunk_overlap)
    else:
        chunks = chunk_text(combined_text, chunking_method)
    
    if not chunks:
        st.error("❌ No valid chunks created")
        st.stop()
    
    chunk_type = chunking_method.split()[0].lower()
    st.success(f"✅ Created {len(chunks)} {chunk_type}")
    
    # Show chunk preview with better formatting
    with st.expander(f"📝 Preview {chunk_type.title()} (showing first 10)", expanded=True):
        for i, (cid, chunk) in enumerate(chunks[:10]):
            st.markdown(f"**{cid}.** {chunk[:300]}{'...' if len(chunk) > 300 else ''}")
            st.caption(f"Length: {len(chunk):,} characters")
            if i < 9:  # Add separator between chunks
                st.divider()
        if len(chunks) > 10:
            st.info(f"... and {len(chunks) - 10} more {chunk_type}")
    
    # Debug: Show chunking statistics
    with st.expander("🔍 Debug: Chunking Statistics", expanded=True):
        total_chunk_chars = sum(len(chunk) for _, chunk in chunks)
        st.write(f"**Total characters in chunks:** {total_chunk_chars:,}")
        st.write(f"**Original text length:** {len(combined_text):,}")
        st.write(f"**Chunking efficiency:** {(total_chunk_chars / len(combined_text) * 100):.1f}%")
        if total_chunk_chars < len(combined_text) * 0.8:  # If we're losing more than 20% of text
            st.warning("⚠️ Warning: Significant text loss during chunking!")
            st.write("This might indicate that the chunking function is not processing all the text.")
    
    # Step 4: Load activities
    st.subheader("📊 Step 4: Load Activities")
    st.info("Loading and validating the activities file...")
    
    try:
        if activities_file.name.endswith(("xlsx", "xls")):
            df_activities = pd.read_excel(activities_file)
        else:
            df_activities = pd.read_csv(activities_file)
        
        if 'Activity Name' not in df_activities.columns:
            st.error("❌ Missing 'Activity Name' column")
            st.stop()
        
        def_col = 'Definition' if 'Definition' in df_activities.columns else 'Definitions'
        if def_col not in df_activities.columns:
            st.error("❌ Missing 'Definition' or 'Definitions' column")
            st.stop()
        
        df_activities = df_activities[['Activity Name', def_col]].rename(columns={def_col: 'Definition'}).dropna()
        st.success(f"✅ Loaded {len(df_activities)} activities")
        
        # Show activities preview
        with st.expander("📋 Preview Activities (showing first 10)", expanded=True):
            st.dataframe(df_activities.head(10), use_container_width=True)
            if len(df_activities) > 10:
                st.info(f"... and {len(df_activities) - 10} more activities")
        
    except Exception as e:
        st.error(f"❌ Error loading activities: {e}")
        st.stop()
    
    # Step 5: Compute similarities with LLM validation
    st.subheader("🧠 Step 5: Computing Similarities with LLM Validation")
    st.info(f"Processing {len(df_activities)} activities × {len(chunks)} {chunk_type} = {len(df_activities) * len(chunks):,} total pairs")
    
    # Show processing configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Activities", len(df_activities))
    with col2:
        st.metric(f"{chunk_type.title()}", len(chunks))
    with col3:
        st.metric("Total Pairs", f"{len(df_activities) * len(chunks):,}")
    
    # Confirmation before starting heavy computation
    proceed_with_analysis = st.checkbox(
        "🚀 Start Analysis", 
        value=False,
        help="Check this box to start the similarity computation and LLM validation"
    )
    
    if not proceed_with_analysis:
        st.info("👆 Check the box above to start the analysis process")
        st.stop()
    
    with st.spinner(f"Computing semantic similarities and LLM validation..."):
        results = compute_all_similarities_with_llm_fast(
            chunks, df_activities, chunk_type, use_semantic, use_llm_validation, model, llm_threshold
        )
    
    results_df = pd.DataFrame(results)
    filtered_df = results_df[results_df['FAISS_Similarity'] >= min_similarity].copy()
    filtered_df = filtered_df.sort_values('FAISS_Similarity', ascending=False)
    
    st.success(f"✅ Computed {len(results_df):,} similarity pairs")
    st.info(f"📊 {len(filtered_df):,} results above threshold {min_similarity}")
    
    # Show LLM validation statistics if enabled
    if use_llm_validation and 'LLM_Validation' in results_df.columns:
        llm_stats = results_df['LLM_Validation'].value_counts()
        st.info(f"""
        🤖 **LLM Validation Results:**
        - Contextual Matches: {llm_stats.get('CONTEXTUAL MATCH', 0):,}
        - No Contextual Matches: {llm_stats.get('NO CONTEXTUAL MATCH', 0):,}
        - Errors: {llm_stats.get('ERROR', 0):,}
        """)
    
    # Step 6: Results
    st.subheader("📈 Step 6: Results")
    
    if len(filtered_df) > 0:
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Activities", len(df_activities))
        with col2:
            st.metric(f"{chunk_type.title()}", len(chunks))
        with col3:
            st.metric("Above Threshold", len(filtered_df))
        with col4:
            avg_sim = filtered_df['FAISS_Similarity'].mean()
            st.metric("Avg Similarity", f"{avg_sim:.3f}")
        
        # Display results
        display_df = filtered_df.groupby('Activity Name').head(max_results)
        
        # Color coding function
        def highlight_scores(row):
            colors = []
            for col in row.index:
                if col == 'LLM_Validation':
                    if row[col] == 'CONTEXTUAL MATCH':
                        colors.append('background-color: #d4edda')  # Green
                    elif row[col] == 'NO CONTEXTUAL MATCH':
                        colors.append('background-color: #f8d7da')  # Red
                    elif row[col] == 'ERROR':
                        colors.append('background-color: #fff3cd')  # Yellow
                    else:
                        colors.append('')
                elif use_semantic and col == 'Semantic_Score':
                    if row[col] >= 0.7:
                        colors.append('background-color: #d4edda')  # Green
                    elif row[col] >= 0.4:
                        colors.append('background-color: #fff3cd')  # Yellow
                    else:
                        colors.append('')
                elif col == 'FAISS_Similarity' and row[col] < 0:
                    colors.append('background-color: #f8d7da')  # Red
                else:
                    colors.append('')
            return colors
        
        # Apply styling
        styled_df = display_df.style.apply(highlight_scores, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Analysis
        with st.expander("📊 Analysis by Activity"):
            if use_semantic and use_llm_validation:
                stats = filtered_df.groupby('Activity Name').agg({
                    'FAISS_Similarity': ['count', 'mean', 'max', 'min'],
                    'Semantic_Score': ['mean', 'max'],
                    'Word_Overlap': ['mean', 'max'],
                    'LLM_Validation': lambda x: (x == 'CONTEXTUAL MATCH').sum()
                }).round(4)
                stats.columns = ['Count', 'Avg_Sim', 'Max_Sim', 'Min_Sim', 'Avg_Semantic', 'Max_Semantic', 'Avg_Overlap', 'Max_Overlap', 'LLM_Matches']
            elif use_semantic:
                stats = filtered_df.groupby('Activity Name').agg({
                    'FAISS_Similarity': ['count', 'mean', 'max', 'min'],
                    'Semantic_Score': ['mean', 'max'],
                    'Word_Overlap': ['mean', 'max']
                }).round(4)
                stats.columns = ['Count', 'Avg_Sim', 'Max_Sim', 'Min_Sim', 'Avg_Semantic', 'Max_Semantic', 'Avg_Overlap', 'Max_Overlap']
            else:
                stats = filtered_df.groupby('Activity Name').agg({
                    'FAISS_Similarity': ['count', 'mean', 'max', 'min']
                }).round(4)
                stats.columns = ['Count', 'Avg_Sim', 'Max_Sim', 'Min_Sim']
            
            st.dataframe(stats.sort_values('Avg_Sim', ascending=False))
        
        # Show LLM matched sentences summary
        if use_llm_validation and 'LLM_Matched_Sentence' in filtered_df.columns:
            with st.expander("🎯 LLM Contextually Matched Sentences"):
                matched_df = filtered_df[filtered_df['LLM_Validation'] == 'CONTEXTUAL MATCH']
                if len(matched_df) > 0:
                    st.write(f"**Found {len(matched_df)} contextually matched sentences:**")
                    for idx, row in matched_df.head(20).iterrows():
                        st.write(f"**{row['Activity Name']}:** {row['LLM_Matched_Sentence'][:300]}...")
                    if len(matched_df) > 20:
                        st.write(f"... and {len(matched_df) - 20} more matches")
                else:
                    st.write("No contextual matches found by LLM validation.")
        
        # Downloads
        st.subheader("💾 Downloads")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            full_csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Complete Results", 
                full_csv, 
                f"{pdf_filename}_complete_{chunk_type}_results.csv", 
                "text/csv"
            )
        
        with col2:
            filtered_csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Filtered Results", 
                filtered_csv,
                f"{pdf_filename}_filtered_{chunk_type}_results.csv", 
                "text/csv"
            )
        
        with col3:
            if use_llm_validation:
                matched_only_df = filtered_df[filtered_df['LLM_Validation'] == 'CONTEXTUAL MATCH']
                if len(matched_only_df) > 0:
                    matched_csv = matched_only_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ LLM Matched Only", 
                        matched_csv,
                        f"{pdf_filename}_llm_matched_{chunk_type}_results.csv", 
                        "text/csv"
                    )
                else:
                    st.info("No LLM matches to download")
    
    else:
        st.warning(f"No results above threshold {min_similarity}")
        
        # Debug info
        if len(results_df) > 0:
            st.subheader("🔍 Debug Information")
            st.write("**Score distribution:**")
            st.write(results_df['FAISS_Similarity'].describe())
            
            negative_examples = results_df[results_df['FAISS_Similarity'] < 0].head(5)
            if len(negative_examples) > 0:
                st.write("**Examples with negative scores:**")
                st.dataframe(negative_examples[['Activity Name', 'FAISS_Similarity']])

else:
    st.info("👆 Upload both a PDF and activities file to begin")
    
    with st.expander("ℹ️ Help"):
        st.markdown("""
        ### How to use:
        1. **Choose Model**: Select OpenAI GPT-4o-mini, DeepSeek LLaMA-70B, or Mistral Large
        2. **Upload PDF**: Project Appraisal Document
        3. **Upload Activities**: CSV/Excel with 'Activity Name' and 'Definition' columns
        4. **Choose chunking method**: Sentences for precision, paragraphs for context
        5. **Enable semantic validation**: For meaning-based matching (recommended)
        6. **Enable LLM validation**: For contextual meaning verification using selected LLM
        7. **Adjust similarity threshold**: Include negative scores to see all results
        
        ### New LLM Validation Features:
        - **Contextual Analysis**: LLM determines if chunk has same meaning as definition
        - **LLM_Validation Column**: Shows "CONTEXTUAL MATCH" or "NO CONTEXTUAL MATCH"
        - **LLM_Output Column**: Shows LLM's reasoning
        - **LLM_Matched_Sentence Column**: Shows matched sentence or "NO CONTEXTUAL SAME MEANING"
        - **Color Coding**: Green for matches, red for non-matches, yellow for errors
        - **Separate Downloads**: Download only LLM-validated matches
        
        ### Semantic validation features:
        - **Word overlap analysis**: Meaningful word matching
        - **Topic alignment**: Same subject areas (teacher, curriculum, etc.)
        - **Action alignment**: Same actions (provide, develop, support, etc.)
        - **Quality scoring**: Combined similarity + semantic validation
        
        ### File naming:
        - Downloads include the PDF filename for easy identification
        - Three download options: Complete, Filtered, and LLM Matched Only
        """)
        
        st.write("**Expected activities file format:**")
        sample = pd.DataFrame({
            'Activity Name': ['Teacher Training', 'Textbooks', 'Infrastructure'],
            'Definition': ['Professional development for teachers', 'Textbook production and distribution', 'School building construction']
        })
        st.dataframe(sample)
        
        st.write("**Model Information:**")
        st.markdown("""
        - **OpenAI GPT-4o-mini**: Fast and cost-effective, good for most tasks
        - **DeepSeek LLaMA-70B**: Large model with strong reasoning capabilities
        - **Mistral Large**: Balanced performance and speed
        """)
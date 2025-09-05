import os
import streamlit as st
import pandas as pd
import PyPDF2
import re
from io import BytesIO
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

import time
from typing import Callable, Any, Tuple

def with_retry(
    fn: Callable[[], Any],
    tries: int = 3,
    base_delay: float = 2.0,
) -> Tuple[bool, Any]:
    """
    Run `fn` with exponential back-off.
    Returns (success, result_or_exception).
    """
    for attempt in range(tries):
        try:
            return True, fn()
        except Exception as err:
            # stop early if it's NOT a 429 or rate-limit type error
            if "429" not in str(err) and "capacity exceeded" not in str(err).lower():
                return False, err
            # last try? give up
            if attempt == tries - 1:
                return False, err
            time.sleep(base_delay * (2 ** attempt))


# Fixed NLTK data downloading
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        # Fallback to older punkt if punkt_tab fails
        try:
            nltk.download('punkt')
        except:
            st.warning("Could not download NLTK data. Sentence tokenization may not work properly.")

# Environment setup (remove actual keys before sharing)
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Check if API keys are loaded
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")

if not os.getenv("GROQ_API_KEY"):
    print("⚠️ Groq API key not found. Please set GROQ_API_KEY in your .env file")
os.environ["MISTRAL_API_KEY"] = "YlSjwZ5j09EsZxOfR0uxmwFhx7NcJ7gg"
# Disable LangSmith tracing to avoid rate limits
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# os.environ["LANGCHAIN_API_KEY"] = "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"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

st.title("Advanced PAD Activity Definition Matcher")

# Add confidence threshold control
confidence_threshold = st.slider("Minimum Confidence Threshold (%)", 0, 30, 15, 
                                help="Activities below this threshold will be flagged for review")


# NEW – slider to decide how similar a chunk must be to qualify
sim_threshold = st.slider(
    "Minimum embedding similarity (0 – 1.0)",
    0.0, 1.0, 0.55, 0.01,
    help="Chunks scoring below this cosine-similarity threshold are ignored",
)


model_choice = st.selectbox("Choose a model to run:", ["OpenAI GPT-4o-mini", "DeepSeek LLaMA-70B", "Mistral Large"])

if model_choice == "OpenAI GPT-4o-mini":
    model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
elif model_choice == "DeepSeek LLaMA-70B":
    model = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2
    )
elif model_choice == "Mistral Large":
    model = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2
    )

pdf_file = st.file_uploader("Upload PAD PDF", type=["pdf"])
csv_file = st.file_uploader("Upload Activities File", type=["xlsx", "xls", "csv"])

def clean_pdf_text(raw_text):
    """Advanced PDF text cleaning to remove boilerplate and noise"""
    # Remove page numbers and headers/footers
    text = re.sub(r'\n\d+\s*\n', '\n', raw_text)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    
    # Remove common boilerplate patterns
    boilerplate_patterns = [
        r'Table of Contents.*?(?=\n[A-Z])',
        r'ABBREVIATIONS AND ACRONYMS.*?(?=\n[A-Z])',
        r'EXECUTIVE SUMMARY.*?(?=\n[A-Z])',
        r'(?:Internal Use Only|Official Use Only|Confidential).*?\n',
        r'World Bank Group.*?\n',
        r'International Bank for Reconstruction and Development.*?\n',
        r'The World Bank.*?\n',
        r'©.*?World Bank.*?\n',
    ]
    
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove table remnants and formatting artifacts
    text = re.sub(r'(\b[A-Z]+\b[\t ]+\d+(\.\d+)?[\t ]+\S+.*?(\n|$))+', '', text)
    text = re.sub(r'\n\d+[\t ]+\S+.*?(\n|$)', '', text)
    
    return text.strip()

import time

def mistral_section_extractor(raw_text: str) -> str | None:
    """
    Try to pull the key PAD sections with Mistral-Large.
    • Retries (2 s → 4 s → 8 s) on HTTP-429 or capacity errors.
    • On final failure returns None, allowing the caller to fall back
      to the pattern-matching extractor.
    """
    target_headings = [
        "A. PDO", "A. Project Development Objective", "A. Project Development Objectives",
        "B. Project Beneficiaries", "Project Beneficiaries", "C. Project Beneficiaries",
        "A. Project Components", "B. Project Components", "Project Components",
        "ANNEX 1B: THEORY OF CHANGE", "ANNEX 2: DETAILED PROJECT DESCRIPTION",
        "Annex 2: Detailed Project Description", "DETAILED PROJECT DESCRIPTION",
        "D. Results Chain", "D. Theory of Change", "RESULTS CHAIN"
    ]

    extraction_prompt = f"""
You are a World Bank document analysis expert. Your task is to extract ONLY the following sections from a Project Appraisal Document (PAD).

TARGET SECTIONS TO EXTRACT:
{', '.join(target_headings)}

DO NOT extract any section not in this list, even if it looks important.

For each matching heading found:
1. Extract the entire section content starting from that heading until the next major section begins.
2. DO NOT summarize. Include full paragraphs, sub-points, tables, activities, and details.
3. STOP extraction at:
   • The next top-level heading (A., B., C., etc.)
   • Roman numerals (I., II., III., etc.)
   • Section titles in ALL CAPS
   • Words like “ANNEX” or “APPENDIX”

RESPONSE FORMAT:
SECTION: [Exact Heading Found]
CONTENT: [Full extracted section content]

---

Repeat this for each target heading found.

If none of the listed headings are found, return: "NO TARGET SECTIONS FOUND"

DOCUMENT TEXT TO ANALYZE:

{raw_text}
""".strip()

    # Initialise the model once per call
    mistral_extractor = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0,
        max_tokens=4000,
    )

    for attempt in range(3):                     # 0, 1, 2
        try:
            response = mistral_extractor.invoke(extraction_prompt)
            return response.content              # ✅ success
        except Exception as err:
            err_str = str(err).lower()

            # Retry only for rate-limit / capacity issues
            retryable = "429" in err_str or "capacity" in err_str
            st.warning(
                f"Mistral extraction attempt {attempt + 1}/3 failed: {err}"
                + (" – retrying…" if retryable and attempt < 2 else "")
            )

            if not retryable or attempt == 2:
                # Give up → signal caller to use the fallback extractor
                return None

            # Exponential back-off: 2 s, 4 s, 8 s
            time.sleep(2 ** (attempt + 1))


def enhanced_text_extraction(pdf_reader):
    """Enhanced text extraction using Mistral AI for intelligent section detection"""
    raw_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    
    # First, clean the raw text
    cleaned_text = clean_pdf_text(raw_text)
    
    st.info(f"📄 Document length: {len(raw_text):,} characters")
    
    # Use Mistral AI for intelligent section extraction
    st.info("🤖 Using Mistral AI to intelligently extract relevant sections...")
    
    # If document is very large, process in chunks
    if len(raw_text) > 50000:  # If larger than 50k characters
        st.info("📚 Large document detected. Processing in chunks for better extraction...")
        
        # Split into chunks and process each
        chunk_size = 40000
        chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
        all_sections = ""
        
        for i, chunk in enumerate(chunks):
            st.info(f"Processing chunk {i+1}/{len(chunks)}...")
            chunk_sections = mistral_section_extractor(chunk)
            if chunk_sections and chunk_sections != "NO TARGET SECTIONS FOUND":
                all_sections += f"\n\n--- CHUNK {i+1} ---\n{chunk_sections}"
        
        if all_sections:
            mistral_sections = all_sections
        else:
            mistral_sections = None
    else:
        mistral_sections = mistral_section_extractor(raw_text)
    
    if mistral_sections and mistral_sections != "NO TARGET SECTIONS FOUND":
        st.success("✅ Mistral AI found relevant sections!")
        
        # Display what sections were found - Fixed: No nested expanders
        st.subheader("📋 Extracted Sections Preview")
        
        # Split by sections and show each one
        if "SECTION:" in mistral_sections:
            sections = mistral_sections.split("SECTION:")[1:]  # Skip first empty split
            
            # Create tabs for different sections instead of nested expanders
            if len(sections) > 1:
                section_names = []
                section_contents = []
                
                for section in sections:
                    if section.strip():
                        lines = section.strip().split('\n', 1)
                        if len(lines) >= 2:
                            section_name = lines[0].strip()
                            section_content = lines[1].replace("CONTENT:", "").strip()
                            section_names.append(section_name)
                            section_contents.append(section_content)
                
                if section_names:
                    # Create tabs for each section
                    tabs = st.tabs([f"📖 {name}" for name in section_names])
                    
                    for i, (tab, content) in enumerate(zip(tabs, section_contents)):
                        with tab:
                            st.text_area(
                                f"Content for {section_names[i]}:", 
                                content[:2000] + ("..." if len(content) > 2000 else ""),
                                height=300,
                                key=f"section_tab_{i}"
                            )
            else:
                # Single section - just show it directly
                section = sections[0]
                lines = section.strip().split('\n', 1)
                if len(lines) >= 2:
                    section_name = lines[0].strip()
                    section_content = lines[1].replace("CONTENT:", "").strip()
                    st.subheader(f"📖 {section_name}")
                    st.text_area(
                        "Content:", 
                        section_content[:2000] + ("..." if len(section_content) > 2000 else ""),
                        height=300,
                        key="single_section"
                    )
        else:
            st.text_area("Full Extracted Content:", mistral_sections[:3000] + ("..." if len(mistral_sections) > 3000 else ""), height=400)
        
        return mistral_sections
    else:
        st.warning("⚠️ Mistral AI couldn't find target sections. Using fallback extraction...")
        
        # Enhanced fallback method with better section detection
        target_headings = [
            "A. PDO", "A. Project Development Objective", "A. Project Development Objectives",
            "B. Project Beneficiaries", "Project Beneficiaries", "C. Project Beneficiaries",
            "A. Project Components", "B. Project Components", "Project Components",
            
            "ANNEX 1B: THEORY OF CHANGE", "ANNEX 2: DETAILED PROJECT DESCRIPTION",
            "Annex 2: Detailed Project Description", "DETAILED PROJECT DESCRIPTION",
            "D. Results Chain", "D. Theory of Change", "RESULTS CHAIN"
        ]
        
        found_sections = {}
        
        for heading in target_headings:
            # Look for the heading and extract content until next major heading
            heading_pattern = re.escape(heading)
            
            # Pattern to find the heading and capture content until next major section
            pattern = rf"({heading_pattern})\s*(.*?)(?=\n\s*[A-Z]\.?\s+[A-Z][^.]*$|\nANNEX\s+|\n[IVX]+\.\s+|\Z)"
            
            match = re.search(pattern, cleaned_text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            
            if match:
                content = match.group(2).strip()
                if content and len(content) > 50:  # Only include substantial content
                    found_sections[heading] = content
                    st.info(f"📋 Found section: {heading} ({len(content)} characters)")
        
        if found_sections:
            # Combine all found sections
            combined_content = ""
            for heading, content in found_sections.items():
                combined_content += f"\n\nSECTION: {heading}\n{content}\n" + "="*50
            
            # Fixed: Use tabs instead of nested expanders
            st.subheader("📋 Fallback Method - Sections Found")
            if len(found_sections) > 1:
                section_names = list(found_sections.keys())
                section_contents = list(found_sections.values())
                
                tabs = st.tabs([f"📖 {name}" for name in section_names])
                
                for i, (tab, content) in enumerate(zip(tabs, section_contents)):
                    with tab:
                        st.text_area(
                            f"Content for {section_names[i]}:", 
                            content[:2000] + ("..." if len(content) > 2000 else ""),
                            height=300,
                            key=f"fallback_tab_{i}"
                        )
            else:
                # Single section found
                heading, content = list(found_sections.items())[0]
                st.subheader(f"📖 {heading}")
                st.text_area(
                    "Content:", 
                    content[:2000] + ("..." if len(content) > 2000 else ""),
                    height=300,
                    key="fallback_single"
                )
            
            return combined_content
        else:
            st.warning("📄 No specific sections found with any method. Using entire cleaned document...")
            return cleaned_text

def sentence_level_chunking(text):
    """Create sentence-level chunks while preserving context"""
    try:
        # Try to use NLTK for sentence tokenization
        sentences = nltk.sent_tokenize(text)
    except (LookupError, AttributeError):
        # Fallback to simple sentence splitting if NLTK fails
        st.warning("Using fallback sentence tokenization")
        # Simple sentence splitting on common sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group sentences into chunks of 3-5 sentences with overlap
    chunks = []
    for i in range(0, len(sentences), 3):
        chunk_sentences = sentences[i:i+5]  # 5 sentences per chunk
        chunk_text = ' '.join(chunk_sentences)
        chunks.append(Document(page_content=chunk_text))
    
    return chunks

def embedding_similarity_score(definition, activity_name, matched_content):
    """Calculate similarity using OpenAI embeddings instead of TF-IDF"""
    if matched_content == "NO RELEVANT CONTEXT FOUND" or not matched_content.strip():
        return 0.0
    
    # Clean the matched content
    cleaned_content = re.sub(r'^\*\s*', '', matched_content, flags=re.MULTILINE)
    cleaned_content = cleaned_content.strip()
    
    if not cleaned_content:
        return 0.0
    
    try:
        # Combine activity name and definition for better matching
        query_text = f"{activity_name}: {definition}"
        
        # Get embeddings for both texts
        query_embedding = embeddings.embed_query(query_text)
        content_embedding = embeddings.embed_query(cleaned_content)
        
        # Calculate cosine similarity
        query_np = np.array(query_embedding).reshape(1, -1)
        content_np = np.array(content_embedding).reshape(1, -1)
        
        similarity = cosine_similarity(query_np, content_np)[0][0]
        
        return round(similarity * 100, 2)
        
    except Exception as e:
        st.warning(f"Embedding similarity calculation failed: {e}")
        # Fallback to simple word overlap
        definition_words = set(definition.lower().split())
        content_words = set(cleaned_content.lower().split())
        
        if len(definition_words) == 0:
            return 0.0
            
        overlap = len(definition_words.intersection(content_words))
        score = (overlap / len(definition_words)) * 100
        return round(score, 2)

def validate_extraction(activity_name, definition, matched_content):
    """Enhanced validation to catch semantic mismatches with detailed explanations"""
    if matched_content == "NO RELEVANT CONTEXT FOUND":
        return "VALID", ""
    
    issues = []
    
    # Check for completely irrelevant content (secretariat, think tanks, etc.)
    irrelevant_keywords = [
        "secretariat", "platform", "think tanks", "consortia", "policy makers",
        "continental policy", "cross-border issues", "regional priority themes"
    ]
    
    if any(keyword in matched_content.lower() for keyword in irrelevant_keywords):
        if not any(keyword in definition.lower() for keyword in irrelevant_keywords):
            issues.append("IRRELEVANT: Matched content about secretariat/platform/think tanks is unrelated to the activity definition")
    
    # Specific semantic validation checks based on your examples
    
    # Procurement mechanism vs equipment acquisition
    if "procurement" in definition.lower() and "mechanism" in definition.lower():
        if "acquisition" in matched_content.lower() or "purchase" in matched_content.lower():
            if not ("mechanism" in matched_content.lower() or "system" in matched_content.lower() or "process" in matched_content.lower()):
                issues.append("SEMANTIC MISMATCH: Definition is about improving procurement mechanisms, but matched content is about acquiring/purchasing equipment")
    
    # Alternative education - must be outside formal schooling
    if "alternative education" in definition.lower() or "outside.*formal.*school" in definition.lower():
        formal_indicators = ["grade", "curriculum", "school", "formal", "regular classroom"]
        if any(indicator in matched_content.lower() for indicator in formal_indicators):
            if not any(alt in matched_content.lower() for alt in ["non-formal", "alternative", "outside school", "community-based"]):
                issues.append("SEMANTIC MISMATCH: Alternative education must be outside formal schooling, but matched content describes formal education")
    
    # Technology for literacy - must mention both technology AND basic literacy
    if "technology" in definition.lower() and "literacy" in definition.lower():
        has_tech = any(tech in matched_content.lower() for tech in ["technology", "digital", "software", "device", "online"])
        has_literacy = any(lit in matched_content.lower() for lit in ["literacy", "reading", "writing", "basic skills"])
        if not (has_tech and has_literacy):
            issues.append("SEMANTIC MISMATCH: Technology for literacy requires both technology AND basic reading/writing skills")
    
    # Assistive technology - must mention technology AND special needs/disabilities
    if "assistive technology" in definition.lower():
        has_tech = any(tech in matched_content.lower() for tech in ["technology", "digital", "device", "software", "assistive"])
        has_special_needs = any(need in matched_content.lower() for need in ["special needs", "disabilit", "impair", "inclusive", "accommodation"])
        if not (has_tech and has_special_needs):
            issues.append("SEMANTIC MISMATCH: Assistive technology requires technology specifically for special needs/disabilities")
    
    # Teacher training semantic check
    if "teacher training" in definition.lower() or "professional development" in definition.lower():
        training_indicators = ["training", "professional development", "capacity building", "skills development", "learning"]
        teacher_indicators = ["teacher", "educator", "instructor", "faculty"]
        
        has_training = any(indicator in matched_content.lower() for indicator in training_indicators)
        has_teacher = any(indicator in matched_content.lower() for indicator in teacher_indicators)
        
        if not (has_training and has_teacher):
            issues.append("SEMANTIC MISMATCH: Teacher training activity must mention both training/development AND teachers")
    
    # Teacher assessment/performance check
    if "assessment" in definition.lower() or "performance" in definition.lower():
        assessment_indicators = ["assess", "evaluat", "monitor", "performance", "measure", "review"]
        has_assessment = any(indicator in matched_content.lower() for indicator in assessment_indicators)
        
        if not has_assessment:
            issues.append("SEMANTIC MISMATCH: Assessment activity must mention evaluation/monitoring/performance measurement")
    
    # Skills development specificity check
    if "skills" in definition.lower() and activity_name.lower() != "skills in general":
        # Check if the matched content mentions the specific type of skills
        if "technical" in definition.lower() or "vocational" in definition.lower():
            if not any(skill in matched_content.lower() for skill in ["technical", "vocational", "tvet", "trade"]):
                issues.append("SEMANTIC MISMATCH: Technical/vocational skills activity matched to general content")
    
    # General semantic overlap check
    definition_words = set(definition.lower().split())
    content_words = set(matched_content.lower().split())
    
    # Remove common stop words for better analysis
    stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "a", "an"}
    definition_words = definition_words - stop_words
    content_words = content_words - stop_words
    
    if definition_words:
        word_overlap = len(definition_words.intersection(content_words)) / len(definition_words)
        if word_overlap < 0.1:  # Less than 10% meaningful word overlap
            issues.append("LOW SEMANTIC OVERLAP: Very low word overlap between definition and matched content")
    
    # Check for generic administrative content matched to specific educational activities
    administrative_terms = ["establishing", "maintaining", "secretariat", "meetings", "platform management", "coordination"]
    educational_terms = ["teach", "learn", "student", "classroom", "curriculum", "pedagog", "instruct", "train"]
    
    has_admin = any(term in matched_content.lower() for term in administrative_terms)
    has_education = any(term in matched_content.lower() for term in educational_terms)
    needs_education = any(term in definition.lower() for term in educational_terms)
    
    if has_admin and not has_education and needs_education:
        issues.append("GENERIC MISMATCH: Administrative content matched to educational activity")
    
    if issues:
        return "FLAGGED", "; ".join(issues)
    return "VALID", ""

if pdf_file and csv_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    # Show extraction method selection
    extraction_method = st.radio(
        "Choose extraction method:",
        ["Mistral AI Smart Extraction", "Traditional Pattern Matching"],
        help="Mistral AI can intelligently identify and extract relevant sections"
    )
    
    if extraction_method == "Mistral AI Smart Extraction":
        extracted_text = enhanced_text_extraction(pdf_reader)
    else:
        # Use traditional method
        raw_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        cleaned_text = clean_pdf_text(raw_text)
        
        headings = [
            "A. PDO", "A. Project Development Objective", "A. Project Development Objectives",
            "B. Project Beneficiaries", "Project Beneficiaries", "C. Project Beneficiaries",
            "A. Project Components", "B. Project Components", "Project Components",
       
            "ANNEX 1B: THEORY OF CHANGE", "ANNEX 2: DETAILED PROJECT DESCRIPTION",
            "Annex 2: Detailed Project Description", "DETAILED PROJECT DESCRIPTION",
            "D. Results Chain", "D. Theory of Change", "RESULTS CHAIN"
        ]
        
        pattern = rf"({'|'.join(re.escape(h) for h in headings)})(.*?)(?=\n(?:[IVXLCDM\d]+\.?\s+)?[A-Z][A-Z\s]+\n|Annex\s+\d+|$)"
        matches = re.findall(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            content_dict = {}
            for heading, content in matches:
                content = content.strip()
                if content:
                    content_dict[heading.strip()] = content
            
            sections_found = list(content_dict.keys())
            st.info(f"📋 Traditional method found sections: {', '.join(sections_found)}")
            extracted_text = "\n\n".join(content_dict.values())
        else:
            st.warning("No sections found with traditional method. Using entire document.")
            extracted_text = cleaned_text
    
    # Sentence-preserving chunking strategy
    if st.checkbox("Use sentence-level chunking (recommended)", value=True):
        chunks = sentence_level_chunking(extracted_text)
    else:
        # Traditional chunking with sentence-preserving separators
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
        docs = [Document(page_content=extracted_text)]
        chunks = text_splitter.split_documents(docs)
    
    # Build vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    #retriever = vectorstore.as_retriever(search_type="similarity", k=7)
    retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": sim_threshold,  # slider value from step 1
        "k": 50                             # safety cap; raise/lower as you like
    }
)

    def load_definitions(file):
        """Load definitions with robust encoding handling"""
        try:
            # First, try to detect if it's actually an Excel file
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            
            # Check for Excel file signatures
            if file_content.startswith(b'PK') or file_content.startswith(b'\xd0\xcf\x11\xe0'):
                # It's likely an Excel file
                df = pd.read_excel(file)
            else:
                # Try different encodings for CSV
                encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
                
                for encoding in encodings_to_try:
                    try:
                        file.seek(0)  # Reset file pointer
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file, encoding=encoding)
                        else:
                            df = pd.read_excel(file)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                else:
                    # If all encodings fail, try with error handling
                    file.seek(0)
                    df = pd.read_csv(file, encoding='utf-8', errors='replace')
                    
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.info("Try saving your file as UTF-8 CSV or Excel format")
            st.stop()

        # Check for required columns with flexible naming
        activity_col = None
        definition_col = None
        
        # Look for Activity Name column (exact match)
        if 'Activity Name' in df.columns:
            activity_col = 'Activity Name'
        else:
            st.error("Uploaded file must contain 'Activity Name' column.")
            st.info(f"Found columns: {list(df.columns)}")
            st.stop()
        
        # Look for Definition column (flexible - can be 'Definition' or 'Definitions')
        if 'Definition' in df.columns:
            definition_col = 'Definition'
        elif 'Definitions' in df.columns:
            definition_col = 'Definitions'
            st.info("Using 'Definitions' column (found plural form)")
        else:
            st.error("Uploaded file must contain 'Definition' or 'Definitions' column.")
            st.info(f"Found columns: {list(df.columns)}")
            st.stop()

        return df[[activity_col, definition_col]].rename(columns={
            activity_col: 'Activity Name',
            definition_col: 'Definition'
        }).dropna().reset_index(drop=True)

    definitions_df = load_definitions(csv_file)
    definitions_df = definitions_df.drop_duplicates(subset=["Activity Name", "Definition"])

    # Few-shot examples for better prompt engineering
    few_shot_examples = [
        {"Activity": "Textbooks",
        "Definition": "Textbook production and distribution",
        "Context": "To complement the focus on Grades 1 and 2 reading and math material of the PREBAT and to ensure continuity in these two subjects, the subcomponent will finance the production and distribution of grades 3 through 6 (CE and CM levels) textbooks and teacher guides in reading and math. Approximately 405,000 textbooks, 10,000 teacher guides, and 10,000 textbook management guides will be printed and distributed across all primary schools in the four targeted regions.",
        "Output": "* To complement the focus on Grades 1 and 2 reading and math material of the PREBAT and to ensure continuity in these two subjects, the subcomponent will finance the production and distribution of grades 3 through 6 (CE and CM levels) textbooks and teacher guides in reading and math.\n* Approximately 405,000 textbooks, 10,000 teacher guides, and 10,000 textbook management guides will be printed and distributed across all primary schools in the four targeted regions."

        },
        {
            "Activity": "EdTech Pillar 1 (Learning Continuity & Acceleration) - Remote-TV",
            "Definition": "Remote Learning using TV",
            "Context": "activities designed to support hybrid education, including support to home-based learning through online technology, television, radio, and print materials, as well as continued support to ongoing reforms with longer-term implications, such as the implementation of the new curriculum and management of the school system. Although several secondary schools have introduced online learning in response to the COVID-19 pandemic, there is considerable potential to further the delivery of online and blended learning. The Project is designed to be adaptable to both fully distant and hybrid models, as well as to face-to-face learning.",
            "Output": "* activities designed to support hybrid education, including support to home-based learning through online technology, television, radio, and print materials, as well as continued support to ongoing reforms with longer-term implications, such as the implementation of the new curriculum and management of the school system."
        },
        {
            "Activity": "Governing Bodies/Boards",   
            "Definition": "Support for the creation or strengthening of governing bodies for tertiary or TVET institutions",
            "Context": "This subcomponent would strengthen the role of the Council for TVET as the coordinating body of TVET provision in Guyana. The Council for TVET (CTVET), established in 2004, reports both to the CTVET Board of Governors and the MOE, and plays an important role in ensuring the quality of TVET programs and that available offerings suit Guyana's development needs. An assessment of the CTVET functions and capacity is conducted, resulting in a plan to strengthen the role of CTVET, staff development.",
            "Output": "* This subcomponent would strengthen the role of the Council for TVET as the coordinating body of TVET provision in Guyana.\n* An assessment of the CTVET functions and capacity is conducted, resulting in a plan to strengthen the role of CTVET, staff development."
        }
    ]

    # Enhanced system prompt with your specific examples as few-shot learning
    system_template = """
You are a World Bank education policy expert specializing in project activity classification.

CRITICAL INSTRUCTIONS:
1. Extract ONLY exact sentences that describe ACTUAL PROJECT ACTIVITIES with the EXACT SAME SEMANTIC MEANING as the definition
2. The matched content must perform the SAME ACTION/PURPOSE described in the definition
3. Do NOT extract sentences about related but different activities
4. Start each extracted sentence with '*'
5. Extract maximum 3 sentences
6. If no semantically matching activities are found, respond exactly: "NO RELEVANT CONTEXT FOUND"

GOOD MATCH EXAMPLES (FOLLOW THESE PATTERNS):

Example 1: ✅ CORRECT
Activity: Textbooks
Definition: "Textbook production and distribution"
Context: "To complement the focus on Grades 1 and 2 reading and math material of the PREBAT and to ensure continuity in these two subjects, the subcomponent will finance the production and distribution of grades 3 through 6 (CE and CM levels) textbooks and teacher guides in reading and math. Approximately 405,000 textbooks, 10,000 teacher guides, and 10,000 textbook management guides will be printed and distributed across all primary schools in the four targeted regions."
Correct Output: "* To complement the focus on Grades 1 and 2 reading and math material of the PREBAT and to ensure continuity in these two subjects, the subcomponent will finance the production and distribution of grades 3 through 6 (CE and CM levels) textbooks and teacher guides in reading and math.\n* Approximately 405,000 textbooks, 10,000 teacher guides, and 10,000 textbook management guides will be printed and distributed across all primary schools in the four targeted regions."
WHY CORRECT: Both definition and content describe "production and distribution" of textbooks.

Example 2: ✅ CORRECT
Activity: Occupational standards/Certification of Vocational Learning
Definition: "Creation or revision of sets of standards to evaluate and certify the competencies of graduates of vocational or technical training programs"
Context: "Enhanced TVET certification is defined and consist of i) a recognized and competency-based skills qualification (e.g. CVQ) in a selected sector, i.e. an economic sector identified as a priority through a labor market needs assessment or by MOE in collaboration with CTVET; and ii) apprenticeship or employability training if not included in the accreditation. The new TVET policy for the 2022-2032 period will identify specific, prioritized goals and targets for TVET development, including strengthening apprenticeship as a means of skills acquisition, including guidelines on internships."
Correct Output: "* Enhanced TVET certification is defined and consist of i) a recognized and competency-based skills qualification (e.g. CVQ) in a selected sector, i.e. an economic sector identified as a priority through a labor market needs assessment or by MOE in collaboration with CTVET; and ii) apprenticeship or employability training if not included in the accreditation.\n* The new TVET policy for the 2022-2032 period will identify specific, prioritized goals and targets for TVET development, including strengthening apprenticeship as a means of skills acquisition, including guidelines on internships."
WHY CORRECT: Both describe creating/defining standards and certification systems for vocational training.

Example 3: ✅ CORRECT
Activity: Governing Bodies/Boards
Definition: "Support for the creation or strengthening of governing bodies for tertiary or TVET institutions"
Context: "This subcomponent would strengthen the role of the Council for TVET as the coordinating body of TVET provision in Guyana. The Council for TVET (CTVET), established in 2004, reports both to the CTVET Board of Governors and the MOE, and plays an important role in ensuring the quality of TVET programs and that available offerings suit Guyana's development needs. An assessment of the CTVET functions and capacity is conducted, resulting in a plan to strengthen the role of CTVET, staff development."
Correct Output: "* This subcomponent would strengthen the role of the Council for TVET as the coordinating body of TVET provision in Guyana.\n* An assessment of the CTVET functions and capacity is conducted, resulting in a plan to strengthen the role of CTVET, staff development."
WHY CORRECT: Both describe strengthening governing bodies for TVET institutions.

Example 4: ✅ CORRECT
Activity: EdTech Pillar 1 (Learning Continuity & Acceleration) - Remote-TV
Definition: "Remote Learning using TV"
Context: "activities designed to support hybrid education, including support to home-based learning through online technology, television, radio, and print materials, as well as continued support to ongoing reforms with longer-term implications, such as the implementation of the new curriculum and management of the school system. Although several secondary schools have introduced online learning in response to the COVID-19 pandemic, there is considerable potential to further the delivery of online and blended learning. The Project is designed to be adaptable to both fully distant and hybrid models, as well as to face-to-face learning."
Correct Output: "* activities designed to support hybrid education, including support to home-based learning through online technology, television, radio, and print materials, as well as continued support to ongoing reforms with longer-term implications, such as the implementation of the new curriculum and management of the school system."
WHY CORRECT: Content specifically mentions "television" as part of remote learning delivery.

BAD MATCH EXAMPLES (DO NOT DO THESE):

Example 1: ❌ WRONG
Activity: EdTech Procurement
Definition: "Procurement mechanisms for devices, EdTech projects, or financing schemes"
Context: "The Project will finance acquisition of equipment to ensure the implementation of the multimodal remote learning platform."
Wrong Output: "* The Project will finance acquisition of equipment to ensure the implementation of the multimodal remote learning platform."
WHY WRONG: Definition is about "procurement mechanisms" (systems/processes) but content is about "acquisition of equipment" (buying things). Different actions.

Example 2: ❌ WRONG
Activity: Alternative education
Definition: "Introduction or piloting of basic education approaches outside the purview of formal schooling"
Context: "The Project is designed to be adaptable to both fully distant and hybrid models, as well as to face-to-face learning."
Wrong Output: "* The Project is designed to be adaptable to both fully distant and hybrid models, as well as to face-to-face learning."
WHY WRONG: Definition requires "outside formal schooling" but content describes different formats within formal schooling.

Example 3: ❌ WRONG
Activity: Literacy
Definition: "Technology to promote literacy"
Context: "Existing TVET offerings will be enhanced with supplemental modules on socioemotional skills"
Wrong Output: "* Existing TVET offerings will be enhanced with supplemental modules on socioemotional skills"
WHY WRONG: Definition requires "technology for literacy (reading/writing)" but content is about socioemotional skills, not literacy.

Example 4: ❌ WRONG
Activity: Assistive Technology
Definition: "Using technology to address special needs"
Context: "The Project will support professional development for secondary level TVET teachers and trainers, including digital and socioemotional skills."
Wrong Output: "* The Project will support professional development for secondary level TVET teachers and trainers, including digital and socioemotional skills."
WHY WRONG: Definition requires "technology for special needs/disabilities" but content is general teacher training, not assistive technology.

Example 5: ❌ WRONG
Activity: Socioeconomically disadvantaged students
Definition: "Interventions that aim to increase educational access, participation and/or learning of socioeconomically disadvantaged students"
Context: "The vulnerable groups that have been identified as more likely to be excluded from the Project benefits include indigenous and female students, teachers, students who live with disabilities, and immigrant populations attending secondary education."
Wrong Output: Should be "NO RELEVANT CONTEXT FOUND"
WHY WRONG: Content describes vulnerable groups but doesn't describe specific interventions targeting socioeconomically disadvantaged students.

Example 6: ❌ WRONG
Activity: Scholarships, stipends, or loans
Definition: "Provision of any form for cash transfer to girls or families for girls' education"
Context: "Communication campaigns related to attending school will include information on the average wage returns to secondary social services (cash transfers)"
Wrong Output: Should be "NO RELEVANT CONTEXT FOUND"
WHY WRONG: Content describes communication campaigns about cash transfers, not actual provision of cash transfers.

SEMANTIC MATCHING RULES:
1. For "production/distribution" → Must mention producing AND distributing
2. For "procurement mechanisms" → Must mention improving/developing procurement systems, NOT just buying things
3. For "outside formal schooling" → Must explicitly be non-formal/alternative education
4. For "technology to promote literacy" → Must mention technology AND basic reading/writing skills
5. For "assistive technology" → Must mention technology AND special needs/disabilities
6. For "cash transfers/scholarships" → Must mention actual provision, not just communication about them
7. For "interventions targeting disadvantaged" → Must describe specific activities, not just identification of groups

STRICT RULE: Only extract if the context describes the EXACT SAME ACTION as the definition. Related activities, descriptions of problems, or mentions without actual activities are NOT matches.

If no exact semantic matches exist, respond: "NO RELEVANT CONTEXT FOUND"
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", """
Activity Name: {activity_name}

Definition: {definition}

Context:
{context}

Extract ONLY exact sentences that describe project activities matching this definition. Start each with '*'. Maximum 3 sentences.
""")
    ])

    chain = prompt_template | model | StrOutputParser()

    results = []
    progress_bar = st.progress(0)
    
    for idx, row in definitions_df.iterrows():
        progress_bar.progress((idx + 1) / len(definitions_df))
        
        definition = row['Definition']
        activity_name = row['Activity Name']
        
        # Use Activity Name + Definition for vectorstore retrieval
        query_text = f"{activity_name}: {definition}"
        docs = retriever.get_relevant_documents(query_text)
        context = "\n".join([doc.page_content for doc in docs])
        faiss_context_for_llm = context.strip()
        docs_with_scores = vectorstore.similarity_search_with_score(query_text, k=50)
        score_lookup = {doc.page_content.strip(): 1 - score for doc, score in docs_with_scores}
        faiss_scores_only = ";  ".join([
            f"{1 - dist:.4f}"
            
            for doc, dist in docs_with_scores
            ]) or "NO MATCHES ABOVE THRESHOLD"


        if not context.strip():
            matched_content = "NO RELEVANT CONTEXT FOUND"
            similarity_score = 0.0
            validation_status = "VALID"
            validation_notes = ""
        else:
            try:
                output = chain.invoke({
                    "activity_name": activity_name,
                    "context": context, 
                    "definition": definition
                })
                matched_content = output.strip()
            except Exception as e:
                st.warning(f"Model failed for {activity_name}: {e}")
                matched_content = "MODEL ERROR"
            
            # Use OpenAI embeddings for similarity calculation
            similarity_score = embedding_similarity_score(definition, activity_name, matched_content)
            
            # Validate the extraction
            validation_status, validation_notes = validate_extraction(activity_name, definition, matched_content)
        
        # Add confidence flag
        confidence_flag = "HIGH" if similarity_score >= confidence_threshold else "LOW"
        
        results.append((
            activity_name, 
            definition,
            faiss_context_for_llm, 
            faiss_scores_only,
            matched_content, 
            similarity_score,
            confidence_flag,
            validation_status,
            validation_notes
        ))

    final_df = pd.DataFrame(results, columns=[
        "Activity Name", 
        "Definition",
        "Context Sent to LLM" , 
        "FAISS Similarity Scores" ,
        "Matched Content", 
        "Similarity Score (%)",
        "Confidence",
        "Validation Status",
        "Validation Notes"
    ])
    
    # Sort by similarity score in descending order
    final_df = final_df.sort_values("Similarity Score (%)", ascending=False).reset_index(drop=True)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Activities", len(final_df))
    with col2:
        high_conf = len(final_df[final_df["Confidence"] == "HIGH"])
        st.metric("High Confidence", high_conf)
    with col3:
        flagged = len(final_df[final_df["Validation Status"] == "FLAGGED"])
        st.metric("Flagged for Review", flagged)
    with col4:
        with_content = len(final_df[final_df["Matched Content"] != "NO RELEVANT CONTEXT FOUND"])
        st.metric("With Matched Content", with_content)
    
    # Filter options
    st.subheader("Filter Results")
    filter_confidence = st.selectbox("Filter by Confidence", ["All", "HIGH", "LOW"])
    filter_validation = st.selectbox("Filter by Validation", ["All", "VALID", "FLAGGED"])
    
    # Apply filters
    filtered_df = final_df.copy()
    if filter_confidence != "All":
        filtered_df = filtered_df[filtered_df["Confidence"] == filter_confidence]
    if filter_validation != "All":
        filtered_df = filtered_df[filtered_df["Validation Status"] == filter_validation]
    
    # Color-code the dataframe
    def highlight_rows(row):
        if row["Confidence"] == "LOW":
            return ['background-color: #ffeeee'] * len(row)
        elif row["Validation Status"] == "FLAGGED":
            return ['background-color: #fff3cd'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = filtered_df.style.apply(highlight_rows, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        csv = final_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Full Results", csv, "enhanced_matched_output.csv", "text/csv")
    
    with col2:
        flagged_df = final_df[final_df["Validation Status"] == "FLAGGED"]
        if not flagged_df.empty:
            flagged_csv = flagged_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Flagged Items", flagged_csv, "flagged_for_review.csv", "text/csv")
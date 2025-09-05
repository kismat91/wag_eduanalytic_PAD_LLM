# WBG EduAnalytic PAD LLM

An AI-powered application for analyzing World Bank Project Appraisal Documents (PADs) with enhanced semantic matching and LLM validation capabilities.

## Overview

WBG EduAnalytic PAD LLM is a comprehensive solution that leverages advanced AI technologies to analyze World Bank Project Appraisal Documents. The application combines document OCR, semantic search, LLM validation, and interactive chat capabilities to provide deep insights and semantic matching for educational project analysis.

## Key Features

### PAD Document Semantic Matcher (app.py)
- **Enhanced OCR**: Mistral OCR with PyPDF2 fallback for high-quality text extraction
- **Intelligent Section Extraction**: Automatically extracts target sections (PDO, Project Components, etc.)
- **Multi-Layer Semantic Matching**: 
  - OpenAI embeddings with cosine similarity
  - Rule-based semantic validation
  - LLM contextual validation with GPT-4o-mini
- **Progressive Filtering**: 3-stage filtering system for optimal accuracy
- **Multiple AI Models**: Support for OpenAI GPT-4o-mini, DeepSeek LLaMA-70B, Mistral Large
- **Advanced Chunking**: Sentences, paragraphs, or fixed-size chunks
- **Comprehensive Scoring**: Combined embedding similarity (70%) + semantic analysis (30%)

### Web Application Features
- **Document Preview**: Access Mistral OCR for high-quality text extraction from PDF documents
- **Chat with PDF**: Ask questions about your documents and receive contextually relevant answers
- **Generate Results**: Create comprehensive reports and insights based on document content
- **Analytics Dashboard**: Track usage metrics, costs, and performance across different models
- **Multi-Model Support**: Choose from various AI models including OpenAI GPT and open-source alternatives
- **Advanced RAG Pipeline**: Retrieval-Augmented Generation for accurate document-based responses

## Project Architecture

### Enhanced PAD Semantic Matcher
- **Streamlit Interface**: User-friendly web interface for document analysis
- **Multi-Modal AI Processing**: 
  - OpenAI text-embedding-3-large for semantic embeddings
  - GPT-4o-mini for contextual validation
  - DeepSeek LLaMA-70B and Mistral Large as alternatives
- **Advanced Filtering Pipeline**:
  1. Keyword-based pre-filtering
  2. High-similarity threshold filtering (default: 0.3)
  3. Top 20% selection for LLM validation
  4. Contextual meaning verification

### Frontend
- React with TypeScript
- Tailwind CSS for responsive UI
- Theme support (Light, Dark, and Futuristic modes)

### Backend
- FastAPI (Python) for efficient API endpoints
- Vector database (FAISS) for semantic document search
- Multiple LLM integrations via API connectors
- Analytics tracking and performance monitoring

## Project Structure

```
/
├── api/                      # FastAPI backend
│   ├── __init__.py
│   ├── .env                  # Environment variables
│   ├── analytics_api.py      # Analytics endpoints
│   ├── analytics_data.json   # Usage data storage
│   ├── chat_api.py           # Chat functionality
│   ├── generation_api.py     # Content generation
│   ├── main.py               # Application entry point
│   └── process_pdf.py        # PDF processing pipeline
│
├── src/                      # Frontend React application
│   ├── components/           # UI components
│   ├── services/             # API service connectors
│   ├── App.tsx               # Main application
│   └── main.tsx              # Entry point
│
├── dist/                     # Built frontend assets
└── node_modules/             # Frontend dependencies
```

## Setup Instructions

### Prerequisites

- Node.js (v16+)
- Python (v3.8+)
- API keys for AI models:
  - OpenAI API key (required)
  - Mistral API key (optional, for OCR)
  - Groq API key (optional, for DeepSeek models)
  - HuggingFace API key (optional)

### PAD Document Semantic Matcher Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/kismat91/wag_eduanalytic_PAD_LLM.git
   cd wag_eduanalytic_PAD_LLM
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create `.env` file from template:
   ```bash
   cp .env.example .env
   ```

5. Edit `.env` file with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   MISTRAL_API_KEY=your_mistral_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   ```

6. Run the PAD Document Semantic Matcher:
   ```bash
   streamlit run app.py
   ```

### Web Application Setup

1. Navigate to the API directory:
   ```bash
   cd api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create `.env` file with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   ```

5. Start the backend server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

6. Frontend Setup:
   ```bash
   npm install
   npm run dev
   ```

The web application will be available at http://localhost:5173
The PAD Semantic Matcher will be available at http://localhost:8501

## Usage Guide

### PAD Document Semantic Matcher

1. **Upload Files**:
   - Upload your PAD PDF document
   - Upload activities CSV/Excel file with 'Activity Name' and 'Definition' columns

2. **Configure Analysis**:
   - Choose AI model (OpenAI GPT-4o-mini, DeepSeek LLaMA-70B, or Mistral Large)
   - Select chunking method (Sentences, Paragraphs, or Fixed size)
   - Enable semantic validation for enhanced accuracy
   - Enable LLM validation for contextual meaning verification

3. **Review Results**:
   - View extracted sections from PAD document
   - Analyze similarity scores and semantic matches
   - Review LLM validation results
   - Download complete, filtered, or LLM-matched results

### Web Application

1. **Analyzing Documents**:
   - Upload your World Bank PDF document
   - View the extracted text and structure
   - Use the search function to find specific information

2. **Chatting with Documents**:
   - Upload a PDF or select from previous uploads
   - Ask questions about the document content
   - Select your preferred AI model
   - Receive contextually relevant answers

3. **Generating Content**:
   - Choose content generation options
   - Select model and parameters
   - Generate reports, summaries, or insights

4. **Analytics Dashboard**:
   - Track token usage, costs, and performance metrics
   - Compare different models and features
   - Filter by time period

## API Endpoints

### Document Processing
- `POST /api/process-pdf`: Process a PDF document
- `GET /api/search-pdf`: Search within the document

### Chat
- `POST /api/chat`: Chat with PDF context

### Generation
- `POST /api/generate`: Generate content based on document context

### Analytics
- `POST /api/track-usage`: Track API usage metrics
- `GET /api/analytics`: Get analytics data

## Technical Features

### PAD Document Semantic Matcher
- **Target Section Extraction**: Automatically finds and extracts:
  - A. PDO (Project Development Objective)
  - Project Beneficiaries
  - Project Components
  - Annex 2: Detailed Project Description
  - Results Chain / Theory of Change

- **Multi-Layer Scoring Algorithm**:
  ```
  FAISS_Similarity = 0.7 × cosine_similarity + 0.3 × (semantic_score × 2 - 1)
  ```

- **Progressive Filtering System**:
  1. Initial threshold filtering (default: 0.3)
  2. Top 20% selection for LLM validation
  3. Contextual meaning verification

- **LLM Validation Features**:
  - Structured decision format: "CONTEXTUAL MATCH" or "NO CONTEXTUAL MATCH"
  - Detailed reasoning for each decision
  - Support for multiple LLM models
  - Error handling and fallback mechanisms

### File Format Support
- **PDF Input**: PAD documents with OCR support
- **Activities Input**: CSV/Excel files with required columns:
  - `Activity Name`: Name of the activity
  - `Definition`: Description of the activity

### Output Formats
- **Complete Results**: All computed similarity pairs
- **Filtered Results**: Results above similarity threshold
- **LLM Matched Only**: Only contextually validated matches

## Performance Optimization
- **Vectorized Computation**: Batch processing of embeddings
- **Aggressive Pre-filtering**: Reduces computation time by 80%
- **Caching**: Streamlit caching for PDF processing
- **Progress Tracking**: Real-time progress indicators

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- World Bank for project structure inspiration
- OpenAI for embedding and LLM capabilities
- Mistral AI for OCR functionality
- Streamlit for the user interface framework

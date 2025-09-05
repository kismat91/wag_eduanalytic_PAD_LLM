import React, { useState, useEffect } from 'react';
import { ChevronDown, FilePlus, FileSpreadsheet, FileText, Download, ThumbsUp, ThumbsDown } from 'lucide-react';
import Button from '../ui/Button';
import Card from '../ui/Card';
import SingleEntryForm from './SingleEntryForm';
import BulkGenerationForm from './BulkGenerationForm';
import GeneratedContent from './GeneratedContent';
import GenerationModelSelector from './GenerationModelSelector';
import { useTheme } from '../ThemeContext';
import axios from 'axios';

type GenerationMode = 'single' | 'bulk';

const API_BASE_URL = 'http://localhost:8002';
// Predefined prompts for the dropdown
const PREDEFINED_PROMPTS = [
  {
    id: 'world-bank-expert',
    name: 'World Bank Expert Mode',
    content: `### Instructions:
You are an expert in World Bank Global Education and education policy analysis. Your task is to determine if the activity name and definition provided in the query align with relevant content in the given context.

### Task:
- Extract up to 3 sentences from the provided context that semantically align with the given activity name and definition.
- Start each sentence with a '*' character.
- If no relevant content exists, respond with: "NO RELEVANT CONTEXT FOUND".
- Do not generate new sentences, rephrase, summarize, or add external information.
- Do not infer meaning beyond what is explicitly stated in the context.
- Not every definition may have meaningful content; in such cases, return "NO RELEVANT CONTEXT FOUND".

### Query:
Activity Name and Definition: {query}

### Context:
{context_text}

### Response Format:
- If relevant sentences are found:
  * Sentence 1 from context
  * Sentence 2 from context
  * Sentence 3 from context (if applicable)
- If no relevant content is found:
  NO RELEVANT CONTEXT FOUND

### Strict Guidelines:
- Only extract sentences exactly as they appear in the provided context.
- Do not provide reasons, explanations, or additional commentary.
- Do not summarize, reword, or infer additional meaning beyond the explicit text.
- Ensure strict semantic alignment between the definition and the extracted sentences.`
  },
  {
    id: 'edu-mode',
    name: 'World Bank Education',
    content: "World-Bank-Edu expert mode → Retrieve ≤ 3 exact sentences that semantically match the 'Activity Name & Definition'; prefix each with '-'. If no match, output exactly NO RELEVANT CONTEXT FOUND. No paraphrase, no inference."
  },
  {
    id: 'verbatim-lines',
    name: 'Verbatim Extraction',
    content: "Task: locate up to three verbatim lines in the supplied context that align with this activity description → {query}. Return each line as - sentence. If nothing aligns, return NO RELEVANT CONTEXT FOUND (all-caps). Do not alter wording or add commentary."
  },
  {
    id: 'policy-analyst',
    name: 'Policy Analyst',
    content: "Instruction (Edu-policy analyst): Scan {context_text}. Copy at most 3 sentences whose meaning overlaps the following activity definition. Bullet them with '-'. If zero overlap, output NO RELEVANT CONTEXT FOUND. Absolutely no re-phrasing or summarizing."
  }
];

const GenerationSection: React.FC = () => {
  // Get theme context
  const { currentTheme, getThemeClasses } = useTheme();
  const theme = getThemeClasses();
  
  const [mode, setMode] = useState<GenerationMode>('single');
  const [generatedContent, setGeneratedContent] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gpt-4.5');
  const [selectedPrompt, setSelectedPrompt] = useState(PREDEFINED_PROMPTS[0]);
  const [customPrompt, setCustomPrompt] = useState('');
  const [isPromptDropdownOpen, setIsPromptDropdownOpen] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<string | null>(null);
  const [previewPrompt, setPreviewPrompt] = useState<string | null>(null);
  const [downloadData, setDownloadData] = useState<any>(null);
  const [availableHeadings, setAvailableHeadings] = useState<string[]>([]);
  const [usePyPDF2, setUsePyPDF2] = useState(false); // Toggle for processing method
  const [pdfFileSize, setPdfFileSize] = useState<number | null>(null); // Track PDF file size for warnings
  const [pdfFile, setPdfFile] = useState<File | null>(null); // Store PDF file for reprocessing

  const handleSingleGeneration = async (
    activity: string, 
    definition: string, 
    file: File | null, 
    modelId: string, 
    analysisMode: 'full_text' | 'target_headings_only'
  ) => {
    if (!file) {
      setGeneratedContent('Error: Please upload a PDF file for context.');
      return;
    }

    // Clear the preview prompt when generating content
    setPreviewPrompt(null);
    
    setIsGenerating(true);
    setSelectedModel(modelId); // Update the selected model state
    setProcessingStatus('Processing PDF...');

    try {
      let pdfContent = null;
      
      // Upload PDF if provided
      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/extract-sections', {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          throw new Error('PDF upload failed');
        }
        
        const result = await response.json();
        pdfContent = result.content;
        setAvailableHeadings(result.available_headings || []);
      }

      setProcessingStatus('Generating content...');
      
      // Generate content using the selected mode
      const generationResponse = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          activity,
          definition,
          pdf_content: pdfContent,
          model_id: modelId,
          analysis_mode: analysisMode
        }),
      });

      if (!generationResponse.ok) {
        throw new Error(`Generation failed: ${generationResponse.status}`);
      }

      const generationResult = await generationResponse.json();
      setGeneratedContent(generationResult.generated_content);
      setDownloadData(generationResult);
    } catch (error: any) {
      console.error('Error generating content:', error);
      setGeneratedContent('Error generating content. Please try again.');
    } finally {
      setIsGenerating(false);
      setProcessingStatus(null);
    }
  };

  const handleBulkGeneration = async (file: File, pdfFile: File, queryLimit: number, modelId: string, analysisMode: 'full_text' | 'target_headings_only') => {
    if (!file || !pdfFile) {
      alert('Please upload both Excel and PDF files');
      return;
    }

    setIsGenerating(true);
    setProcessingStatus('Starting bulk content generation...');
    setGeneratedContent(null);
    setDownloadData(null);

    try {
      let pdfContent = null;
      
      // Upload PDF if provided
      if (pdfFile) {
        const formData = new FormData();
        formData.append('file', pdfFile);
        
        setProcessingStatus('Extracting text from PDF...');
        
        // Choose endpoint based on user selection
        const pdfEndpoint = usePyPDF2 ? '/api/process-pdf-pypdf2' : '/api/process-pdf';
        const response = await axios.post(`${API_BASE_URL}${pdfEndpoint}`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: usePyPDF2 ? 120000 : 300000 // 2 minutes for PyPDF2, 5 minutes for OCR
        });
        
        if (response.data.structured_pages) {
          // Create plain text representation
          pdfContent = response.data.structured_pages
            .map((page: any) => `${page.plain_text}`)
            .join('\n\n');
          
          setProcessingStatus(`PDF processed successfully. Extracted ${response.data.structured_pages.length} pages.`);
          
          // Show note if provided
          if (response.data.note) {
            console.log(response.data.note);
            // Add a user-friendly suggestion for large PDFs
            if (pdfFile && pdfFile.size > 10 * 1024 * 1024) { // > 10MB
              setGeneratedContent(`PDF processed successfully! Note: For large PDFs like yours (${(pdfFile.size / (1024 * 1024)).toFixed(1)} MB), consider using the PyPDF2 endpoint for faster processing in the future.`);
            }
          }
        } else {
          throw new Error('PDF processing failed: No structured pages returned');
        }
      }

      if (!pdfContent) {
        setIsGenerating(false);
        setProcessingStatus(null);
        setGeneratedContent('Error: Failed to extract text from PDF. Please try again with a different file.');
        return;
      }
      
      // Get the prompt to use (either selected predefined or custom)
      const promptToUse = customPrompt.trim() ? customPrompt : selectedPrompt.content;
      
      // Upload Excel file for bulk processing
      setProcessingStatus(`Processing ${queryLimit === 0 ? 'all' : queryLimit} entries through RAG pipeline...`);
      const formData = new FormData();
      formData.append('file', file);
      formData.append('query_limit', queryLimit.toString());
      
      if (pdfContent) {
        formData.append('pdf_content', pdfContent);
      }
      
      formData.append('model', modelId);
      formData.append('prompt', promptToUse);
      formData.append('analysis_mode', analysisMode);
      
      setProcessingStatus('Starting bulk content generation...');
      const response = await axios.post(`${API_BASE_URL}/api/generate-bulk`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 600000 // 10 minutes timeout for bulk processing
      });
      
      setGeneratedContent(response.data.content || `Processed ${file.name} successfully. Generated content for ${queryLimit === 0 ? 'all' : queryLimit} entries.`);
      // Store data for CSV download
      setDownloadData(response.data);
    } catch (error: any) {
      console.error('Error processing bulk generation:', error);
      
      // Get more specific error information
      let errorMessage = `Error: Failed to process ${file.name}. `;
      
      if (error.response) {
        // Server responded with error status
        if (error.response.data && error.response.data.detail) {
          errorMessage += error.response.data.detail;
        } else {
          errorMessage += `Server error: ${error.response.status} - ${error.response.statusText}`;
        }
      } else if (error.request) {
        // Request was made but no response received
        errorMessage += "No response from server. Please check your connection.";
      } else {
        // Something else happened
        errorMessage += error.message || "Unknown error occurred.";
      }
      
      setGeneratedContent(errorMessage);
    } finally {
      setIsGenerating(false);
      setProcessingStatus(null);
    }
  };

  const handleModeChange = (newMode: GenerationMode) => {
    setMode(newMode);
    setGeneratedContent(null);
    setPreviewPrompt(null);
  };

  const handleSelectPrompt = (prompt: typeof PREDEFINED_PROMPTS[0]) => {
    setSelectedPrompt(prompt);
    setIsPromptDropdownOpen(false);
  };

  const handlePreviewPrompt = () => {
    // Show the prompt in the preview area (sidebar), not in the main content area
    const promptToUse = customPrompt.trim() ? customPrompt : selectedPrompt.content;
    setPreviewPrompt(promptToUse);
  };

  const handleDownload = () => {
    if (!generatedContent) return;
    
    // Create a content type and format for download
    let contentType = 'text/html';
    let fileExtension = '.html';
    let content = generatedContent;
    
    // For CSV downloads in bulk mode
    if (mode === 'bulk' && downloadData && downloadData.results) {
      // Convert results to CSV
      const csvContent = convertResultsToCSV(downloadData.results);
      content = csvContent;
      contentType = 'text/csv';
      fileExtension = '.csv';
    }
    
    // Create a Blob with the content
    const blob = new Blob([content], { type: contentType });
    
    // Create a URL for the Blob
    const url = URL.createObjectURL(blob);
    
    // Create a temporary anchor element
    const a = document.createElement('a');
    a.href = url;
    
    // Set the filename based on the mode
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    a.download = mode === 'single' 
      ? `activity-match-${timestamp}${fileExtension}` 
      : `bulk-matches-${timestamp}${fileExtension}`;
    
    // Append the anchor to the body
    document.body.appendChild(a);
    
    // Trigger a click on the anchor
    a.click();
    
    // Clean up
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  // Helper function to convert results to CSV
  const convertResultsToCSV = (results: any[]) => {
    if (!results || results.length === 0) return '';
    
    // Get headers from the first result
    const headers = Object.keys(results[0]).filter(key => key !== 'status');
    
    // Create CSV header row
    let csv = headers.join(',') + '\n';
    
    // Add each row of data
    results.forEach(result => {
      const row = headers.map(header => {
        // Escape commas and quotes in the content
        let value = result[header] || '';
        if (typeof value === 'string') {
          value = value.replace(/"/g, '""');
          // If the value contains commas, quotes or newlines, wrap it in quotes
          if (value.includes(',') || value.includes('"') || value.includes('\n')) {
            value = `"${value}"`;
          }
        }
        return value;
      }).join(',');
      
      csv += row + '\n';
    });
    
    return csv;
  };

  // Fix for ModelSelector to prevent auto-generation
  const handleModelChange = (modelId: string) => {
    setSelectedModel(modelId);
    // Don't trigger generation here
  };

  const handlePdfFileChange = async (e: React.ChangeEvent<HTMLInputElement> | File) => {
    let file: File;
    
    if (e instanceof File) {
      // Called from BulkGenerationForm with a File object
      file = e;
    } else {
      // Called from file input change event
      if (e.target && e.target.files && e.target.files.length > 0) {
        file = e.target.files[0];
      } else {
        return; // No file selected
      }
    }
    
    setPdfFileSize(file.size);
    setPdfFile(file); // Store the file for potential reprocessing
    
    // Auto-select PyPDF2 for large PDFs (>10MB) or when Target Headings Only mode is selected
    if (file.size > 10 * 1024 * 1024) {
      setUsePyPDF2(true);
    }
    
    // Immediately process the PDF to extract headings
    setProcessingStatus('Processing PDF to extract headings...');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      // Use PyPDF2 for better heading preservation
      const response = await axios.post(`${API_BASE_URL}/api/process-pdf-pypdf2`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 120000 // 2 minutes for PyPDF2
      });
      
      if (response.data.structured_pages) {
        // Update the available headings if they exist
        if (response.data.available_headings) {
          setAvailableHeadings(response.data.available_headings);
          console.log('Extracted headings from PDF:', response.data.available_headings);
        }
        
        setProcessingStatus('PDF processed successfully. Headings extracted.');
        
        // Clear the status after a few seconds
        setTimeout(() => setProcessingStatus(null), 3000);
      }
    } catch (error) {
      console.error('Error processing PDF for headings:', error);
      setProcessingStatus('Warning: Could not extract headings from PDF. Will use full text analysis.');
      setTimeout(() => setProcessingStatus(null), 3000);
    }
  };

  // Function to automatically set PyPDF2 when Target Headings Only mode is selected
  const handleAnalysisModeChange = async (mode: 'full_text' | 'target_headings_only') => {
    console.log('Analysis mode changed to:', mode);
    console.log('Current pdfFile:', pdfFile);
    console.log('Current usePyPDF2:', usePyPDF2);
    
    if (mode === 'target_headings_only') {
      console.log('Setting usePyPDF2 to true for Target Headings Only mode');
      // Automatically use PyPDF2 for better heading preservation in Target Headings Only mode
      setUsePyPDF2(true);
      
      // If we have a PDF file, reprocess it with PyPDF2 for better heading detection
      if (pdfFile) {
        console.log('Reprocessing PDF with PyPDF2...');
        setProcessingStatus('Reprocessing PDF with PyPDF2 for better heading detection...');
        
        try {
          const formData = new FormData();
          formData.append('file', pdfFile);
          
          console.log('Calling PyPDF2 endpoint...');
          // Use PyPDF2 endpoint for better heading preservation
          const response = await axios.post(`${API_BASE_URL}/api/process-pdf-pypdf2`, formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            timeout: 120000 // 2 minutes for PyPDF2
          });
          
          console.log('PyPDF2 response:', response.data);
          
          if (response.data.structured_pages) {
            // Create plain text representation
            const reprocessedPdfContent = response.data.structured_pages
              .map((page: any) => `${page.plain_text}`)
              .join('\n\n');
            
            console.log('Reprocessed PDF content length:', reprocessedPdfContent.length);
            console.log('Available headings from PyPDF2:', response.data.available_headings);
            
            // Update the available headings if they exist
            if (response.data.available_headings) {
              setAvailableHeadings(response.data.available_headings);
              console.log('Updated available headings:', response.data.available_headings);
            }
            
            setProcessingStatus('PDF reprocessed successfully with PyPDF2. Headings should now be detected correctly.');
            
            // Clear the status after a few seconds
            setTimeout(() => setProcessingStatus(null), 3000);
          }
        } catch (error) {
          console.error('Error reprocessing PDF with PyPDF2:', error);
          setProcessingStatus('Warning: Could not reprocess PDF with PyPDF2. Using existing content.');
          setTimeout(() => setProcessingStatus(null), 3000);
        }
      } else {
        console.log('No PDF file available for reprocessing');
      }
    }
  };

  return (
    <div className={`w-full h-full flex ${theme.background}`}>
      {/* Main content area */}
      <div className={`flex-1 p-6 overflow-y-auto ${theme.background}`}>
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <h1 className={`text-3xl font-bold mb-2 ${theme.title}`}>
              Content Extraction
            </h1>
            <p className={theme.secondaryText}>
              Extract context-matched content based on your input and uploaded PDFs
            </p>
          </div>

          <div className="mb-6">
            <div className={`flex space-x-2 border-b ${theme.border}`}>
              <button
                className={`px-4 py-2 font-medium text-sm border-b-2 ${
                  mode === 'single'
                    ? `border-amber-500 ${currentTheme === 'futuristic' ? 'text-blue-300' : currentTheme === 'dark' ? 'text-amber-300' : 'text-amber-600'}`
                    : `border-transparent ${theme.secondaryText} hover:${theme.text} hover:border-gray-300`
                }`}
                onClick={() => handleModeChange('single')}
              >
                Single Entry
              </button>
              <button
                className={`px-4 py-2 font-medium text-sm border-b-2 ${
                  mode === 'bulk'
                    ? `border-amber-500 ${currentTheme === 'futuristic' ? 'text-blue-300' : currentTheme === 'dark' ? 'text-amber-300' : 'text-amber-600'}`
                    : `border-transparent ${theme.secondaryText} hover:${theme.text} hover:border-gray-300`
                }`}
                onClick={() => handleModeChange('bulk')}
              >
                Bulk Generation
              </button>
            </div>
          </div>

          {processingStatus && (
            <div className={`mb-4 p-3 ${
              currentTheme === 'futuristic' 
                ? 'bg-blue-900/30 border border-blue-500/30' 
                : currentTheme === 'dark'
                  ? 'bg-blue-900/30 border border-blue-700/50'
                  : 'bg-blue-50 border border-blue-200'
            } rounded-md`}>
              <p className={`${
                currentTheme === 'futuristic' 
                  ? 'text-blue-300' 
                  : currentTheme === 'dark'
                    ? 'text-blue-300'
                    : 'text-blue-700'
              } flex items-center`}>
                <svg className="animate-spin h-4 w-4 mr-2 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                {processingStatus}
              </p>
            </div>
          )}

          {/* Modified layout structure to place the form above and content below */}
          {mode === 'single' ? (
            // Single mode layout - keeps the side-by-side layout
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <div className={`${theme.card} p-6 rounded-lg ${
                  currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow'
                }`}>
                  <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                    Generate Single Entry
                  </h2>
                  <SingleEntryForm 
                    onGenerate={handleSingleGeneration} 
                    availableHeadings={availableHeadings}
                  />
                </div>
              </div>

              <div>
                {generatedContent ? (
                  <div>
                    <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                      Extracted Content
                    </h2>
                    <GeneratedContent 
                      content={generatedContent} 
                      mode={mode} 
                      onDownload={handleDownload}
                      theme={theme}
                    />
                  </div>
                ) : (
                  <div>
                    <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                      Extracted Content
                    </h2>
                    <div className={`h-full flex items-center justify-center ${theme.cardHighlight} border ${theme.border} border-dashed rounded-lg p-8`}>
                      <div className="text-center">
                        <div className={`mx-auto h-16 w-16 rounded-full ${
                          currentTheme === 'futuristic' 
                            ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20' 
                            : currentTheme === 'dark'
                              ? 'bg-amber-800/20'
                              : 'bg-amber-100'
                        } flex items-center justify-center mb-4`}>
                          <FilePlus className={`h-8 w-8 ${
                            currentTheme === 'futuristic' 
                              ? 'text-blue-300' 
                              : currentTheme === 'dark'
                                ? 'text-amber-300'
                                : 'text-amber-600'
                          }`} />
                        </div>
                        <h3 className={`text-lg font-medium mb-1 ${theme.text}`}>No Content Generated Yet</h3>
                        <p className={`text-sm ${theme.secondaryText} max-w-xs mx-auto`}>
                          Fill in the activity details and generate to see content here
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            // Bulk mode layout - Form above, generated content below
            <div className="flex flex-col gap-8">
              <div>
                <div className={`${theme.card} p-6 rounded-lg ${
                  currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow'
                }`}>
                  <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                    Bulk Generation
                  </h2>
                  
                  {/* Processing Method Toggle */}
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Processing Method
                    </label>
                    <div className="flex items-center space-x-4">
                      <label className="flex items-center">
                        <input
                          type="radio"
                          name="processingMethod"
                          value="ocr"
                          checked={!usePyPDF2}
                          onChange={() => setUsePyPDF2(false)}
                          className="mr-2"
                        />
                        <span className="text-sm">OCR (Better quality, slower)</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="radio"
                          name="processingMethod"
                          value="pypdf2"
                          checked={usePyPDF2}
                          onChange={() => setUsePyPDF2(true)}
                          className="mr-2"
                        />
                        <span className="text-sm">PyPDF2 (Faster, good for large files)</span>
                      </label>
                    </div>
                    {pdfFileSize && pdfFileSize > 10 * 1024 * 1024 && (
                      <p className="mt-1 text-xs text-amber-600">
                        ⚠️ Large PDF detected ({(pdfFileSize / (1024 * 1024)).toFixed(1)} MB). PyPDF2 recommended for better performance.
                      </p>
                    )}
                    {usePyPDF2 && (
                      <p className="mt-1 text-xs text-blue-600">
                        ℹ️ PyPDF2 selected for better heading preservation in Target Headings Only mode.
                      </p>
                    )}
                  </div>
                  
                  <BulkGenerationForm 
                    onGenerate={handleBulkGeneration} 
                    onPdfFileChange={handlePdfFileChange}
                    onAnalysisModeChange={handleAnalysisModeChange}
                  />
                </div>
              </div>

              {generatedContent ? (
                <div className="w-full">
                  <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                    Extracted Content
                  </h2>
                  <GeneratedContent 
                    content={generatedContent} 
                    mode={mode} 
                    onDownload={handleDownload}
                    theme={theme}
                  />
                </div>
              ) : (
                <div>
                  <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                    Extracted Content
                  </h2>
                  <div className={`h-full flex items-center justify-center ${theme.cardHighlight} border ${theme.border} border-dashed rounded-lg p-8`}>
                    <div className="text-center">
                      <div className={`mx-auto h-16 w-16 rounded-full ${
                        currentTheme === 'futuristic' 
                          ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20' 
                          : currentTheme === 'dark'
                            ? 'bg-amber-800/20'
                            : 'bg-amber-100'
                      } flex items-center justify-center mb-4`}>
                        <FileSpreadsheet className={`h-8 w-8 ${
                          currentTheme === 'futuristic' 
                            ? 'text-blue-300' 
                            : currentTheme === 'dark'
                              ? 'text-amber-300'
                              : 'text-amber-600'
                        }`} />
                      </div>
                      <h3 className={`text-lg font-medium mb-1 ${theme.text}`}>No Content Extracted Yet</h3>
                      <p className={`text-sm ${theme.secondaryText} max-w-xs mx-auto`}>
                        Upload your Excel file to generate bulk content
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Right sidebar for prompts - with improved spacing */}
      <div className={`hidden lg:block w-80 border-l ${theme.border} ${theme.card} p-4 overflow-y-auto`}>
        {/* Added top padding for better spacing */}
        <div className="pt-12">
          <h3 className={`text-lg font-semibold mb-4 ${theme.title}`}>Generation Prompts</h3>
          
          {/* Predefined Prompts Dropdown */}
          <div className="mb-5">
            <h4 className={`text-sm font-medium ${theme.text} mb-2`}>Select a Predefined Prompt</h4>
            <div className="relative">
              <button
                onClick={() => setIsPromptDropdownOpen(!isPromptDropdownOpen)}
                className={`flex items-center justify-between w-full px-3 py-2 border ${theme.border} rounded-md ${
                  currentTheme === 'futuristic'
                    ? 'focus:ring-blue-500 focus:border-blue-500'
                    : currentTheme === 'dark'
                      ? 'focus:ring-amber-500 focus:border-amber-500'
                      : 'focus:ring-amber-500 focus:border-amber-500'
                } ${theme.cardHighlight} text-sm ${theme.text}`}
              >
                <span>{selectedPrompt.name}</span>
                <ChevronDown className={`h-4 w-4 ${theme.secondaryText}`} />
              </button>
              
              {isPromptDropdownOpen && (
                <div className={`absolute z-10 mt-1 w-full ${theme.card} shadow-lg rounded-md ring-1 ring-black ring-opacity-5`}>
                  <div className="py-1">
                    {PREDEFINED_PROMPTS.map((prompt) => (
                      <button
                        key={prompt.id}
                        onClick={() => handleSelectPrompt(prompt)}
                        className={`block w-full text-left px-4 py-2 text-sm ${
                          selectedPrompt.id === prompt.id
                            ? currentTheme === 'futuristic' 
                              ? 'bg-blue-900/30 text-blue-300' 
                              : currentTheme === 'dark'
                                ? 'bg-amber-900/50 text-amber-300'
                                : 'bg-amber-50 text-amber-600'
                            : `${theme.text} hover:${theme.cardHighlight}`
                        }`}
                      >
                        {prompt.name}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {/* Custom Prompt Input */}
          <div className="mb-5">
            <h4 className={`text-sm font-medium ${theme.text} mb-2`}>Or Use a Custom Prompt</h4>
            <textarea
              placeholder="Enter a custom prompt (will override selected prompt above)"
              className={`w-full px-3 py-2 border ${theme.border} rounded-md ${
                currentTheme === 'futuristic'
                  ? 'focus:ring-blue-500 focus:border-blue-500'
                  : 'focus:ring-amber-500 focus:border-amber-500'
              } ${theme.cardHighlight} text-sm ${theme.text} placeholder-${
                currentTheme === 'futuristic'
                  ? 'blue-400/50'
                  : currentTheme === 'dark'
                    ? 'gray-500'
                    : 'gray-400'
              }`}
              rows={6}
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
            />
            <p className={`mt-1 text-xs ${theme.secondaryText}`}>
              Use {'{query}'} for activity name & definition and {'{context_text}'} for PDF content
            </p>
          </div>
          
          {/* RAG Pipeline Information */}
          <div className={`mb-5 p-3 ${
            currentTheme === 'futuristic' 
              ? 'bg-blue-900/30 border border-blue-500/30' 
              : currentTheme === 'dark'
                ? 'bg-amber-900/30 border border-amber-700/30'
                : 'bg-amber-50 border border-amber-200'
          } rounded-lg`}>
            <h4 className={`text-sm font-medium ${
              currentTheme === 'futuristic' 
                ? 'text-blue-300' 
                : currentTheme === 'dark'
                  ? 'text-amber-300'
                  : 'text-amber-800'
            } mb-2`}>RAG Pipeline Process</h4>
            <ol className={`text-xs ${
              currentTheme === 'futuristic' 
                ? 'text-blue-300/80' 
                : currentTheme === 'dark'
                  ? 'text-amber-300/80'
                  : 'text-amber-700'
            } list-decimal pl-4 space-y-1`}>
              <li>Extract text from uploaded PDF</li>
              <li>Create embeddings and chunked content</li>
              <li>Find most relevant chunks for each entry</li>
              <li>Apply prompt to determine sentence matches</li>
              <li>Return formatted results in table format</li>
            </ol>
          </div>
          
          {/* Preview Prompt Button */}
          <button 
            onClick={handlePreviewPrompt}
            className={`w-full mb-5 px-4 py-2 border ${theme.border} rounded-md ${
              currentTheme === 'futuristic' 
                ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20' 
                : currentTheme === 'dark'
                  ? 'bg-amber-700 hover:bg-amber-800 text-white'
                  : 'bg-white hover:bg-gray-50 text-gray-700'
            }`}
          >
            Preview Selected Prompt
          </button>
          
          {/* Preview Prompt Content Area */}
          {previewPrompt && (
            <div className={`p-3 border ${theme.border} rounded-lg ${theme.cardHighlight} mt-2`}>
              <h4 className={`text-xs font-medium ${theme.text} mb-2`}>Prompt Preview:</h4>
              <div className={`text-xs ${theme.secondaryText} whitespace-pre-wrap max-h-[300px] overflow-y-auto`}>
                {previewPrompt}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default GenerationSection;
import React, { useState, useRef } from 'react';
import { Upload } from 'lucide-react';
import Button from '../ui/Button';
import GenerationModelSelector from './GenerationModelSelector';
import AnalysisModeSelector from '../analysis/AnalysisModeSelector';

interface BulkGenerationFormProps {
  onGenerate: (excelFile: File, pdfFile: File, queryLimit: number, modelId: string, analysisMode: 'full_text' | 'target_headings_only') => void;
  availableHeadings?: string[];
  allDetectedHeadings?: string[];
  onPdfFileChange?: (file: File | React.ChangeEvent<HTMLInputElement>) => void; // External PDF file change handler
  onAnalysisModeChange?: (mode: 'full_text' | 'target_headings_only') => void; // External analysis mode change handler
}

const BulkGenerationForm: React.FC<BulkGenerationFormProps> = ({ 
  onGenerate, 
  availableHeadings = [], 
  allDetectedHeadings = [], 
  onPdfFileChange,
  onAnalysisModeChange 
}) => {
  const [excelFile, setExcelFile] = useState<File | null>(null);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [queryLimit, setQueryLimit] = useState(10);
  const [selectedModel, setSelectedModel] = useState('gpt-4.5'); // Set default model
  const [analysisMode, setAnalysisMode] = useState<'full_text' | 'target_headings_only'>('full_text');
  const excelFileInputRef = useRef<HTMLInputElement>(null);
  const pdfFileInputRef = useRef<HTMLInputElement>(null);

  const handleExcelFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setExcelFile(e.target.files[0]);
    }
  };

  const handlePdfFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setPdfFile(file);
      
      // Call external handler if provided
      if (onPdfFileChange) {
        onPdfFileChange(file);
      }
    }
  };

  const triggerExcelFileInput = () => {
    if (excelFileInputRef.current) {
      excelFileInputRef.current.click();
    }
  };

  const triggerPdfFileInput = () => {
    if (pdfFileInputRef.current) {
      pdfFileInputRef.current.click();
    }
  };

  const handleModelChange = (modelId: string) => {
    // Just update the state, don't trigger generation
    setSelectedModel(modelId);
  };

  const handleAnalysisModeChange = (mode: 'full_text' | 'target_headings_only') => {
    setAnalysisMode(mode);
    if (onAnalysisModeChange) {
      onAnalysisModeChange(mode);
    }
  };

  const handleGenerate = () => {
    if (!excelFile) {
      alert('Please upload an Excel file');
      return;
    }
    
    if (!pdfFile) {
      alert('Please upload a PDF file');
      return;
    }
    
    onGenerate(excelFile, pdfFile, queryLimit, selectedModel, analysisMode);
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Extract Bulk Content</h3>
        <GenerationModelSelector 
          onModelChange={handleModelChange} 
          defaultModel={selectedModel}
        />
      </div>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Excel or CSV File
          </label>
          <div className="flex items-center">
            <input
              ref={excelFileInputRef}
              type="file"
              className="hidden"
              accept=".xlsx,.xls,.csv"
              onChange={handleExcelFileChange}
            />
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={triggerExcelFileInput}
              icon={<Upload className="w-4 h-4" />}
            >
              {excelFile ? excelFile.name : 'Choose Excel/CSV file'}
            </Button>
          </div>
          {excelFile && (
            <p className="mt-1 text-xs text-gray-500">
              Selected file: {excelFile.name} ({Math.round(excelFile.size / 1024)} KB)
            </p>
          )}
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Supporting PDF
          </label>
          <div className="flex items-center">
            <input
              ref={pdfFileInputRef}
              type="file"
              className="hidden"
              accept=".pdf"
              onChange={handlePdfFileChange}
            />
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={triggerPdfFileInput}
              icon={<Upload className="w-4 h-4" />}
            >
              {pdfFile ? pdfFile.name : 'Choose PDF file'}
            </Button>
          </div>
          {pdfFile && (
            <p className="mt-1 text-xs text-gray-500">
              Selected file: {pdfFile.name} ({Math.round(pdfFile.size / 1024)} KB)
            </p>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Analysis Mode
          </label>
          <AnalysisModeSelector
            availableHeadings={availableHeadings}
            selectedMode={analysisMode}
            onModeChange={handleAnalysisModeChange}
            allDetectedHeadings={allDetectedHeadings}
          />
        </div>
        
        <div>
          <label htmlFor="queryLimit" className="block text-sm font-medium text-gray-700 mb-1">
            Query Limit (0 = unlimited)
          </label>
          <input
            id="queryLimit"
            type="number"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-amber-500 focus:border-amber-500"
            value={queryLimit}
            onChange={(e) => setQueryLimit(parseInt(e.target.value) || 0)}
            min={0}
          />
        </div>
        
        <Button
          size="lg"
          fullWidth
          onClick={handleGenerate}
          className="mt-4"
        >
          Extract Bulk Content
        </Button>
      </div>
    </div>
  );
};

export default BulkGenerationForm;
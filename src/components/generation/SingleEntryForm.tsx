import React, { useState, useRef } from 'react';
import { Upload } from 'lucide-react';
import Button from '../ui/Button';
import GenerationModelSelector from './GenerationModelSelector';
import AnalysisModeSelector from '../analysis/AnalysisModeSelector';

interface SingleEntryFormProps {
  onGenerate: (
    activity: string, 
    definition: string, 
    file: File | null, 
    modelId: string, 
    analysisMode: 'full_text' | 'target_headings_only'
  ) => Promise<void>;
  availableHeadings?: string[];
}

const SingleEntryForm: React.FC<SingleEntryFormProps> = ({ onGenerate, availableHeadings }) => {
  const [activity, setActivity] = useState('');
  const [definition, setDefinition] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState('gpt-4.5');
  const [analysisMode, setAnalysisMode] = useState<'full_text' | 'target_headings_only'>('full_text');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleModelChange = (modelId: string) => {
    setSelectedModel(modelId);
  };

  const handleAnalysisModeChange = (mode: 'full_text' | 'target_headings_only') => {
    setAnalysisMode(mode);
  };

  const handleGenerate = async () => {
    if (!activity.trim()) {
      alert('Please enter an activity name');
      return;
    }
    
    if (!definition.trim()) {
      alert('Please enter a definition');
      return;
    }
    
    if (!file) {
      alert('Please upload a PDF file');
      return;
    }
    
    await onGenerate(activity, definition, file, selectedModel, analysisMode);
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Generate Single Entry</h3>
        <GenerationModelSelector 
          onModelChange={handleModelChange} 
          defaultModel={selectedModel}
        />
      </div>
      
      <div className="space-y-4">
        <div>
          <label htmlFor="activity" className="block text-sm font-medium text-gray-700 mb-1">
            Activity Name
          </label>
          <input
            id="activity"
            type="text"
            value={activity}
            onChange={(e) => setActivity(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter activity name"
          />
        </div>

        <div>
          <label htmlFor="definition" className="block text-sm font-medium text-gray-700 mb-1">
            Definition
          </label>
          <textarea
            id="definition"
            value={definition}
            onChange={(e) => setDefinition(e.target.value)}
            rows={3}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter activity definition"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Analysis Mode
          </label>
          <AnalysisModeSelector
            mode={analysisMode}
            onModeChange={handleAnalysisModeChange}
            availableHeadings={availableHeadings}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            PDF File
          </label>
          <div className="flex items-center space-x-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="hidden"
            />
            <Button
              onClick={triggerFileInput}
              variant="outline"
              className="flex items-center space-x-2"
            >
              <Upload className="w-4 h-4" />
              <span>Choose PDF</span>
            </Button>
            {file && (
              <span className="text-sm text-gray-600">
                {file.name}
              </span>
            )}
          </div>
        </div>

        <Button
          onClick={handleGenerate}
          className="w-full"
          disabled={!activity.trim() || !definition.trim() || !file}
        >
          Generate Content
        </Button>
      </div>
    </div>
  );
};

export default SingleEntryForm;
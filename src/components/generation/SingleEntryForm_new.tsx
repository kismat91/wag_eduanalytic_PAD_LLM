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
  ) => void;
  availableHeadings: string[];
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

  const handleGenerate = () => {
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
    
    onGenerate(activity, definition, file, selectedModel, analysisMode);
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
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-amber-500 focus:border-amber-500"
            placeholder="Enter the name of the activity"
            value={activity}
            onChange={(e) => setActivity(e.target.value)}
          />
        </div>
        
        <div>
          <label htmlFor="definition" className="block text-sm font-medium text-gray-700 mb-1">
            Definition
          </label>
          <textarea
            id="definition"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-amber-500 focus:border-amber-500"
            rows={4}
            placeholder="Enter a detailed definition of the activity"
            value={definition}
            onChange={(e) => setDefinition(e.target.value)}
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Supporting PDF
          </label>
          <div className="flex items-center">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf"
              onChange={handleFileChange}
            />
            <Button
              onClick={triggerFileInput}
              className="flex items-center"
            >
              <Upload className="h-4 w-4 mr-2" />
              {file ? file.name : 'Choose PDF'}
            </Button>
          </div>
        </div>

        {/* Analysis Mode Selector */}
        <AnalysisModeSelector
          availableHeadings={availableHeadings}
          selectedMode={analysisMode}
          onModeChange={handleAnalysisModeChange}
        />
        
        <Button 
          onClick={handleGenerate}
          className="w-full bg-amber-600 hover:bg-amber-700 text-white"
        >
          Generate Content
        </Button>
      </div>
    </div>
  );
};

export default SingleEntryForm;

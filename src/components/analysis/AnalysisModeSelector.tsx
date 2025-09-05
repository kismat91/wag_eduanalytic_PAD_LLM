import React from 'react';
import { Info } from 'lucide-react';
import { useTheme } from '../ThemeContext';

interface AnalysisModeProps {
  availableHeadings: string[];
  selectedMode: 'full_text' | 'target_headings_only';
  onModeChange: (mode: 'full_text' | 'target_headings_only') => void;
}

const AnalysisModeSelector: React.FC<AnalysisModeProps> = ({
  availableHeadings,
  selectedMode,
  onModeChange
}) => {
  const { getThemeClasses } = useTheme();
  const theme = getThemeClasses();

  return (
    <div className="space-y-4">
      {/* Analysis Mode Selection */}
      <div>
        <label className={`block text-sm font-medium ${theme.text} mb-2`}>
          Analysis Mode
        </label>
        <div className="space-y-3">
          <label className="flex items-start cursor-pointer">
            <input
              type="radio"
              name="analysisMode"
              value="full_text"
              checked={selectedMode === 'full_text'}
              onChange={() => onModeChange('full_text')}
              className="mr-3 mt-1"
            />
            <div>
              <span className={`${theme.text} font-medium`}>
                Full Text Analysis
              </span>
              <div className="flex items-center mt-1">
                <Info className="h-4 w-4 text-gray-400 mr-2" />
                <span className={`${theme.tertiaryText} text-sm`}>
                  Analyze the entire document content
                </span>
              </div>
            </div>
          </label>
          
          <label className="flex items-start cursor-pointer">
            <input
              type="radio"
              name="analysisMode"
              value="target_headings_only"
              checked={selectedMode === 'target_headings_only'}
              onChange={() => onModeChange('target_headings_only')}
              className="mr-3 mt-1"
            />
            <div>
              <span className={`${theme.text} font-medium`}>
                Target Headings Only
              </span>
              <div className="flex items-center mt-1">
                <Info className="h-4 w-4 text-gray-400 mr-2" />
                <span className={`${theme.tertiaryText} text-sm`}>
                  Analyze only content under specific PAD headings (PDO, Components, Beneficiaries, etc.)
                </span>
              </div>
              
              {/* Show detected headings */}
              {selectedMode === 'target_headings_only' && (
                <div className="mt-2 ml-6">
                  {availableHeadings.length > 0 ? (
                    <div className={`${theme.card} border ${theme.border} rounded-md p-3`}>
                      <span className={`${theme.text} text-sm font-medium`}>
                        Target headings detected in document ({availableHeadings.length}):
                      </span>
                      <ul className="mt-2 space-y-1">
                        {availableHeadings.map((heading, index) => (
                          <li key={index} className={`${theme.tertiaryText} text-xs flex items-center`}>
                            <span className="w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                            {heading}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : (
                    <div className={`${theme.card} border border-orange-300 rounded-md p-3`}>
                      <span className="text-orange-600 text-sm">
                        ⚠️ No target headings detected in document. Will fall back to full text analysis.
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </label>
        </div>
      </div>
    </div>
  );
};

export default AnalysisModeSelector;

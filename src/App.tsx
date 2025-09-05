import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { FileText, MessageSquare, Zap } from 'lucide-react';
import Sidebar from './components/layout/Sidebar';
import ExtractionSection from './components/extraction/ExtractionSection';
import ChatSection from './components/chat/ChatSection';
import GenerationSection from './components/generation/GenerationSection';
import AboutSection from './components/about/AboutSection';
import AnalyticsSection from './components/analytics/AnalyticsSection';
import LandingPage from './components/LandingPage';
import { ThemeProvider } from './components/ThemeContext';
import ThemeSwitcher from './components/ThemeSwitcher';

// Define application sections
export type Section = 'landing' | 'about' | 'extraction' | 'generation' | 'chat' | 'analytics';

// Component to handle navigation state
function AppContent() {
  const [checkComplete, setCheckComplete] = useState(false);
  const [activeSection, setActiveSection] = useState<Section>('landing');
  const location = useLocation();

  // Use useEffect to handle localStorage check and route-based navigation
  useEffect(() => {
    // Check localStorage for hasSeenLanding flag
    const hasSeenLanding = localStorage.getItem('hasSeenLanding') === 'true';
    
    // Set active section based on localStorage
    setActiveSection(hasSeenLanding ? 'about' : 'landing');
    setCheckComplete(true);
  }, []);

  // Update active section based on current route
  useEffect(() => {
    const path = location.pathname;
    if (path === '/about') setActiveSection('about');
    else if (path === '/extraction') setActiveSection('extraction');
    else if (path === '/generation') setActiveSection('generation');
    else if (path === '/chat') setActiveSection('chat');
    else if (path === '/analytics') setActiveSection('analytics');
    else if (path === '/') setActiveSection('landing');
  }, [location]);

  // Handle "Get Started" button click
  const handleGetStarted = () => {
    localStorage.setItem('hasSeenLanding', 'true');
    setActiveSection('about');
  };

  // Don't render anything until check is complete to avoid flashing
  if (!checkComplete) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading application...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen">
      {/* Only show sidebar when not on landing page */}
      {activeSection !== 'landing' && (
        <Sidebar activeSection={activeSection} setActiveSection={setActiveSection} />
      )}
      
      {/* Theme Switcher at top-right (only when not on landing) */}
      {activeSection !== 'landing' && (
        <div className="absolute top-4 right-4 z-10">
          <ThemeSwitcher />
        </div>
      )}
      
      {/* Main Content */}
      <main className={`${activeSection !== 'landing' ? 'flex-1' : 'w-full'} overflow-y-auto`}>
        <Routes>
          <Route path="/" element={<LandingPage onGetStarted={handleGetStarted} />} />
          <Route path="/about" element={<AboutSection />} />
          <Route path="/extraction" element={<ExtractionSection />} />
          <Route path="/generation" element={<GenerationSection />} />
          <Route path="/chat" element={<ChatSection />} />
          <Route path="/analytics" element={<AnalyticsSection />} />
          {/* Redirect any unknown routes to landing */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <ThemeProvider>
      <Router>
        <AppContent />
      </Router>
    </ThemeProvider>
  );
}

export default App;
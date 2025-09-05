#!/usr/bin/env python3
"""
Simple launcher script for the FastAPI application
"""
import uvicorn
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting World Bank PAD Analyzer API...")
    print("Access the API at: http://localhost:8002")
    print("API documentation at: http://localhost:8002/docs")
    
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8002,
        reload=True,
        log_level="info"
    )

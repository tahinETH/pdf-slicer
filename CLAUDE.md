# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Flask-based web application providing utility tools for PDF and audio processing. The app features:
- **PDF Slicer**: Extract table of contents, text, and crop specific sections from PDF documents
- **Audio Transcriber**: Transcribe audio files using Groq's Whisper API with AI-powered cleaning and summarization

## Architecture

### Backend (app.py)
- Flask application with CORS enabled
- Three main service integrations:
  - Groq API for audio transcription (Whisper)
  - OpenAI API for transcript cleaning and summarization
  - PyPDF2/pdfplumber for PDF processing
- File upload handling with size limits (32MB PDFs, 19.5MB audio)
- Automatic audio chunking for large files

### Frontend
- **index.html**: Landing page with tool navigation
- **pdf_slicer.html**: Interactive PDF TOC extraction and text extraction
- **transcriber.html**: Audio upload and transcription interface with AI processing options

## Development Commands

### Running the Application
```bash
# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies (if needed)
pip install flask flask-cors PyPDF2 pdfplumber groq openai python-dotenv pydub

# Run the application
python app.py
```
The app runs on port 9001 by default.

### Environment Setup
The application requires API keys in a `.env` file:
- `GROQ_API_KEY`: For Whisper transcription
- `OPENAI_API_KEY`: For transcript processing
- `ANTHROPIC_API_KEY`: Currently configured but not actively used

## Key Implementation Details

### PDF Processing
- Extracts TOC from PDF bookmarks or content patterns
- Builds hierarchical tree structure for navigation
- Supports text extraction from page ranges
- Can crop and download PDF sections

### Audio Processing
- Supports multiple formats: MP3, WAV, M4A, AAC, OGG, FLAC, WMA, OPUS
- Files under 19MB processed directly
- Larger files automatically chunked into 5-minute segments
- Transcription via Groq Whisper API
- Optional AI cleaning (removes filler words, fixes grammar)
- Optional AI summarization with GPT-4

### File Structure
- `/uploads`: Temporary storage for uploaded files
- `/venv`: Python virtual environment (excluded from git)

## Testing Approach
No formal test suite exists. Manual testing recommended:
1. Test PDF upload and TOC extraction
2. Test text extraction from specific pages
3. Test audio file upload and transcription
4. Verify AI cleaning and summarization features
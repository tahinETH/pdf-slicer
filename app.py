from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import PyPDF2
import pdfplumber
from werkzeug.utils import secure_filename
import re
from io import BytesIO
from flask import send_file
from groq import Groq
from openai import OpenAI
import tempfile
import uuid
from pydub import AudioSegment
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac', 'wma', 'opus'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Groq client
client = Groq()

# Initialize OpenAI client
openai_client = OpenAI()

# Store for chunked audio files (in production, use Redis or database)
audio_chunks_store = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def extract_toc_from_pdf(pdf_path):
    """Extract table of contents from PDF"""
    toc = []
    
    try:
        # Try with PyPDF2 first
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check if PDF has bookmarks (outline)
            if pdf_reader.outline:
                def extract_outline(outline, level=0):
                    for item in outline:
                        if isinstance(item, list):
                            extract_outline(item, level + 1)
                        else:
                            if hasattr(item, 'title'):
                                # Get the page number
                                page_num = None
                                if hasattr(item, 'page') and hasattr(item.page, 'idnum'):
                                    # Find the page number from the destination
                                    for i, page in enumerate(pdf_reader.pages):
                                        if page.indirect_ref.idnum == item.page.idnum:
                                            page_num = i + 1
                                            break
                                
                                toc.append({
                                    'title': item.title,
                                    'level': level,
                                    'page': page_num
                                })
                
                extract_outline(pdf_reader.outline)
        
        # If no bookmarks found, try to extract TOC from content
        if not toc:
            with pdfplumber.open(pdf_path) as pdf:
                # Search for TOC patterns in first 20 pages
                for i, page in enumerate(pdf.pages[:20]):
                    text = page.extract_text()
                    if text:
                        # Look for common TOC patterns
                        lines = text.split('\n')
                        for line in lines:
                            # Pattern: Chapter/Section number followed by title and page number
                            match = re.match(r'^(\d+\.?\d*\.?\d*)\s+(.+?)\s+(\d+)$', line.strip())
                            if match:
                                toc.append({
                                    'title': f"{match.group(1)} {match.group(2)}",
                                    'level': match.group(1).count('.'),
                                    'page': int(match.group(3))
                                })
                            # Pattern: Title followed by dots and page number
                            match2 = re.match(r'^(.+?)\.{3,}\s*(\d+)$', line.strip())
                            if match2:
                                toc.append({
                                    'title': match2.group(1).strip(),
                                    'level': 0,
                                    'page': int(match2.group(2))
                                })
                        
                        # If we found TOC entries, stop searching
                        if len(toc) > 5:
                            break
    
    except Exception as e:
        print(f"Error extracting TOC: {str(e)}")
    
    return toc

def build_toc_tree(flat_toc, total_pages=None):
    """Convert flat TOC list to hierarchical tree structure"""
    if not flat_toc:
        return []
    
    # Sort by page number to ensure correct order
    flat_toc = sorted(flat_toc, key=lambda x: x.get('page', 0) or 0)
    
    # Create a tree structure
    tree = []
    stack = []
    
    for i, item in enumerate(flat_toc):
        node = {
            'id': f'toc-{i}',
            'title': item['title'],
            'level': item['level'],
            'page': item['page'],
            'children': [],
            'start_page': item['page'],
            'end_page': None  # Will be calculated
        }
        
        # Find the parent based on level
        while stack and stack[-1]['level'] >= node['level']:
            stack.pop()
        
        if stack:
            # Add as child to the last item in stack
            stack[-1]['children'].append(node)
        else:
            # Add to root
            tree.append(node)
        
        stack.append(node)
    
    # Get total pages for end page calculation if not provided
    if total_pages is None:
        total_pages = max([item.get('page', 0) or 0 for item in flat_toc] + [100])  # Default to 100 if no pages
    
    # Post-process to set end pages
    for i, node in enumerate(tree):
        if i < len(tree) - 1:
            set_end_page_recursive(node, tree[i + 1]['start_page'] - 1)
        else:
            set_end_page_recursive(node, total_pages)
    
    return tree

def set_end_page_recursive(node, max_end_page):
    """Recursively set end pages for a node and its children"""
    if node['children']:
        # Has children, so its end page is determined by its last child
        for i, child in enumerate(node['children']):
            if i < len(node['children']) - 1:
                set_end_page_recursive(child, node['children'][i + 1]['start_page'] - 1)
            else:
                set_end_page_recursive(child, max_end_page)
        node['end_page'] = node['children'][-1]['end_page']
    else:
        # Leaf node
        node['end_page'] = max_end_page

def extract_text_from_pages(pdf_path, start_page, end_page):
    """Extract text from specific page range"""
    extracted_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Adjust for 0-based indexing
            start_idx = max(0, start_page - 1)
            end_idx = min(len(pdf.pages), end_page)
            
            for i in range(start_idx, end_idx):
                page = pdf.pages[i]
                text = page.extract_text()
                if text:
                    extracted_text.append({
                        'page': i + 1,
                        'text': text
                    })
    
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
    
    return extracted_text

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/pdf-slicer')
def pdf_slicer():
    return send_from_directory('.', 'pdf_slicer.html')

@app.route('/transcriber')
def transcriber():
    return send_from_directory('.', 'transcriber.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get total pages first
        total_pages = 0
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            total_pages = len(pdf_reader.pages)
        
        # Extract TOC
        flat_toc = extract_toc_from_pdf(filepath)
        
        # Build hierarchical tree and update end pages
        tree_toc = build_toc_tree(flat_toc, total_pages)
        
        return jsonify({
            'filename': filename,
            'filepath': filepath,
            'toc': flat_toc,  # Keep flat structure for compatibility
            'toc_tree': tree_toc,  # New hierarchical structure
            'total_pages': total_pages
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/extract-text', methods=['POST'])
def extract_text():
    data = request.json
    filepath = data.get('filepath')
    start_page = data.get('start_page', 1)
    end_page = data.get('end_page', 1)
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    extracted_text = extract_text_from_pages(filepath, start_page, end_page)
    
    # Combine all text for easy copying
    combined_text = '\n\n'.join([f"--- Page {item['page']} ---\n{item['text']}" for item in extracted_text])
    
    return jsonify({
        'pages': extracted_text,
        'combined_text': combined_text
    })

@app.route('/crop-pdf', methods=['POST'])
def crop_pdf():
    data = request.json
    filepath = data.get('filepath')
    start_page = data.get('start_page', 1)
    end_page = data.get('end_page', 1)
    title = data.get('title', 'cropped')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Create a PDF writer object
        pdf_writer = PyPDF2.PdfWriter()
        
        # Read the original PDF
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Add pages from start_page to end_page (1-indexed)
            for page_num in range(start_page - 1, min(end_page, len(pdf_reader.pages))):
                pdf_writer.add_page(pdf_reader.pages[page_num])
        
        # Create a BytesIO object to store the PDF in memory
        output_buffer = BytesIO()
        pdf_writer.write(output_buffer)
        output_buffer.seek(0)
        
        # Generate filename
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        filename = f"{safe_title}_pages_{start_page}-{end_page}.pdf"
        
        return send_file(
            output_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({'error': f'Failed to crop PDF: {str(e)}'}), 500

def chunk_audio_file(file_path, chunk_duration_ms=300000):  # 5 minutes chunks
    """Split audio file into chunks for processing"""
    try:
        # Load audio file
        audio = AudioSegment.from_file(file_path)
        
        # Calculate number of chunks needed
        total_duration = len(audio)
        num_chunks = math.ceil(total_duration / chunk_duration_ms)
        
        chunks = []
        for i in range(num_chunks):
            start_time = i * chunk_duration_ms
            end_time = min(start_time + chunk_duration_ms, total_duration)
            
            # Extract chunk
            chunk = audio[start_time:end_time]
            
            # Save chunk to temporary file
            chunk_id = str(uuid.uuid4())
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            chunk.export(temp_file.name, format='wav')
            
            chunks.append({
                'id': chunk_id,
                'file_path': temp_file.name,
                'start_time': start_time / 1000,  # Convert to seconds
                'end_time': end_time / 1000,
                'duration': (end_time - start_time) / 1000
            })
        
        return chunks
    except Exception as e:
        print(f"Error chunking audio: {str(e)}")
        return None

def transcribe_audio_chunk(file_path):
    """Transcribe a single audio chunk using Groq Whisper"""
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
            return transcription.text
    except Exception as e:
        print(f"Error transcribing audio chunk: {str(e)}")
        raise e

def clean_transcript_with_llm(transcript):
    """Clean up transcript using GPT-4.1 to make it more coherent"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional transcript editor. Your job is to clean up and improve the coherence of audio transcripts while preserving the original meaning and content. 

Please:
- Fix grammatical errors and improve sentence structure
- Remove unnecessary filler words (um, uh, like, you know, etc.) but keep natural speech patterns
- Correct obvious transcription errors
- Improve punctuation and capitalization
- Break up run-on sentences into more readable segments
- Maintain the speaker's original tone and meaning
- Do not add new information or change the core message

Return only the cleaned transcript without any additional commentary."""
                },
                {
                    "role": "user",
                    "content": f"Please clean up this transcript:\n\n{transcript}"
                }
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error cleaning transcript: {str(e)}")
        raise e

def summarize_transcript_with_llm(transcript):
    """Summarize transcript using GPT-4.1"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional summarizer. Create a concise, well-structured summary of the given transcript that captures the key points, main ideas, and important details.

Please:
- Identify and highlight the main topics and themes
- Preserve important details and specific information
- Use clear, professional language
- Structure the summary with bullet points or paragraphs as appropriate
- Include key quotes or statements when relevant
- Maintain objectivity and accuracy

Format your response with:
1. A brief overview paragraph
2. Key points in bullet format
3. Any important conclusions or action items (if applicable)"""
                },
                {
                    "role": "user",
                    "content": f"Please summarize this transcript:\n\n{transcript}"
                }
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error summarizing transcript: {str(e)}")
        raise e

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_audio_file(file.filename):
        return jsonify({'error': 'Invalid audio file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(filepath)
        
        # Check file size - if under 19MB, process directly
        file_size = os.path.getsize(filepath)
        max_direct_size = 19 * 1024 * 1024  # 19MB
        
        if file_size <= max_direct_size:
            # Direct transcription
            try:
                transcription_text = transcribe_audio_chunk(filepath)
                
                # Clean up
                os.remove(filepath)
                
                return jsonify({
                    'transcription': transcription_text,
                    'chunks': None
                })
            except Exception as e:
                os.remove(filepath)
                return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
        else:
            # File too large, need chunking
            chunks = chunk_audio_file(filepath)
            if not chunks:
                os.remove(filepath)
                return jsonify({'error': 'Failed to process audio file'}), 500
            
            # Store chunks info for later processing
            audio_chunks_store[file_id] = {
                'original_file': filepath,
                'chunks': chunks
            }
            
            # Return chunk information
            return jsonify({
                'file_id': file_id,
                'chunks': [
                    {
                        'id': chunk['id'],
                        'start_time': chunk['start_time'],
                        'end_time': chunk['end_time'],
                        'duration': chunk['duration']
                    }
                    for chunk in chunks
                ],
                'transcription': None
            })
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/transcribe-chunk', methods=['POST'])
def transcribe_chunk():
    data = request.json
    chunk_id = data.get('chunk_id')
    file_id = data.get('file_id')
    
    if not chunk_id or not file_id:
        return jsonify({'error': 'Missing chunk_id or file_id'}), 400
    
    if file_id not in audio_chunks_store:
        return jsonify({'error': 'File not found'}), 404
    
    # Find the specific chunk
    chunks = audio_chunks_store[file_id]['chunks']
    chunk = next((c for c in chunks if c['id'] == chunk_id), None)
    
    if not chunk:
        return jsonify({'error': 'Chunk not found'}), 404
    
    try:
        # Transcribe the chunk
        transcription_text = transcribe_audio_chunk(chunk['file_path'])
        
        # Clean up chunk file
        os.remove(chunk['file_path'])
        
        return jsonify({
            'transcription': transcription_text,
            'chunk_id': chunk_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Chunk transcription failed: {str(e)}'}), 500

@app.route('/clean-transcript', methods=['POST'])
def clean_transcript():
    data = request.json
    transcript = data.get('transcript')
    
    if not transcript:
        return jsonify({'error': 'No transcript provided'}), 400
    
    try:
        cleaned_transcript = clean_transcript_with_llm(transcript)
        return jsonify({
            'cleaned_transcript': cleaned_transcript
        })
    except Exception as e:
        return jsonify({'error': f'Failed to clean transcript: {str(e)}'}), 500

@app.route('/summarize-transcript', methods=['POST'])
def summarize_transcript():
    data = request.json
    transcript = data.get('transcript')
    
    if not transcript:
        return jsonify({'error': 'No transcript provided'}), 400
    
    try:
        summary = summarize_transcript_with_llm(transcript)
        return jsonify({
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': f'Failed to summarize transcript: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=9001) 
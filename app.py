from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import PyPDF2
import pdfplumber
from werkzeug.utils import secure_filename
import re
from io import BytesIO
from flask import send_file

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
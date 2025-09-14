#!/usr/bin/env python3
"""
Flask server to connect HTML frontend with RAG backend
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sys
from rag_system import FrenchRAG

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Initialize RAG system
print("🦙 Initializing French RAG system...")
rag = FrenchRAG(use_local_llama=True)
print(f"🎯 RAG Model detected: {rag.current_model}")
print(f"🔧 RAG Model available: {bool(rag.current_model)}")

# Test a simple query to verify it works
if rag.current_model:
    print("🧪 Testing RAG query...")
    test_response = rag.query("test")
    print(f"✅ RAG test successful: {len(test_response)} characters")
else:
    print("⚠️ RAG model not available - will use educational mode")

# Load documents if available
if os.path.exists('course_materials'):
    print("📚 Loading course materials...")
    rag.load_documents('course_materials')
else:
    print("📂 No course_materials folder found - using built-in knowledge")

@app.route('/')
def index():
    """Serve the HTML interface"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return """
        <h1>🇫🇷 Assistant Français B2</h1>
        <p>index.html not found. Please make sure it's in the same directory.</p>
        """

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with RAG"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        print(f"💬 User query: {message}")
        
        # Use RAG system to get response
        response = rag.query(message)
        
        print(f"🤖 RAG response length: {len(response)} chars")
        
        return jsonify({
            'response': response,
            'model': rag.current_model or 'educational_mode',
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'response': f"⚠️ Erreur temporaire: {str(e)}",
            'status': 'error'
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status"""
    return jsonify({
        'ollama_model': rag.current_model,
        'model_available': bool(rag.current_model),
        'documents_loaded': rag.collection.count() if hasattr(rag.collection, 'count') else 0,
        'status': 'ready' if rag.current_model else 'no_model'
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'French RAG Assistant',
        'model': rag.current_model or 'educational_mode'
    })

if __name__ == '__main__':
    print("🚀 Starting French RAG Server...")
    print("📡 Server will be available at: http://localhost:5000")
    print("🦙 RAG Model:", rag.current_model or "Educational Mode")
    print("📚 Documents loaded:", rag.collection.count() if hasattr(rag.collection, 'count') else 0)
    print("-" * 50)
    
    # Install Flask and CORS if needed
    try:
        import flask_cors
    except ImportError:
        print("⚠️  Installing required packages...")
        os.system("pip install flask flask-cors")
    
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        threaded=True
    )
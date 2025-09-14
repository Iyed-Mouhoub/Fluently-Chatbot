#!/usr/bin/env python3
"""
Flask server to connect HTML frontend with RAG backend (OpenRouter support)
"""
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sys
from rag_system import FrenchRAG

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Initialize RAG system with OpenRouter
print("🌐 Initializing French RAG system with OpenRouter...")

# You can set your OpenRouter API key in multiple ways:
# 1. Environment variable: export OPENROUTER_API_KEY="your_key_here"
# 2. Pass directly to constructor (not recommended for production)
# 3. Create a .env file with OPENROUTER_API_KEY=your_key_here

rag = FrenchRAG(
    use_openrouter=True,
    # openrouter_api_key="your_api_key_here"  # Uncomment and add your key if not using env variable
)

print(f"🎯 RAG Model: {rag.current_model}")
print(f"🔧 Model available: {bool(rag.current_model)}")
print(f"🌐 Using OpenRouter: {rag.use_openrouter}")

# Test a simple query to verify it works
if rag.current_model:
    print("🧪 Testing RAG query...")
    try:
        test_response = rag.query("Bonjour")
        print(f"✅ RAG test successful: {len(test_response)} characters")
    except Exception as e:
        print(f"⚠️ RAG test failed: {e}")
else:
    print("⚠️ RAG model not available")

# Load documents if available
if os.path.exists('course_materials'):
    print("📚 Loading course materials...")
    rag.load_documents('course_materials')
    print(f"📊 Documents loaded: {rag.index.ntotal} chunks")
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
        <p>Using OpenRouter API: {}</p>
        <p>Model: {}</p>
        """.format(rag.use_openrouter, rag.current_model or 'None')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with RAG"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        print(f"💬 User query: {message}")
        
        # Check if model is available
        if not rag.current_model:
            return jsonify({
                'response': "❌ Aucun modèle disponible. Vérifiez votre clé API OpenRouter.",
                'model': 'none',
                'status': 'error'
            })

        # Use RAG system to get response
        response = rag.query(message)
        print(f"🤖 RAG response length: {len(response)} chars")

        return jsonify({
            'response': response,
            'model': rag.current_model,
            'provider': 'openrouter' if rag.use_openrouter else 'ollama',
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
        'model': rag.current_model,
        'model_available': bool(rag.current_model),
        'provider': 'openrouter' if rag.use_openrouter else 'ollama',
        'documents_loaded': rag.index.ntotal,
        'status': 'ready' if rag.current_model else 'no_model',
        'api_key_configured': bool(rag.openrouter_api_key) if rag.use_openrouter else None
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'French RAG Assistant',
        'model': rag.current_model or 'none',
        'provider': 'openrouter' if rag.use_openrouter else 'ollama'
    })

@app.route('/api/models', methods=['GET'])
def available_models():
    """Get available models"""
    if rag.use_openrouter:
        return jsonify({
            'provider': 'openrouter',
            'current_model': rag.current_model,
            'available_models': rag.openrouter_models
        })
    else:
        return jsonify({
            'provider': 'ollama',
            'current_model': rag.current_model,
            'available_models': rag.ollama_models if hasattr(rag, 'ollama_models') else []
        })

@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    try:
        data = request.get_json()
        new_model = data.get('model', '').strip()
        
        if not new_model:
            return jsonify({'error': 'No model specified'}), 400
            
        if rag.use_openrouter and new_model in rag.openrouter_models:
            rag.current_model = new_model
            return jsonify({
                'success': True,
                'message': f'Switched to {new_model}',
                'current_model': rag.current_model
            })
        elif not rag.use_openrouter and new_model in getattr(rag, 'ollama_models', []):
            rag.current_model = new_model
            return jsonify({
                'success': True,
                'message': f'Switched to {new_model}',
                'current_model': rag.current_model
            })
        else:
            return jsonify({'error': 'Model not available'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting French RAG Server...")
    print("📡 Server will be available at: http://localhost:5000")
    print("🌐 Provider:", "OpenRouter" if rag.use_openrouter else "Ollama")
    print("🤖 Model:", rag.current_model or "None")
    print("📚 Documents loaded:", rag.index.ntotal)
    print("-" * 50)
    
    # Install required packages if needed
    try:
        import flask_cors
    except ImportError:
        print("⚠️ Installing required packages...")
        os.system("pip install flask flask-cors")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
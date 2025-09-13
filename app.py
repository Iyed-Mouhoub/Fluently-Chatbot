import streamlit as st
from rag_system import FrenchRAG
import os
import time
import requests

# Page config
st.set_page_config(
    page_title="Assistant Français B2",
    page_icon="🇫🇷",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
.stChatMessage {
    padding: 1rem;
    border-radius: 10px;
}
.success-box {
    padding: 10px;
    border-left: 5px solid #28a745;
    background-color: #d4edda;
    margin: 10px 0;
    border-radius: 5px;
}
.warning-box {
    padding: 10px;
    border-left: 5px solid #ffc107;
    background-color: #fff3cd;
    margin: 10px 0;
    border-radius: 5px;
}
.info-box {
    padding: 10px;
    border-left: 5px solid #17a2b8;
    background-color: #d1ecf1;
    margin: 10px 0;
    border-radius: 5px;
}
.error-box {
    padding: 10px;
    border-left: 5px solid #dc3545;
    background-color: #f8d7da;
    margin: 10px 0;
    border-radius: 5px;
}
.install-code {
    background-color: #f8f9fa;
    padding: 8px;
    border-radius: 4px;
    font-family: monospace;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🇫🇷 Assistant Français B2")
st.markdown("*Chatbot interactif avec IA Llama locale*")

# Check Ollama status
def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = [model['name'] for model in response.json().get('models', [])]
            return True, models
        return False, []
    except:
        return False, []

# Sidebar for configuration and status
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Check Ollama status
    ollama_running, available_models = check_ollama_status()
    
    if ollama_running:
        st.markdown("""
        <div class="success-box">
        ✅ <b>Ollama détecté</b><br>
        🦙 Service local actif
        </div>
        """, unsafe_allow_html=True)
        
        if available_models:
            st.markdown(f"""
            <div class="info-box">
            🎯 <b>Modèles disponibles:</b><br>
            {'<br>'.join([f'• {model}' for model in available_models[:5]])}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            ⚠️ <b>Aucun modèle installé</b><br>
            Voir les instructions d'installation ci-dessous
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-box">
        ❌ <b>Ollama non détecté</b><br>
        📌 Service local non démarré
        </div>
        """, unsafe_allow_html=True)
        
        # Installation instructions
        st.markdown("---")
        st.title("🛠️ Installation Ollama")
        
        st.markdown("""
        **Étape 1: Installer Ollama**
        """)
        
        st.markdown("""
        <div class="install-code">
        # Windows/Mac: Télécharger depuis<br>
        https://ollama.ai/download<br><br>
        # Linux:<br>
        curl -fsSL https://ollama.ai/install.sh | sh
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Étape 2: Installer un modèle léger**
        """)
        
        recommended_models = [
            ("llama3.2:1b", "1.3GB", "Très rapide, recommandé"),
            ("tinyllama", "637MB", "Ultra léger"),
            ("phi3:mini", "2.3GB", "Efficace de Microsoft"),
            ("gemma:2b", "1.4GB", "Léger de Google")
        ]
        
        for model, size, desc in recommended_models:
            st.markdown(f"""
            <div class="info-box">
            <b>{model}</b> ({size})<br>
            {desc}<br>
            <code>ollama pull {model}</code>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        **Étape 3: Démarrer Ollama**
        """)
        
        st.markdown("""
        <div class="install-code">
        ollama serve
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Étape 4: Actualiser cette page**
        """)
        
        if st.button("🔄 Vérifier Ollama", use_container_width=True):
            st.rerun()

# Initialize RAG system
@st.cache_resource
def init_rag_system():
    return FrenchRAG(use_local_llama=True)

# Load RAG system
if 'rag_initialized' not in st.session_state:
    with st.spinner("🔄 Initialisation du système RAG..."):
        try:
            st.session_state.rag = init_rag_system()
            
            # Load documents
            if os.path.exists('course_materials'):
                st.session_state.rag.load_documents('course_materials')
                st.session_state.documents_loaded = True
                chunk_count = st.session_state.rag.collection.count()
            else:
                st.session_state.documents_loaded = False
                chunk_count = 0
                
            st.session_state.rag_initialized = True
            st.session_state.chunk_count = chunk_count
            
        except Exception as e:
            st.error(f"❌ Erreur d'initialisation: {str(e)}")
            st.stop()

# Display system status in sidebar
with st.sidebar:
    st.markdown("---")
    st.title("📊 Statut du Système")
    
    # Model status
    if hasattr(st.session_state, 'rag') and st.session_state.rag.current_model:
        st.markdown(f"""
        <div class="success-box">
        🦙 <b>Modèle actif:</b><br>
        {st.session_state.rag.current_model}
        </div>
        """, unsafe_allow_html=True)
    elif ollama_running:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Aucun modèle sélectionné</b><br>
        Installez un modèle léger
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-box">
        ❌ <b>Mode éducatif uniquement</b><br>
        Ollama non configuré
        </div>
        """, unsafe_allow_html=True)
    
    # Documents status
    if st.session_state.get('documents_loaded', False):
        st.markdown(f"""
        <div class="success-box">
        ✅ <b>Documents chargés</b><br>
        📄 {st.session_state.get('chunk_count', 0)} chunks disponibles
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Aucun document trouvé</b><br>
        📁 Placez vos fichiers dans 'course_materials/'
        </div>
        """, unsafe_allow_html=True)

# Initialize chat history
if 'messages' not in st.session_state:
    # Welcome message based on system status
    if ollama_running and hasattr(st.session_state, 'rag') and st.session_state.rag.current_model:
        welcome_msg = f"""Bonjour! 👋 Je suis votre assistant français B2 avec IA Llama locale.

🦙 **Modèle actuel:** {st.session_state.rag.current_model if hasattr(st.session_state, 'rag') else 'En cours de chargement...'}

Je peux vous aider avec:
- 📚 Grammaire et conjugaison
- 🗣️ Conversations et dialogues  
- 📖 Vocabulaire et expressions
- 🎯 Exercices pratiques

N'hésitez pas à me poser vos questions en français! 🇫🇷"""
    else:
        welcome_msg = """Bonjour! 👋 Je suis votre assistant français B2.

📚 **Mode éducatif actuel** (sans IA locale)

Je peux vous aider avec:
- 📚 Grammaire et conjugaison
- 🗣️ Conversations et dialogues
- 📖 Vocabulaire et expressions  
- 🎯 Exercices pratiques

🛠️ **Pour activer l'IA Llama:** Suivez les instructions dans la barre latérale.

N'hésitez pas à me poser vos questions en français! 🇫🇷"""
    
    st.session_state.messages = [
        {"role": "assistant", "content": welcome_msg}
    ]

# Chat interface
st.subheader("💬 Conversation")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Posez votre question en français..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("💭 Génération de la réponse..."):
            try:
                start_time = time.time()
                response = st.session_state.rag.query(prompt)
                end_time = time.time()
                
                # Display response
                st.markdown(response)
                
                # Add timing info for debugging
                response_time = end_time - start_time
                if response_time > 5:  # If response took long, show timing
                    st.caption(f"⏱️ Temps de réponse: {response_time:.1f}s")
                    
            except Exception as e:
                error_response = f"""❌ **Erreur temporaire**

Je ne peux pas traiter votre question maintenant: `{str(e)[:100]}`

💡 **Essayez**:
- Reformuler votre question plus simplement
- Utiliser une des questions d'exemple
- Vérifier que Ollama fonctionne correctement"""
                
                st.markdown(error_response)
                response = error_response
            
            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Example questions in sidebar
with st.sidebar:
    st.markdown("---")
    st.title("💡 Questions d'exemple")
    
    example_questions = [
        "Salut! Je veux apprendre le français",
        "Explique-moi le passé composé",
        "Créons un dialogue au restaurant",
        "Différence entre 'cependant' et 'pourtant'",
        "Comment conjuguer 'être' au subjonctif?",
        "Aide-moi avec la prononciation"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(question, key=f"ex_{i}", use_container_width=True):
            # Add question to chat
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Generate response
            try:
                response = st.session_state.rag.query(question)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ Erreur: {str(e)[:50]}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            st.rerun()

# Statistics in sidebar
with st.sidebar:
    st.markdown("---")
    st.title("📈 Statistiques")
    st.markdown(f"💬 **Messages échangés**: {len(st.session_state.messages)}")
    
    # System mode
    if ollama_running and hasattr(st.session_state, 'rag') and st.session_state.rag.current_model:
        st.markdown("🤖 **Mode**: IA Llama + RAG")
    else:
        st.markdown("📚 **Mode**: Éducatif seul")
    
    # Clear chat button
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
        st.rerun()

# Quick setup guide
with st.sidebar:
    st.markdown("---")
    st.title("🚀 Guide rapide")
    
    with st.expander("📖 Instructions complètes"):
        st.markdown("""
        **Installation complète:**
        
        1. **Installer Ollama:**
           - Windows/Mac: https://ollama.ai/download
           - Linux: `curl -fsSL https://ollama.ai/install.sh | sh`
        
        2. **Choisir un modèle léger:**
           ```bash
           # Recommandé (1.3GB)
           ollama pull llama3.2:1b
           
           # Alternative ultra-légère (637MB)
           ollama pull tinyllama
           
           # Microsoft (2.3GB)
           ollama pull phi3:mini
           ```
        
        3. **Démarrer Ollama:**
           ```bash
           ollama serve
           ```
        
        4. **Lancer l'app:**
           ```bash
           streamlit run app.py
           ```
        
        **💡 Conseils:**
        - Commencez avec `llama3.2:1b` (bon équilibre)
        - Si RAM limitée, utilisez `tinyllama`
        - Vérifiez que le port 11434 est libre
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em;">
🎓 Assistant Français B2 | 
Propulsé par Ollama + Llama + RAG | 
<a href="https://ollama.ai" target="_blank">Ollama</a> pour l'IA locale
</div>
""", unsafe_allow_html=True)
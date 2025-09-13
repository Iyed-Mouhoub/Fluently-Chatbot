import streamlit as st
from rag_system import FrenchRAG
import os

# Initialize
if 'rag' not in st.session_state:
    st.session_state.rag = FrenchRAG()
    if os.path.exists('course_materials'):
        with st.spinner("🔄 Chargement des documents du cours..."):
            st.session_state.rag.load_documents('course_materials')
        st.session_state.loaded = True
    else:
        st.session_state.loaded = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

# UI
st.title("🇫🇷 Assistant Français B2")
st.write("Chatbot interactif basé sur votre cours de français")

# File check
if not st.session_state.loaded:
    st.error("❌ Dossier 'course_materials/' non trouvé ou vide")
    st.info("📁 Assurez-vous que vos fichiers .txt sont dans le dossier 'course_materials/'")
    st.stop()

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
        with st.spinner("🤔 Je réfléchis..."):
            response = st.session_state.rag.query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with example questions
st.sidebar.title("💡 Questions d'exemple")
example_questions = [
    "Explique-moi la différence entre 'vénérer' et 'respecter'",
    "Aide-moi avec un dialogue au café",
    "Quels sont les temps du passé en français?",
    "Créons un exercice de compréhension",
    "Explique-moi les expressions avec des animaux",
    "Comment prononcer les chiffres 5, 6, 9, 10?"
]

for question in example_questions:
    if st.sidebar.button(question, key=f"ex_{hash(question)}"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("🤔 Je réfléchis..."):
            response = st.session_state.rag.query(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Status info
st.sidebar.markdown("---")
st.sidebar.markdown(f"📊 **Status**: Prêt")
st.sidebar.markdown(f"📁 **Documents**: Chargés")
st.sidebar.markdown(f"💬 **Messages**: {len(st.session_state.messages)}")
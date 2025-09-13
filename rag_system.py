import chromadb
from sentence_transformers import SentenceTransformer
import os
import ollama
import PyPDF2

class FrenchRAG:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.create_collection("french_course")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_pdf_text(self, pdf_path):
        """Extract text from a single PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text
    
    def load_documents(self, folder_path):
        documents = []
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            content = ""
            
            # Handle PDF files
            if filename.endswith('.pdf'):
                print(f"Converting {filename}...")
                content = self.extract_pdf_text(file_path)
            
            # Handle TXT files
            elif filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Skip other file types
            else:
                continue
            
            if content.strip():
                # Split into chunks
                chunks = content.split('\n\n')
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        documents.append({
                            'id': f"{filename}_{i}",
                            'text': chunk.strip(),
                            'source': filename
                        })
        
        print(f"Processed {len(documents)} text chunks from {len([f for f in os.listdir(folder_path) if f.endswith(('.pdf', '.txt'))])} files")
        
        # Add to ChromaDB
        if documents:
            texts = [doc['text'] for doc in documents]
            ids = [doc['id'] for doc in documents]
            embeddings = self.encoder.encode(texts).tolist()
            
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids
            )
            
    def query(self, question, n_results=3):
        # Get relevant documents
        query_embedding = self.encoder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Build context
        context = "\n".join(results['documents'][0])
        
        # Create French prompt
        prompt = f"""Tu es un assistant pédagogique spécialisé en français niveau B2. 

        INSTRUCTIONS IMPORTANTES:
        - Le contexte extrait peut être incomplet ou fragmenté (provient de PDFs convertis)
        - Si le contexte est vague, utilise tes connaissances linguistiques pour compléter et clarifier
        - Fais des liens logiques entre les éléments du cours pour donner des réponses cohérentes
        - Adapte ton niveau au B2: explications claires mais sophistiquées
        - Sois interactif: pose des questions, propose des exercices, donne des exemples
        - Si tu manques d'informations spécifiques du cours, complète avec tes connaissances générales du français

        Contexte du cours: {context}

        Question de l'étudiant: {question}

        Réponds en français de manière pédagogique, en combinant intelligemment le contenu du cours avec tes connaissances linguistiques pour offrir une réponse complète et utile."""
        
        # Query Ollama
        response = ollama.chat(
            model='mistral',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']
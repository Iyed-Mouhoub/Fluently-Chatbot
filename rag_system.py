import chromadb
from sentence_transformers import SentenceTransformer
import os
import requests
import json
import time
import re
from typing import Optional

class FrenchRAG:
    def __init__(self, use_local_llama=True):
        # ChromaDB setup
        self.client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.client.get_collection("french_course")
            print("📚 Collection existante chargée")
        except:
            self.collection = self.client.create_collection("french_course")
            print("📚 Nouvelle collection créée")
            
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Local Llama setup
        self.use_local_llama = use_local_llama
        self.ollama_url = "http://localhost:11434/api/generate"
        
        self.models = [
            "llama3.2:1b",
            "llama3.1:8b", 
            "llama3.2:3b",
            "tinyllama",
            "phi3:mini",
            "gemma:2b",
            "mistral",
        ]
        
        self.current_model = None
        self.check_ollama_models()
        
        # Define simple greetings that don't need document search
        self.simple_greetings = {
            'salut', 'bonjour', 'bonsoir', 'hello', 'hi', 'hey', 'coucou'
        }
    
    def check_ollama_models(self):
        """Check which models are available in Ollama"""
        if not self.use_local_llama:
            return
            
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                print(f"🦙 Ollama détecté avec {len(available_models)} modèles")
                print(f"📋 Modèles disponibles: {', '.join(available_models)}")
                
                for model in self.models:
                    if any(model in available for available in available_models):
                        self.current_model = model
                        print(f"✅ Modèle sélectionné: {model}")
                        break
                
                if not self.current_model and available_models:
                    self.current_model = available_models[0]
                    print(f"📋 Utilisation du modèle disponible: {self.current_model}")
                elif not self.current_model:
                    print("⚠️ Aucun modèle trouvé.")
                    
            else:
                print("❌ Ollama non détecté")
                
        except requests.exceptions.ConnectionError:
            print("🔌 Ollama non démarré")
        except Exception as e:
            print(f"❌ Erreur Ollama: {e}")

    def clean_pdf_content(self, text):
        """Clean messy PDF content"""
        if not text:
            return ""
            
        # Remove common PDF artifacts
        text = re.sub(r'--- Page \d+ ---', '', text)
        text = re.sub(r'@daily\.french_', '', text)
        text = re.sub(r'Prof : Labed Nada', '', text)
        text = re.sub(r'Niveau intermédiaire B1/B2', '', text)
        text = re.sub(r'ATELIER DE.*?CONVERSATION', 'Atelier de conversation', text)
        text = re.sub(r'🔵|🎯', '', text)
        
        # Remove excessive whitespace and duplicates
        lines = []
        seen_lines = set()
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 15 and line not in seen_lines:
                lines.append(line)
                seen_lines.add(line)
        
        return '\n'.join(lines[:3])
        
    def extract_pdf_text(self, pdf_path):
        """Extract and clean text from PDF"""
        import PyPDF2
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        cleaned_text = self.clean_pdf_content(page_text)
                        if cleaned_text:
                            text += f"Page {page_num + 1}:\n{cleaned_text}\n\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        
        return self.clean_pdf_content(text)
    
    def load_documents(self, folder_path):
        if self.collection.count() > 0:
            print(f"📚 Documents déjà chargés ({self.collection.count()} chunks)")
            return
            
        documents = []
        print("📄 Traitement des documents...")
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            content = ""
            
            if filename.endswith('.pdf'):
                print(f"📄 Conversion {filename}...")
                content = self.extract_pdf_text(file_path)
            elif filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                content = self.clean_pdf_content(content)
            else:
                continue
            
            if content.strip():
                chunks = self.smart_chunk_text(content)
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 30:
                        documents.append({
                            'id': f"{filename}_{i}",
                            'text': chunk.strip(),
                            'source': filename
                        })
        
        if documents:
            print(f"📊 Traitement de {len(documents)} chunks...")
            texts = [doc['text'] for doc in documents]
            ids = [doc['id'] for doc in documents]
            embeddings = self.encoder.encode(texts).tolist()
            
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids
            )
            print(f"✅ {len(documents)} chunks ajoutés à la base de données")
            
    def smart_chunk_text(self, text):
        """Smart text chunking - optimized for smaller models"""
        if len(text) < 200:
            return [text]
            
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_size = 300 if "llama3.2:1b" in str(self.current_model) else 500
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(current_chunk + paragraph) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def is_context_relevant(self, question: str, context: str) -> bool:
        """Check if retrieved context is actually relevant to the question"""
        if not context or len(context.strip()) < 20:
            return False
            
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        
        # Remove common French words that don't indicate relevance
        common_words = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'à', 'il', 'elle', 'dans', 'pour', 'avec', 'sur', 'par'}
        question_words -= common_words
        context_words -= common_words
        
        if not question_words:
            return False
            
        # Calculate overlap
        overlap = len(question_words.intersection(context_words))
        relevance_score = overlap / len(question_words)
        
        print(f"🔍 Relevance score: {relevance_score:.2f} (threshold: 0.2)")
        return relevance_score >= 0.2

    def is_simple_greeting(self, question: str) -> bool:
        """Check if the question is a simple greeting"""
        question_clean = re.sub(r'[^\w\s]', '', question.lower()).strip()
        words = question_clean.split()
        
        # Single word greetings
        if len(words) == 1 and words[0] in self.simple_greetings:
            return True
            
        # Simple greeting patterns
        greeting_patterns = [
            r'^(salut|bonjour|bonsoir|hello|hi|hey|coucou)$',
            r'^(salut|bonjour|bonsoir|hello|hi|hey|coucou)\s+(comment|ça)\s+va',
            r'^comment\s+allez\s+vous',
            r'^comment\s+ça\s+va'
        ]
        
        for pattern in greeting_patterns:
            if re.match(pattern, question_clean):
                return True
                
        return False

    def query_local_llama(self, question: str, context: str = "") -> Optional[str]:
        """Query local Llama model with anti-hallucination measures"""
        
        if not self.current_model:
            return None
        
        # Use different prompts based on whether we have relevant context
        if context.strip() and self.is_context_relevant(question, context):
            prompt = f"""Tu es FrancoBot, un professeur de français. Réponds seulement avec les informations du cours ci-dessous.

CONTENU DU COURS:
{context}

QUESTION: {question}

RÈGLES IMPORTANTES:
- Utilise SEULEMENT les informations du contenu du cours ci-dessus
- Utilise ton connaissance générale du français pour expliquer, pas pour répondre
- Si le contenu ne répond pas à la question, dis "Je n'ai pas cette information dans le cours"
- Ne mélange pas avec tes connaissances générales
- Reste focalisé sur la question

RÉPONSE:"""
        else:
            prompt = f"""Tu es FrancoBot, un professeur de français.

QUESTION: {question}

Donne une réponse claire, expliquée et concise.

RÉPONSE:"""

        # Anti-hallucination settings - very conservative
        payload = {
            "model": self.current_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,      # Much lower - reduces creativity/hallucination
                "num_predict": 150,      # Shorter responses
                "top_p": 0.7,           # More focused
                "top_k": 10,            # Much more constrained vocabulary
                "repeat_penalty": 1.3,   # Strongly discourage repetition
                "stop": ["\n\nQUESTION:", "\n\nRÈGLES:", "CONTENU:", "###", "\n\n\n"]
            }
        }
        
        try:
            print(f"🦙 Génération avec {self.current_model} (mode anti-hallucination)...")
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                # Clean up response
                answer = re.sub(r'^RÉPONSE:\s*', '', answer, flags=re.IGNORECASE)
                answer = re.sub(r'\n\n+', '\n\n', answer)
                
                # Quality check - reject obviously bad responses
                if self.is_response_valid(answer, question):
                    print(f"✅ Réponse valide ({len(answer)} caractères)")
                    return answer
                else:
                    print("❌ Réponse rejetée (qualité insuffisante)")
                    return None
                    
            else:
                print(f"❌ Erreur Ollama {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("⏰ Timeout")
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
        
        return None

    def is_response_valid(self, response: str, question: str) -> bool:
        """Validate response quality to reject hallucinations"""
        if not response or len(response) < 5:
            return False
            
        # Reject responses with obvious hallucination indicators
        hallucination_indicators = [
            'giraffe', 'video game', 'resident evil', 'japan', 'flop', 'carton',
            'spectacle', 'adolescent', 'aujourd\'hui dans notre cours',
            'premièrement', 'ensuite', 'maintenant je vais vous poser'
        ]
        
        response_lower = response.lower()
        for indicator in hallucination_indicators:
            if indicator in response_lower:
                print(f"🚫 Hallucination détectée: '{indicator}'")
                return False
        
        # Reject overly long responses from small models (usually a bad sign)
        if "llama3.2:1b" in str(self.current_model) and len(response) > 300:
            print("🚫 Réponse trop longue pour ce modèle")
            return False
            
        # For greetings, expect simple responses
        if self.is_simple_greeting(question) and len(response) > 150:
            print("🚫 Réponse trop complexe pour un salut")
            return False
            
        return True

    def query(self, question: str, n_results: int = 2) -> str:
        """Main query method with intelligent context filtering"""
        
        # Handle simple greetings without document search
        if self.is_simple_greeting(question):
            print("👋 Salut simple détecté - pas de recherche documentaire")
            if self.use_local_llama and self.current_model:
                response = self.query_local_llama(question, "")
                if response:
                    return response
            return "Bonjour ! Je suis votre professeur de français. Comment puis-je vous aider aujourd'hui ?"
        
        # Get relevant context from documents
        context = ""
        try:
            if self.collection.count() > 0:
                query_embedding = self.encoder.encode([question]).tolist()
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=n_results
                )
                if results['documents'][0]:
                    raw_context = "\n".join(results['documents'][0])
                    # Only use context if it's actually relevant
                    if self.is_context_relevant(question, raw_context):
                        context = raw_context
                        print(f"📚 Contexte pertinent: {len(context)} caractères")
                    else:
                        print("📚 Contexte non pertinent - ignoré")
        except Exception as e:
            print(f"⚠️ Erreur recherche documents: {e}")
        
        # Try local Llama
        if self.use_local_llama and self.current_model:
            response = self.query_local_llama(question, context)
            if response:
                return response
            else:
                print("❌ Le modèle n'a pas pu générer une réponse valide")
                return "Je n'arrive pas à répondre clairement à cette question. Pouvez-vous la reformuler ?"
        else:
            return "Aucun modèle disponible. Veuillez installer et démarrer Ollama."

# Testing
if __name__ == "__main__":
    print("🦙 Assistant Français Anti-Hallucination")
    print("=" * 50)
    
    rag = FrenchRAG(use_local_llama=True)
    
    if os.path.exists('course_materials'):
        rag.load_documents('course_materials')
    else:
        print("📂 Aucun dossier course_materials trouvé")
    
    # Test different types of queries
    test_questions = [
        "Salut",
        "Bonjour comment ça va ?",
        "Explique-moi le passé composé",
        "Quelle est la différence entre être et avoir ?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")
        response = rag.query(question)
        print(response)
        print()
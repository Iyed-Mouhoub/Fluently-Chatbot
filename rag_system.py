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
        
        # Updated models list with Llama 3.1:8b as priority
        self.models = [
            "llama3.1:8b",      # 4.9GB - High quality model you have
            "llama3.2:3b",      # 2.0GB - Light but capable  
            "llama3.2:1b",      # 1.3GB - Very lightweight
            "tinyllama",        # 637MB - Ultra lightweight
            "phi3:mini",        # 2.3GB - Microsoft's efficient model
            "gemma:2b",         # 1.4GB - Google's lightweight model
            "mistral",          # Fallback option
        ]
        
        self.current_model = None
        self.check_ollama_models()
    
    def check_ollama_models(self):
        """Check which models are available in Ollama"""
        if not self.use_local_llama:
            return
            
        try:
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                print(f"🦙 Ollama détecté avec {len(available_models)} modèles")
                print(f"📋 Modèles disponibles: {', '.join(available_models)}")
                
                # Find the first available model from our preference list
                for model in self.models:
                    if any(model in available for available in available_models):
                        self.current_model = model
                        print(f"✅ Modèle sélectionné: {model}")
                        break
                
                if not self.current_model and available_models:
                    # Use the first available model
                    self.current_model = available_models[0]
                    print(f"🔋 Utilisation du modèle disponible: {self.current_model}")
                elif not self.current_model:
                    print("⚠️ Aucun modèle trouvé. Utilisation du mode éducatif.")
                    self.suggest_model_installation()
                    
            else:
                print("❌ Ollama non détecté")
                self.suggest_ollama_installation()
                
        except requests.exceptions.ConnectionError:
            print("🔌 Ollama non démarré")
            self.suggest_ollama_installation()
        except Exception as e:
            print(f"❌ Erreur Ollama: {e}")
            
    def suggest_ollama_installation(self):
        """Provide installation instructions"""
        print("""
🦙 INSTALLATION OLLAMA (Recommandé):

1. Télécharger Ollama:
   https://ollama.ai/download

2. Installer un modèle:
   ollama pull llama3.1:8b     # Haute qualité (vous l'avez déjà!)
   ollama pull llama3.2:1b     # 1.3GB - Très rapide
   ollama pull tinyllama       # 637MB - Ultra léger

3. Démarrer Ollama:
   ollama serve

4. Redémarrer l'application
        """)
    
    def suggest_model_installation(self):
        """Suggest installing a model"""
        print("""
💡 INSTALLER UN MODÈLE:

Dans votre terminal:
ollama pull llama3.1:8b    # Haute qualité (recommandé)
ollama pull llama3.2:1b    # Léger: seulement 1.3GB
ollama pull tinyllama      # Ultra léger: 637MB

Puis redémarrez cette application.
        """)

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
        text = re.sub(r'🔵|🎯', '', text)  # Remove emojis
        
        # Remove excessive whitespace and duplicates
        lines = []
        seen_lines = set()
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 15 and line not in seen_lines:
                lines.append(line)
                seen_lines.add(line)
        
        return '\n'.join(lines[:3])  # Limit to first 3 unique lines
        
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
        # Check if already loaded
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
        """Smart text chunking - larger chunks for Llama 3.1:8b"""
        if len(text) < 200:
            return [text]
            
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        # Larger chunks for the more capable Llama 3.1:8b model
        chunk_size = 1000 if "llama3.1:8b" in str(self.current_model) else 600
        
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

    def query_local_llama(self, question: str, context: str = "") -> Optional[str]:
        """Query local Llama model via Ollama - optimized for Llama 3.1:8b"""
        
        if not self.current_model:
            return None
            
        # Enhanced prompt for the more capable Llama 3.1:8b model
        if context:
            prompt = f"""Tu es un professeur de français expérimenté qui enseigne le niveau B2. Tu es patient, pédagogique et tu donnes des explications claires avec de bons exemples.

Contexte du cours:
{context[:600]}

Question de l'étudiant: {question}

Instructions:
- Réponds en français uniquement
- Sois pédagogique et structuré
- Donne des exemples concrets et pratiques  
- Si c'est de la grammaire, explique les règles clairement
- Si c'est du vocabulaire, montre l'usage en contexte
- Si c'est un dialogue, crée une conversation réaliste
- Adapte ton niveau au B2 (intermédiaire-avancé)

Réponse:"""
        else:
            prompt = f"""Tu es un professeur de français expérimenté qui enseigne le niveau B2.

Question de l'étudiant: {question}

Instructions:
- Réponds en français uniquement
- Sois pédagogique et structuré
- Donne des exemples concrets
- Adapte au niveau B2

Réponse:"""

        # Optimized settings for Llama 3.1:8b
        payload = {
            "model": self.current_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 400 if "llama3.1:8b" in str(self.current_model) else 200,
                "top_p": 0.9,
                "stop": ["\n\nQuestion:", "\n\nInstructions:", "Étudiant:"]
            }
        }
        
        try:
            print(f"🦙 Génération avec {self.current_model}...")
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60  # Longer timeout for larger model
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                if answer and len(answer) > 20:
                    print(f"✅ Réponse générée ({len(answer)} caractères)")
                    model_display = "Llama 3.1 8B" if "llama3.1:8b" in str(self.current_model) else self.current_model
                    return f"🦙 **{model_display}:**\n\n{answer}"
            else:
                print(f"❌ Erreur Ollama {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print("⏰ Timeout - Le modèle prend trop de temps")
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
        
        return None

    def get_educational_fallback(self, question: str, context: str = "") -> str:
        """Educational fallback when local model fails"""
        question_lower = question.lower()
        clean_context = self.clean_pdf_content(context) if context else ""
        
        # Installation guide for users
        if any(word in question_lower for word in ['salut', 'bonjour', 'hello', 'aide', 'aider', 'apprendre']):
            model_status = f"🦙 Modèle local: {self.current_model}" if self.current_model else "❌ Aucun modèle local détecté"
            
            installation_guide = ""
            if not self.current_model:
                installation_guide = """
**🔧 Pour activer l'IA locale:**
1. Vérifier: `ollama list` 
2. Si pas de modèles: `ollama pull llama3.1:8b`
3. Démarrer: `ollama serve`
4. Redémarrer cette app"""
            
            return f"""Salut ! 🇫🇷 Assistant Français B2 avec IA Llama

**Statut:** {model_status}

**Je peux vous aider avec :**
• 📚 **Grammaire** : conjugaisons, temps, syntaxe
• 🗣️ **Conversation** : dialogues pratiques
• 📖 **Vocabulaire** : mots, expressions, nuances
• 🎯 **Exercices** : pratique interactive

{installation_guide}

**Contenu de votre cours :**
{clean_context if clean_context else "Cours de français niveau B1/B2"}

**💡 Questions d'exemple :**
• "Explique-moi le passé composé"
• "Différence entre 'cependant' et 'pourtant'"
• "Créons un dialogue au restaurant"

**🎯 Avec Llama 3.1:8b :** Réponses détaillées et de haute qualité ! 🚀"""

        # Specific French topics with good examples
        elif 'passé composé' in question_lower or 'passé' in question_lower:
            return """⏰ **Le Passé Composé - Niveau B2**

**🏗️ Formation :** AUXILIAIRE + PARTICIPE PASSÉ

**Auxiliaires :**
• **AVOIR** (majorité) : J'ai mangé, elle a fini, nous avons vu
• **ÊTRE** (16 verbes) : Je suis allé(e), elle est née, ils sont morts

**⚖️ Accords du participe passé :**
• Avec ÊTRE : accord avec le sujet → "Elle est partie"
• Avec AVOIR : accord avec COD antéposé → "La lettre qu'il a écrite"

**🎯 Les 16 verbes avec ÊTRE :**
Aller, venir, entrer, sortir, arriver, partir, monter, descendre, naître, mourir, rester, tomber, retourner, passer, devenir + tous les pronominaux

**✏️ Exercice rapide :**
1. Marie ____ (partir) hier → est partie
2. Les livres que j'____ (lire) → ai lus  
3. Nous ____ (se réveiller) tôt → nous sommes réveillé(e)s

**💡 Astuce :** "DR & MRS VANDERTRAMP" pour mémoriser les verbes avec être !"""

        elif any(word in question_lower for word in ['dialogue', 'conversation', 'restaurant', 'café']):
            return """🗣️ **Dialogue au Restaurant - Niveau B2**

**Situation :** Dîner dans un restaurant français

**Serveur :** Bonsoir, avez-vous réservé ?
**Client :** Oui, une table pour deux au nom de Martin.
**Serveur :** Parfait, suivez-moi. Voici la carte.
**Client :** Merci. Que me conseillez-vous comme entrée ?
**Serveur :** Je vous recommande la salade de chèvre chaud, elle est excellente.
**Client :** D'accord, et comme plat principal ?
**Serveur :** Le saumon grillé aux légumes de saison est très apprécié.
**Client :** Parfait. Pour les boissons ?
**Serveur :** Puis-je vous suggérer un vin blanc sec ?
**Client :** Excellente idée. L'addition, s'il vous plaît.

**🎯 Expressions utiles :**
• "Que me conseillez-vous ?" (demander conseil)
• "Je vous recommande..." (conseiller)
• "Puis-je vous suggérer..." (suggestion polie)
• "C'est délicieux !" (compliment)

**💡 Registre B2 :** Utilisez le vouvoiement et les formules de politesse !"""

        else:
            return f"""🎓 **Assistant Français B2** (Mode éducatif)

**Votre question :** "{question}"

**🦙 Statut IA locale :** {f"Disponible ({self.current_model})" if self.current_model else "Non configuré"}

**Contenu trouvé :**
{clean_context if clean_context else "Contenu général de français B2"}

**💡 Sujets que je maîtrise bien :**
• Grammaire avancée (subjonctif, concordance des temps)
• Vocabulaire nuancé (registres de langue)
• Expression écrite et orale
• Culture et civilisation françaises

**🎯 Questions efficaces :**
• "Explique-moi [règle grammaticale]"
• "Différence entre [mot1] et [mot2]"
• "Comment exprimer [idée] poliment"
• "Exercice sur [sujet précis]"

**🚀 Modèle recommandé :** Llama 3.1:8b pour des réponses détaillées !

Quelle question précise puis-je traiter pour vous ? 🇫🇷"""

    def query(self, question: str, n_results: int = 3) -> str:
        """Main query method"""
        # Get context from documents
        try:
            query_embedding = self.encoder.encode([question]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            context = "\n".join(results['documents'][0]) if results['documents'][0] else ""
        except:
            context = ""
        
        # Try local Llama first
        if self.use_local_llama and self.current_model:
            llama_response = self.query_local_llama(question, context)
            if llama_response:
                return llama_response
            else:
                print("🔄 Modèle local indisponible, mode éducatif activé...")
        
        # Fallback to educational responses
        return self.get_educational_fallback(question, context)

# Usage and testing
if __name__ == "__main__":
    print("🦙 Assistant Français avec Llama 3.1:8b")
    print("=" * 50)
    
    # Initialize with local Llama
    rag = FrenchRAG(use_local_llama=True)
    
    # Load documents
    if os.path.exists('course_materials'):
        rag.load_documents('course_materials')
    
    # Test queries
    test_questions = [
        "salut je veux apprendre le français",
        "Explique-moi le passé composé avec des exemples",
        "Créons un dialogue au restaurant"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")
        response = rag.query(question)
        print(response)
        print()
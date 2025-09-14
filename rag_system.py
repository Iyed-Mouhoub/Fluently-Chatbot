#!/usr/bin/env python3

"""
Enhanced FrenchRAG with FAISS retriever, OpenRouter API support, and optimized token management
"""

import os
import re
import time
import json
import requests
import faiss
from typing import Optional
from sentence_transformers import SentenceTransformer
import PyPDF2

class FrenchRAG:
    def __init__(self, use_openrouter=True, openrouter_api_key=None, index_path="french_course.index"):
        # Embedding model
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = self.encoder.get_sentence_embedding_dimension()

        # FAISS index (L2 distance; we normalize embeddings so it works as cosine)
        self.index_path = index_path
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"📚 FAISS index loaded ({self.index.ntotal} vectors)")
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            print("📚 New FAISS index created (empty)")

        self.docs = []  # keep raw texts
        self.ids = []   # keep mapping ids

        # OpenRouter configuration
        self.use_openrouter = use_openrouter
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

        # Enhanced OpenRouter models with token limits
        self.openrouter_models = [
            "anthropic/claude-3-haiku",              # Good balance, 4096 tokens
            "meta-llama/llama-3.1-8b-instruct:free", # Free, 8192 tokens
            "google/gemma-2-9b-it:free",             # Free, 8192 tokens
            "microsoft/phi-3-medium-128k-instruct:free", # Free, high context
            "microsoft/phi-3-mini-128k-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free"
        ]

        # Model-specific token configurations
        self.model_token_limits = {
            "anthropic/claude-3-haiku": 800,
            "meta-llama/llama-3.1-8b-instruct:free": 600,
            "google/gemma-2-9b-it:free": 600,
            "microsoft/phi-3-medium-128k-instruct:free": 800,
            "microsoft/phi-3-mini-128k-instruct:free": 500,
            "mistralai/mistral-7b-instruct:free": 500,
            "meta-llama/llama-3.2-3b-instruct:free": 500
        }

        self.current_model = None

        if self.use_openrouter:
            self.setup_openrouter()
        else:
            # Fallback to local Ollama setup
            self.setup_ollama()

        # Greetings
        self.simple_greetings = {"salut", "bonjour", "bonsoir", "hello", "hi", "hey", "coucou"}

    def get_max_tokens_for_model(self, model_name: str) -> int:
        """Get the optimal max_tokens setting for a specific model"""
        return self.model_token_limits.get(model_name, 2000)  # Default to 2000

    # ---------------------- OpenRouter setup ----------------------
    def setup_openrouter(self):
        if not self.openrouter_api_key:
            print("❌ OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable or pass it to constructor.")
            print("💡 Get your API key from: https://openrouter.ai/")
            return

        try:
            # Test connection with a simple request
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            # Use the first available model as default
            self.current_model = self.openrouter_models[0]
            print(f"🌐 OpenRouter configured with model: {self.current_model}")
            print(f"🔢 Max tokens for this model: {self.get_max_tokens_for_model(self.current_model)}")
            print(f"🔑 API Key: {'*' * (len(self.openrouter_api_key)-8) + self.openrouter_api_key[-4:]}")

        except Exception as e:
            print(f"❌ Error setting up OpenRouter: {e}")

    # ---------------------- Ollama fallback ----------------------
    def setup_ollama(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_models = [
            "llama3.2:1b", "llama3.1:8b", "llama3.2:3b",
            "tinyllama", "phi3:mini", "gemma:2b", "mistral"
        ]

        self.check_ollama_models()

    def check_ollama_models(self):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                print(f"🦙 Ollama detected with {len(models)} models")

                for m in self.ollama_models:
                    if any(m in av for av in models):
                        self.current_model = m
                        break

                if not self.current_model and models:
                    self.current_model = models[0]
            else:
                print("❌ Ollama not detected")

        except requests.exceptions.ConnectionError:
            print("🔌 Ollama not running")
        except Exception as e:
            print(f"❌ Ollama error: {e}")

    # ---------------------- PDF cleaning (unchanged) ----------------------
    def clean_pdf_content(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r'--- Page \d+ ---', '', text)
        text = re.sub(r'@daily\.french_', '', text)
        text = re.sub(r'Prof : Labed Nada', '', text)
        text = re.sub(r'Niveau intermédiaire B1/B2', '', text)
        text = re.sub(r'ATELIER DE.*?CONVERSATION', 'Atelier de conversation', text)
        text = re.sub(r'[🔵🎯]', '', text)

        lines, seen = [], set()
        for line in text.splitlines():
            line = line.strip()
            if line and len(line) > 15 and line not in seen:
                lines.append(line)
                seen.add(line)

        return "\n".join(lines[:3])

    def extract_pdf_text(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    cleaned = self.clean_pdf_content(page_text)
                    if cleaned:
                        text += f"Page {i+1}:\n{cleaned}\n\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

        return self.clean_pdf_content(text)

    # ---------------------- Load docs into FAISS (unchanged) ----------------------
    def load_documents(self, folder_path: str):
        if self.index.ntotal > 0:
            print(f"📚 Documents already loaded ({self.index.ntotal} chunks)")
            return

        docs = []
        ids = []

        print("📄 Processing documents...")

        for fn in os.listdir(folder_path):
            fp = os.path.join(folder_path, fn)
            content = ""

            if fn.endswith(".pdf"):
                print(f"📄 Converting {fn}...")
                content = self.extract_pdf_text(fp)
            elif fn.endswith(".txt"):
                with open(fp, "r", encoding="utf-8") as f:
                    content = self.clean_pdf_content(f.read())

            if not content.strip():
                continue

            for i, chunk in enumerate(self.smart_chunk_text(content)):
                if len(chunk.strip()) > 30:
                    docs.append(chunk.strip())
                    ids.append(f"{fn}_{i}")

        if docs:
            print(f"📊 Adding {len(docs)} chunks to FAISS...")
            emb = self.encoder.encode(docs, normalize_embeddings=True)
            self.index.add(emb)
            self.docs.extend(docs)
            self.ids.extend(ids)

            faiss.write_index(self.index, self.index_path)
            print("✅ Index saved")

    # ---------------------- Chunking (unchanged) ----------------------
    def smart_chunk_text(self, text: str):
        if len(text) < 200:
            return [text]

        paras = text.split("\n\n")
        chunks, cur = [], ""
        limit = 500  # Reasonable limit for most models

        for p in paras:
            p = p.strip()
            if not p:
                continue

            if len(cur + p) > limit:
                if cur:
                    chunks.append(cur.strip())
                cur = p
            else:
                cur += ("\n\n" if cur else "") + p

        if cur:
            chunks.append(cur.strip())

        return chunks

    # ---------------------- Helpers (unchanged) ----------------------
    def is_context_relevant(self, question: str, ctx: str) -> bool:
        if not ctx or len(ctx.strip()) < 20:
            return False

        qw = set(re.findall(r"\b\w+\b", question.lower()))
        cw = set(re.findall(r"\b\w+\b", ctx.lower()))

        stop = {"le","la","les","un","une","des","de","du","et","à","il","elle","dans","pour","avec","sur","par"}
        qw -= stop
        cw -= stop

        if not qw:
            return False

        overlap = len(qw & cw) / len(qw)
        print(f"🔍 Relevance score: {overlap:.2f} (thr=0.2)")
        return overlap >= 0.2

    def is_simple_greeting(self, q: str) -> bool:
        clean = re.sub(r"[^\w\s]", "", q.lower()).strip()

        if clean in self.simple_greetings:
            return True

        pats = [
            r"^(salut|bonjour|bonsoir|hello|hi|hey|coucou)\s*(comment|ça)?",
            r"^comment\s+allez\s+vous",
            r"^comment\s+ça\s+va"
        ]

        return any(re.match(p, clean) for p in pats)

    # ---------------------- Enhanced OpenRouter Generation ----------------------
    def query_openrouter(self, question: str, ctx: str = "") -> Optional[str]:
        if not self.openrouter_api_key or not self.current_model:
            return None

        # Get optimal token limit for current model
        max_tokens = self.get_max_tokens_for_model(self.current_model)

        # Construct the prompt
        if ctx.strip() and self.is_context_relevant(question, ctx):
            system_prompt = """Tu es FrancoBot, un professeur de français expérimenté a l'institut Fluently. Fluently offre cette formation pour de differents niveaux. Il s'agit de 20 seances qui durent 2h, dans chaque seance, on entamme des expressions familieres, grammaires, partie orale, et audio/video. Utilise uniquement les informations du contenu de cours fourni pour répondre aux questions. Sois précis, pédagogique et concis. Donne une explication claire avec un exemple."""

            user_prompt = f"""CONTENU DU COURS:
{ctx}

QUESTION: {question}

Réponds en utilisant uniquement les informations du cours ci-dessus. Fournit une explication claire et concise."""
        else:
            system_prompt = "Tu es FrancoBot, un professeur de français expérimenté. Réponds de manière pédagogique et concise avec un exemple."
            user_prompt = question

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Optional: for analytics
            "X-Title": "French RAG Assistant"  # Optional: for analytics
        }

        payload = {
            "model": self.current_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,  # Dynamic token limit based on model
            "temperature": 0.1,
            "top_p": 0.7,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.1
        }

        try:
            print(f"🌐 Generating with OpenRouter ({self.current_model}, max_tokens: {max_tokens})...")

            response = requests.post(
                self.openrouter_url,
                headers=headers,
                json=payload,
                timeout=45  # Increased timeout for longer responses
            )

            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    answer = data['choices'][0]['message']['content'].strip()
                    if self.is_response_valid(answer, question):
                        print(f"✅ Generated response: {len(answer)} characters")
                        return answer
                    else:
                        print("⚠️ Response validation failed")
                        return None
                else:
                    print("❌ No choices in OpenRouter response")
            else:
                error_msg = response.text
                print(f"❌ OpenRouter API error {response.status_code}: {error_msg}")

        except requests.exceptions.Timeout:
            print("⏰ OpenRouter request timeout (increased to 45s)")
        except Exception as e:
            print(f"❌ OpenRouter error: {e}")

        return None

    # ---------------------- Enhanced Ollama Generation ----------------------
    def query_local_llama(self, question: str, ctx: str = "") -> Optional[str]:
        if not self.current_model:
            return None

        if ctx.strip() and self.is_context_relevant(question, ctx):
            prompt = f"""Tu es FrancoBot, un professeur de français a l'institut Fluently. Réponds uniquement avec les infos du cours. Sois clair et concis dans tes explications. Sois clair, concis et pertinent.

CONTENU DU COURS:
{ctx}

QUESTION: {question}

RÉPONSE:"""
        else:
            prompt = f"Tu es FrancoBot, un professeur de français. Réponds de manière pédagogique et concise avec un exemple.\n\nQUESTION: {question}\n\nRÉPONSE:"

        payload = {
            "model": self.current_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1, 
                "num_predict": 2000,  # Increased for longer responses
                "top_p": 0.7,
                "top_k": 10, 
                "repeat_penalty": 1.3,
                "stop": ["\n\nQUESTION:", "\n\nRÈGLES:", "CONTENU:", "###", "\n\n\n"]
            }
        }

        try:
            print(f"🦙 Generating with Ollama ({self.current_model}, max_predict: 2000)...")
            r = requests.post(self.ollama_url, json=payload, timeout=30)

            if r.status_code == 200:
                ans = r.json().get("response", "").strip()
                ans = re.sub(r"^RÉPONSE:\s*", "", ans, flags=re.I)

                if self.is_response_valid(ans, question):
                    print(f"✅ Generated response: {len(ans)} characters")
                    return ans
                else:
                    return None
            else:
                print(f"❌ Ollama error {r.status_code}")

        except requests.exceptions.Timeout:
            print("⏰ Ollama timeout")
        except Exception as e:
            print(f"❌ Ollama error: {e}")

        return None

    def is_response_valid(self, resp: str, q: str) -> bool:
        if not resp or len(resp) < 5:
            return False

        # Remove overly restrictive validation for longer responses
        bad = ['giraffe','video game','resident evil','japan']
        if any(b in resp.lower() for b in bad):
            return False

        # Allow longer responses for greetings too
        if self.is_simple_greeting(q) and len(resp) > 500:
            return False

        return True

    # ---------------------- Main query method ----------------------
    def query(self, question: str, n_results: int = 3) -> str:  # Increased n_results for more context
        if self.is_simple_greeting(question):
            # Handle simple greetings
            if self.use_openrouter:
                r = self.query_openrouter(question, "")
            else:
                r = self.query_local_llama(question, "")

            if r:
                return r

            return "Bonjour ! Je suis votre professeur de français. Comment puis-je vous aider aujourd'hui ?"

        context = ""

        try:
            # Retrieve relevant context from FAISS
            if self.index.ntotal > 0:
                qvec = self.encoder.encode([question], normalize_embeddings=True)
                D, I = self.index.search(qvec, n_results)

                cands = [self.docs[i] for i in I[0] if i >= 0 and i < len(self.docs)]
                raw_ctx = "\n".join(cands)

                if self.is_context_relevant(question, raw_ctx):
                    context = raw_ctx
                    print(f"📚 Relevant context: {len(context)} characters")
                else:
                    print("📚 Context ignored (not relevant)")

        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")

        # Generate response
        if self.use_openrouter:
            ans = self.query_openrouter(question, context)
        else:
            ans = self.query_local_llama(question, context)

        return ans or "Je n'arrive pas à répondre clairement à cette question. Pouvez-vous la reformuler ?"

# ---------------------- Test ----------------------
if __name__ == "__main__":
    print("🌐 Enhanced French Assistant with OpenRouter/FAISS")
    print("🔢 Optimized for longer, more detailed responses")

    # Initialize with OpenRouter (set your API key as environment variable)  
    rag = FrenchRAG(use_openrouter=True)

    if os.path.exists("course_materials"):
        rag.load_documents("course_materials")
    else:
        print("📂 No course_materials folder found")

    # Test queries
    test_queries = [
        "Salut",
        "Explique-moi le passé composé avec des exemples",
        "Quelle est la différence entre être et avoir ? Donne-moi plusieurs exemples."
    ]

    for q in test_queries:
        print("\n" + "="*40)
        print("Q:", q)
        print("A:", rag.query(q))

import os
import gradio as gr
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import re
import requests
import json
import platform
from collections import defaultdict
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("api_key")
INDEX_NAME = "document-analysis"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Embedding model

# Configure Ollama API - use localhost as Ollama should be running locally
OLLAMA_API_BASE = "http://localhost:11434/api"
OLLAMA_MODEL = "phi3:mini"  # Ollama model name

# Detect OS for better error messages
is_windows = platform.system() == "Windows"
is_wsl = "microsoft" in platform.uname().release.lower() or "wsl" in platform.uname().release.lower()

# Company to PDF mapping - maps company names to their respective PDF files
COMPANY_PDF_MAPPING = {
    "enersys": ["EnerSys-2023-10K.pdf", "EnerSys-2017-10K.pdf"],
    "amazon": ["Amazon10k2022.pdf"],
    "apple": ["Apple_10-K-2021.pdf"],
    "nvidia": ["Nvidia.pdf"],
    "tesla": ["Tesla.pdf"],
    "lockheed": ["Lockheed_martin_10k.pdf"],
    "advent": ["Advent_Technologies_2022_10K.pdf"],
    "transdigm": ["TransDigm-2022-10K.pdf"]
}

def check_ollama_available():
    """Check if Ollama is available by sending a request to list models"""
    try:
        print(f"Trying to connect to Ollama at {OLLAMA_API_BASE}...")
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            if OLLAMA_MODEL in available_models:
                print(f"Ollama is available with {OLLAMA_MODEL}")
                return True
            else:
                print(f"Warning: {OLLAMA_MODEL} not found in Ollama. Available models: {available_models}")
                return False
        else:
            print(f"Ollama API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        return False

# Initialize Pinecone connection and models
def init_pinecone_and_embeddings():
    """Initialize Pinecone and embedding model"""
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    # Check if Ollama is available
    ollama_available = check_ollama_available()
    
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists(MODELS_DIR):
            print(f"Creating models directory at {MODELS_DIR}...")
            os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Check if embedding model exists
        embedding_model_path = os.path.join(MODELS_DIR, "embedding_model")
        if os.path.exists(embedding_model_path):
            print("Loading embedding model from disk...")
            embeddings_model = SentenceTransformer(embedding_model_path)
        else:
            print("Embedding model not found locally, downloading...")
            embeddings_model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Save the embedding model for future use
            try:
                print(f"Saving embedding model to {embedding_model_path}...")
                embeddings_model.save(embedding_model_path)
                print("Embedding model saved successfully!")
            except Exception as e:
                print(f"Warning: Failed to save embedding model: {e}")
                # Continue without saving
    except Exception as e:
        print(f"Warning: Error with models directory: {e}")
        print("Continuing with in-memory embedding model only...")
        embeddings_model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Models loaded successfully!")
    return pc, index, embeddings_model, ollama_available

# Initialize Pinecone and embedding model
pc, index, embeddings_model, ollama_available = init_pinecone_and_embeddings()

def semantic_search(query, namespace, top_k=3, score_threshold=0.4, max_results=2):
    """
    Perform semantic search with score filtering and result limiting.

    Args:
        query: Query string
        namespace: Namespace to search within
        top_k: Number of candidates to retrieve from Pinecone
        score_threshold: Minimum similarity score
        max_results: Max results to return after filtering

    Returns:
        Dictionary of results grouped by document_id
    """
    query_embedding = embeddings_model.encode(query, convert_to_numpy=True).tolist()

    response = index.query(
        vector=query_embedding,
        namespace=namespace,
        top_k=top_k,
        include_metadata=True
    )

    # Filter and sort matches
    filtered_matches = sorted(
        [m for m in response.get("matches", []) if m["score"] >= score_threshold],
        key=lambda x: x["score"],
        reverse=True
    )[:max_results]

    # Group by document_id
    grouped_results = defaultdict(list)
    for match in filtered_matches:
        metadata = match.get("metadata", {})
        doc_id = metadata.get("document_id", f"unknown-{match['id']}")
        grouped_results[doc_id].append({
            "score": match["score"],
            "text": metadata.get("text", "No preview available")
        })

    return grouped_results

def format_documents(results_by_doc: Dict) -> str:
    """Format retrieved documents for inclusion in the context."""
    if not results_by_doc:
        return "No relevant information found."
    
    formatted_text = ""
    for doc_id, matches in results_by_doc.items():
        for match in matches:
            formatted_text += f"\n--- From document: {doc_id} (score: {match['score']:.2f}) ---\n"
            formatted_text += match["text"].strip() + "\n"
    
    return formatted_text

def generate_answer_with_ollama(context: str, query: str) -> str:
    """Generate an answer using Ollama's Phi-3 model based on context and query."""
    prompt = f"""You are a helpful assistant that answers questions about financial reports and SEC filings.
Answer the user's question based solely on the provided context.
If you don't know the answer based on the context, just say so.
Don't make up information and cite the relevant parts of the document when possible.

Context:
{context}

User Question:
{query}

Answer:"""
    
    # Make API call to Ollama
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            },
            timeout=120  # Increase timeout to 120 seconds
        )
        
        if response.status_code == 200:
            return response.json().get("response", "Error generating response.")
        else:
            return f"Error: Ollama returned status code {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def process_query(company: str, query: str) -> str:
    """Process a query for a specific company."""
    if not query.strip():
        return "Please enter a query."
    
    # Search Pinecone for relevant context
    results = semantic_search(
        query=query,
        namespace=company.lower(),
        top_k=3,
        score_threshold=0.4,
        max_results=2
    )
    
    # Format retrieved documents
    context = format_documents(results)
    
    # Generate answer using Phi-3 via Ollama 
    if "No relevant information found" in context:
        answer = "I couldn't find relevant information to answer your question in the selected document."
    else:
        try:
            answer = generate_answer_with_ollama(context, query)
        except Exception as e:
            answer = f"Error: Unable to connect to Ollama. Please make sure Ollama is installed and running with the phi3:mini model."
    
    return answer

def create_interface():
    """Create Gradio interface for the PDF QA system."""
    # Create list of PDF options for dropdown
    pdf_options = []
    for company, pdfs in COMPANY_PDF_MAPPING.items():
        for pdf in pdfs:
            pdf_options.append(f"{company.capitalize()}: {pdf}")
    
    # Define custom CSS for a more modern and professional UI
    css = """
    :root {
        --primary-color: #2c6fdb;
        --primary-hover: #245bb9;
        --text-color: #333333;
        --light-text: #666666;
        --bg-color: #f5f5f5;
        --card-bg: #ffffff;
        --border-color: #dddddd;
        --radius: 3px;
        --shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    body {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: Arial, Helvetica, sans-serif;
        font-size: 16px;
    }
    
    .container {
        max-width: 1500px;
        margin: auto;
        padding: 1.5rem;
    }
    
    .header {
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .title {
        color: var(--text-color);
        font-size: 2.2rem;
        font-weight: normal;
        margin-bottom: 0.75rem;
        font-family: Arial, Helvetica, sans-serif;
    }
    
    .subtitle {
        color: var(--light-text);
        font-size: 1.25rem;
        font-weight: normal;
        font-family: Arial, Helvetica, sans-serif;
    }
    
    .card {
        background-color: var(--card-bg);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        width: 90%;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Fix for dropdown and textbox width */
    .input-box, .output-box {
        width: 95% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: normal;
        margin-bottom: 1rem;
        color: var(--text-color);
    }
    
    .warning {
        background-color: #fff7ed;
        border-left: 3px solid #f97316;
        color: #333333;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border-radius: 2px;
        font-size: 1rem;
    }
    
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        color: var(--light-text);
        font-size: 0.9rem;
        border-top: 1px solid var(--border-color);
    }
    
    /* Gradio component overrides */
    .gradio-container {
        max-width: 100% !important;
    }
    
    /* Input field styling */
    input, textarea, select {
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        padding: 0.75rem !important;
    }
    
    .answer-box {
        border-left: 2px solid #cccccc;
        background-color: #fafafa;
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        padding: 1rem !important;
    }
    
    button.primary {
        background-color: var(--primary-color) !important;
        font-weight: normal !important;
        font-family: Arial, Helvetica, sans-serif !important;
        width: 60% !important;
        margin: 0 auto !important;
        display: block !important;
        font-size: 1.2rem !important;
        padding: 0.75rem 1.5rem !important;
    }
    
    button.primary:hover {
        background-color: var(--primary-hover) !important;
    }
    
    /* Labels */
    label span {
        font-weight: normal !important;
        font-size: 1.1rem !important;
        font-family: Arial, Helvetica, sans-serif !important;
        margin-bottom: 0.5rem !important;
        display: inline-block !important;
    }
    """
    
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes="container"):
            # Header section
            with gr.Column(elem_classes="header"):
                gr.HTML('<div class="title"> Financial Reports QA</div>')
                gr.HTML('<div class="subtitle">Extract valuable information from financial reports</div>')
            
            # Warning message if Ollama is not available
            if not ollama_available:
                if is_windows:
                    instructions = (
                        '<div class="warning">⚠️ <strong>Warning:</strong> Ollama is not available. '
                        'Make sure Ollama is running by opening PowerShell and running: <pre>ollama serve</pre>'
                        'And ensure the phi3:mini model is installed with: <pre>ollama pull phi3:mini</pre></div>'
                    )
                elif is_wsl:
                    instructions = (
                        '<div class="warning">⚠️ <strong>Warning:</strong> Ollama is not available. '
                        'Install Ollama in WSL using: <pre>curl -fsSL https://ollama.com/install.sh | sh</pre>'
                        'Then start it with: <pre>ollama serve</pre>'
                        'And pull the model: <pre>ollama pull phi3:mini</pre></div>'
                    )
                else:
                    instructions = (
                        '<div class="warning">⚠️ <strong>Warning:</strong> Ollama is not available. '
                        'Make sure to install Ollama, run it with the "ollama serve" command, '
                        'and pull the phi3:mini model with "ollama pull phi3:mini".</div>'
                    )
                gr.HTML(instructions)
            
            # Input card
            with gr.Column(elem_classes="card"):
                gr.HTML('<div class="section-title">📁 Select Report & Ask Question</div>')
                
                with gr.Row(equal_height=True):
                    pdf_dropdown = gr.Dropdown(
                        choices=pdf_options, 
                        label="Financial Report",
                        value=pdf_options[0] if pdf_options else None,
                        container=True,
                        interactive=True,
                        elem_classes="input-box"
                    )
                
                query_input = gr.Textbox(
                    lines=3, 
                    label="Your Question",
                    placeholder="Example: What were the total R&D expenses for 2022?",
                    container=True,
                    elem_classes="input-box"
                )
                        
                submit_btn = gr.Button(
                    "Get Answer", 
                    variant="primary",
                    elem_classes="primary",
                    size="lg"
                )
            
            # Answer card
            with gr.Column(elem_classes="card"):
                gr.HTML('<div class="section-title">Response based on the selected report</div>')
                answer_output = gr.Textbox(
                    label="Answer", 
                    lines=10,
                    show_copy_button=True,
                    container=True,
                    elem_classes="answer-box output-box"
                )
            
            # Footer
            gr.HTML('<div class="footer">Powered by Pinecone Vector Database and Phi-3 AI | <strong>Financial Reports QA System</strong> v1.0</div>')
            
            # Handle submission
            def on_submit(pdf_selection, query):
                if not pdf_selection:
                    return "Please select a financial report first."
                    
                company = pdf_selection.split(":")[0].lower()
                answer = process_query(company, query)
                return answer
            
            submit_btn.click(
                fn=on_submit,
                inputs=[pdf_dropdown, query_input],
                outputs=answer_output
            )
    
    return demo

# Launch the Gradio app
if __name__ == "__main__":
    demo = create_interface()
    # Set share=False for production use
    demo.launch(share=True)
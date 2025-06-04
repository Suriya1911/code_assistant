"""
Canadian Criminal Code Assistant
Fixed version with proper syntax and enhanced analytics - Dark banner removed
"""

import streamlit as st
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Import required libraries
try:
    import google.generativeai as genai
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from dotenv import load_dotenv
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.code("pip install google-generativeai python-dotenv langchain langchain-community sentence-transformers faiss-cpu pandas plotly")
    st.stop()

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Canadian Criminal Code Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium dark theme CSS - Fixed banner issue
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Orbitron:wght@400;500;600;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #0a1128 100%);
        font-family: 'Poppins', sans-serif;
        color: #e2e8f0;
        min-height: 100vh;
        overflow-x: hidden;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main .block-container {
        padding: 1rem 3rem 0 3rem;
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
        min-height: auto;
    }
    
    .main-header {
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(15,15,35,0.95) 0%, rgba(26,26,46,0.9) 25%, rgba(22,33,62,0.85) 50%, rgba(15,52,96,0.9) 75%, rgba(10,17,40,0.95) 100%);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(120,119,198,0.2);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.05), inset 0 1px 0 rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .header-content { flex: 1; text-align: center; }
    .header-metrics { display: flex; gap: 1rem; align-items: center; }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #7877c6 0%, #ff77c6 25%, #77c6ff 50%, #c677ff 75%, #7877c6 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.8rem;
        line-height: 1.1;
        letter-spacing: -0.02em;
        animation: rainbowShift 6s ease-in-out infinite;
    }
    
    @keyframes rainbowShift {
        0%, 100% { background-position: 0% 50%; }
        25% { background-position: 100% 0%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #a0aec0;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    .header-metric {
        background: linear-gradient(135deg, rgba(15,15,35,0.8) 0%, rgba(26,26,46,0.7) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(120,119,198,0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        min-width: 120px;
        transition: all 0.3s ease;
    }
    
    .header-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        border-color: rgba(255,119,198,0.5);
    }
    
    .header-metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.3rem 0;
        font-family: 'Orbitron', monospace;
        line-height: 1;
    }
    
    .header-metric-label {
        font-size: 0.7rem;
        color: #a0aec0;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .header-metric.questions .header-metric-value { color: #10b981; }
    .header-metric.total .header-metric-value { color: #f59e0b; }
    .header-metric.database .header-metric-value { color: #3b82f6; }
    
    .tab-section {
        margin: 0;
        background: linear-gradient(135deg, rgba(15,15,35,0.9) 0%, rgba(26,26,46,0.85) 25%, rgba(22,33,62,0.8) 50%, rgba(15,52,96,0.85) 75%, rgba(10,17,40,0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(120,119,198,0.2);
        border-radius: 28px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(15,15,35,0.9) 0%, rgba(26,26,46,0.8) 100%);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(120,119,198,0.3);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        gap: 1rem;
        display: flex;
        justify-content: space-between;
        width: 100%;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 16px;
        color: #a0aec0;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 1rem 1.5rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 1px;
        flex: 1;
        text-align: center;
        min-width: 0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7877c6 0%, #ff77c6 100%);
        color: #ffffff;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(120,119,198,0.4), 0 0 20px rgba(255,119,198,0.3);
        transform: translateY(-2px);
    }
    
    .definition-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 4px solid #10b981;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        backdrop-filter: blur(15px);
        box-shadow: 0 15px 35px rgba(16, 185, 129, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    .definition-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #6ee7b7;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .definition-title::before {
        content: '◆';
        font-size: 1.2rem;
        color: #10b981;
    }
    
    .definition-content {
        color: #d1fae5;
        font-size: 1.1rem;
        line-height: 1.7;
        margin: 0;
    }
    
    .stTextInput > div > div > input {
        background: transparent;
        border: 2px solid #ffffff;
        border-radius: 8px;
        color: #ffffff;
        font-size: 1.1rem;
        padding: 1.8rem 1rem 1.8rem 1rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 400;
        transition: all 0.3s ease;
        height: 90px !important;
        line-height: 1.3;
        box-sizing: border-box;
        vertical-align: top;
        display: flex;
        align-items: center;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #7877c6;
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #a0aec0;
        opacity: 0.8;
        font-size: 1rem;
    }
    
    .stTextInput > div > div {
        min-height: 94px !important;
        height: 94px !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .stTextInput > div {
        margin-bottom: 1rem;
        min-height: 94px;
    }
    
    .stTextInput {
        margin-bottom: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(120,119,198,0.8);
        background: rgba(15,15,35,0.95);
        box-shadow: 0 0 0 4px rgba(120,119,198,0.2), 0 15px 35px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
        transform: translateY(-2px);
        color: #ffffff;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8;
        opacity: 0.7;
        font-size: 0.9rem;
    }
    
    .response-container {
        background: linear-gradient(135deg, rgba(15,15,35,0.95) 0%, rgba(26,26,46,0.9) 25%, rgba(22,33,62,0.85) 50%, rgba(15,52,96,0.9) 75%, rgba(10,17,40,0.95) 100%);
        backdrop-filter: blur(25px);
        border: 1px solid rgba(120,119,198,0.2);
        border-radius: 28px;
        padding: 3.5rem;
        margin: 3rem 0;
        border-left: 4px solid #7877c6;
        box-shadow: 0 25px 50px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    
    .response-content {
        font-size: 1.15rem;
        line-height: 1.9;
        color: #e2e8f0;
        font-weight: 400;
    }
    
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 2rem 0 1.5rem 0;
        text-align: center;
        color: transparent;
        background: linear-gradient(135deg, #7877c6 0%, #ff77c6 100%);
        background-clip: text;
        -webkit-background-clip: text;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7877c6 0%, #ff77c6 50%, #77c6ff 100%);
        background-size: 200% 200%;
        color: #ffffff;
        border: none;
        border-radius: 16px;
        padding: 1.2rem 2.5rem;
        font-weight: 700;
        font-size: 1rem;
        font-family: 'Poppins', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 10px 30px rgba(120,119,198,0.4), inset 0 1px 0 rgba(255,255,255,0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 20px 50px rgba(120,119,198,0.6), 0 0 30px rgba(255,119,198,0.4);
    }
    
    .educational-notice {
        margin: 1rem 0 0 0;
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, rgba(15,15,35,0.8) 0%, rgba(26,26,46,0.7) 100%);
        backdrop-filter: blur(15px);
        border: 2px solid #ef4444;
        border-left: 6px solid #dc2626;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(239,68,68,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .educational-notice::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #ef4444, transparent);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    .educational-text {
        color: #ffffff;
        font-size: 1rem;
        line-height: 1.7;
        margin: 0;
        font-weight: 600;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    .educational-text strong {
        color: #ffffff;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }
    
    /* Custom styling for expanders */
    .stExpander {
        background: linear-gradient(135deg, rgba(15,15,35,0.8) 0%, rgba(26,26,46,0.7) 100%);
        border: 1px solid rgba(120,119,198,0.3);
        border-radius: 16px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    
    .stExpander > div:first-child {
        background: transparent !important;
        border: none !important;
    }
    
    .stExpander > div:first-child > div {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        padding: 1.2rem 1.5rem !important;
    }
    
    .stExpander > div:first-child:hover {
        background: rgba(120,119,198,0.1) !important;
    }
    
    @media (max-width: 768px) {
        .main-title { font-size: 2rem; }
        .main-header {
            flex-direction: column;
            gap: 1.5rem;
        }
        .header-metrics {
            justify-content: center;
        }
        .tab-section { padding: 1.5rem; }
        .main .block-container { padding: 1rem 2rem; }
    }
</style>
""", unsafe_allow_html=True)

class CriminalCodeAssistant:
    def __init__(self):
        self.setup_api()
        
    def setup_api(self):
        """Setup Google Generative AI"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("🔐 **API Configuration Required**")
            st.markdown("Please create a `.env` file with your Google API key:")
            st.code("GOOGLE_API_KEY=your_api_key_here", language="bash")
            st.markdown("[🔗 Get your API key from Google AI Studio](https://aistudio.google.com/app/apikey)")
            st.stop()
        
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            st.error(f"❌ **API Configuration Failed:** {e}")
            st.stop()

    @st.cache_resource
    def load_embeddings_and_db(_self):
        """Load embeddings and FAISS database"""
        try:
            with st.spinner("🧠 Loading AI models and legal database..."):
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                index_path = Path("faiss_index")
                if not index_path.exists():
                    st.error("📚 **Knowledge Base Not Found**")
                    st.markdown("Please build the FAISS index first:")
                    st.code("python build_index.py", language="bash")
                    st.stop()
                
                db = FAISS.load_local(
                    str(index_path), 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                return embeddings, db
                
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.stop()

    @st.cache_data
    def get_relevant_documents(_self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents"""
        try:
            _, db = _self.load_embeddings_and_db()
            retriever = db.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(query)
            
            doc_data = []
            for doc in docs:
                doc_data.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return doc_data
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def extract_key_term(self, query: str) -> str:
        """Extract main legal concept from query"""
        legal_terms = [
            "murder", "manslaughter", "assault", "theft", "robbery", "fraud", 
            "conspiracy", "criminal negligence", "sexual assault", "kidnapping"
        ]
        
        query_lower = query.lower()
        for term in legal_terms:
            if term in query_lower:
                return term
        return None

    def generate_definition(self, query: str, context: str) -> Optional[str]:
        """Generate definition"""
        key_term = self.extract_key_term(query)
        if not key_term:
            return None
            
        definition_prompt = f"""Based on Canadian criminal law context, provide a brief definition of "{key_term}" in 2-3 sentences.

CONTEXT: {context[:800]}

Definition:"""

        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            response = model.generate_content([definition_prompt])
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating definition: {e}")
            return None

    def generate_response(self, query: str, context: str) -> str:
        """Generate response"""
        prompt = f"""You are a Canadian criminal law assistant. Provide clear, accurate information based on the context.

CONTEXT: {context}
QUESTION: {query}

Provide a professional response including relevant section numbers and always remind users to consult legal professionals."""

        models_to_try = ["gemini-1.5-flash", "gemini-1.5-flash-002", "gemini-1.5-pro"]
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name=model_name)
                response = model.generate_content([prompt])
                return response.text
            except Exception as e:
                logger.error(f"Model {model_name} failed: {e}")
                continue
        
        return "❌ **Response Generation Failed** - Please try again later."

    def process_uploaded_file(self, uploaded_file) -> Optional[str]:
        """Process uploaded file"""
        try:
            content = uploaded_file.read().decode("utf-8")
            if len(content.strip()) < 100:
                st.warning("⚠️ File seems too short. Please ensure it contains substantial legal content.")
            return content
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return None

def initialize_session_state():
    """Initialize session state"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

def display_main_interface(assistant: CriminalCodeAssistant):
    """Display main interface"""
    
    # Header with metrics
    st.markdown(f'''
    <div class="main-header">
        <div class="header-content">
            <h1 class="main-title">Canadian Criminal Code Assistant</h1>
            <p class="main-subtitle">
                Advanced AI-powered legal intelligence platform for comprehensive Canadian criminal law analysis
            </p>
        </div>
        <div class="header-metrics">
            <div class="header-metric questions">
                <div class="header-metric-value">{len(st.session_state.chat_history)}</div>
                <div class="header-metric-label">Active Queries</div>
            </div>
            <div class="header-metric total">
                <div class="header-metric-value">{st.session_state.total_queries}</div>
                <div class="header-metric-label">Total Operations</div>
            </div>
            <div class="header-metric database">
                <div class="header-metric-value">Ready</div>
                <div class="header-metric-label">Database Status</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Tab section - removed the div wrapper
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Ask Questions", "📋 Upload Files", "🔬 Analysis", "🗂️ Mission Archive"])
    
    with tab1:
        st.markdown("### ⚡ Legal Intelligence Query Center")
        query = st.text_input(
            "",
            placeholder="Enter your legal question here (e.g., 'What are the penalties for theft under $5000?')",
            help="Ask questions about Canadian criminal law, procedures, penalties, or legal definitions",
            label_visibility="collapsed"
        )
        
        if query and query.strip():
            try:
                with st.spinner("⚡ Processing legal intelligence..."):
                    relevant_docs = assistant.get_relevant_documents(query, k=5)
                    context = "\n\n".join([doc["content"] for doc in relevant_docs])
                    
                    if context.strip():
                        definition = assistant.generate_definition(query, context)
                        response = assistant.generate_response(query, context)
                        
                        if definition:
                            st.markdown(f'''
                            <div class="definition-box">
                                <div class="definition-title">◈ Quick Definition</div>
                                <div class="definition-content">{definition}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        st.markdown('<h2 class="section-header">⚡ Intelligence Analysis Report</h2>', unsafe_allow_html=True)
                        st.markdown(f'''
                        <div class="response-container">
                            <div class="response-content">{response}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        if relevant_docs:
                            st.markdown("### ⚙️ **Supporting Legal Sources**")
                            
                            for i, doc in enumerate(relevant_docs, 1):
                                metadata = doc["metadata"]
                                section_id = metadata.get('id', f'Unknown-{i}')
                                section_title = metadata.get('title', 'Legal Provision')
                                category = metadata.get('section_type', 'Unknown').replace('_', ' ').title()
                                page_num = metadata.get('page_start', 'N/A')
                                
                                # Create a more descriptive title with highlighted page number
                                display_title = f"⬢ Section {section_id}"
                                if section_title and section_title != 'Legal Provision':
                                    display_title += f" - {section_title[:60]}{'...' if len(section_title) > 60 else ''}"
                                
                                with st.expander(display_title, expanded=False):
                                    # Header info with prominent page number
                                    st.markdown(f"""
                                    <div style="
                                        background: linear-gradient(135deg, rgba(120,119,198,0.1) 0%, rgba(255,119,198,0.05) 100%);
                                        border-radius: 12px;
                                        padding: 1.5rem;
                                        margin-bottom: 1rem;
                                        border: 1px solid rgba(120,119,198,0.2);
                                    ">
                                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                                            <div style="display: flex; align-items: center; gap: 1rem;">
                                                <div>
                                                    <div style="font-size: 1.3rem; font-weight: 700; color: #7877c6; margin-bottom: 0.3rem;">
                                                        ⬢ Section {section_id}
                                                    </div>
                                                    <div style="font-size: 0.9rem; color: #a0aec0;">
                                                        Category: {category}
                                                    </div>
                                                </div>
                                                <div style="
                                                    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                                                    color: #ffffff;
                                                    padding: 0.8rem 1.2rem;
                                                    border-radius: 12px;
                                                    font-size: 1.1rem;
                                                    font-weight: 800;
                                                    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                                                    box-shadow: 0 4px 15px rgba(245,158,11,0.4), inset 0 1px 0 rgba(255,255,255,0.2);
                                                    border: 2px solid rgba(255,255,255,0.2);
                                                    font-family: 'Orbitron', monospace;
                                                ">
                                                    ◉ Page {page_num}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Content with better formatting
                                    st.markdown("**◈ Legal Text:**")
                                    st.markdown(f"""
                                    <div style="
                                        background: rgba(15,15,35,0.6);
                                        border-left: 4px solid #7877c6;
                                        border-radius: 8px;
                                        padding: 1.5rem;
                                        margin: 1rem 0;
                                        font-size: 1.05rem;
                                        line-height: 1.7;
                                        color: #e2e8f0;
                                        font-family: 'Poppins', sans-serif;
                                    ">
                                        {doc['content']}
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.chat_history.append((query, response, timestamp))
                        st.session_state.total_queries += 1
                        
                    else:
                        st.error("❌ No relevant legal intelligence found for your query")
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    with tab2:
        st.markdown("### 📋 **Document Upload Center**")
        st.info("⚡ Upload legal documents (.txt format) to enhance analysis")
        
        uploaded_file = st.file_uploader(
            "Upload Legal Document",
            type=["txt"],
            help="Upload legal documents for analysis"
        )
        
        if uploaded_file:
            uploaded_text = assistant.process_uploaded_file(uploaded_file)
            if uploaded_text:
                st.success(f"✅ Document uploaded ({len(uploaded_text):,} characters)")
                
                with st.expander("◈ **Document Preview**", expanded=False):
                    st.text_area("Content", uploaded_text[:1000] + "..." if len(uploaded_text) > 1000 else uploaded_text, height=200, disabled=True)
        
        # Add question functionality to uploaded documents
        st.markdown("---")
        st.markdown("### ⚡ **Query Uploaded Document**")
        st.info("⚡ Ask questions about your uploaded document")
        
        doc_query = st.text_input(
            "",
            placeholder="Ask a question about the uploaded document (e.g., 'What does this document say about theft?')",
            help="Ask questions about the content of your uploaded document",
            label_visibility="collapsed",
            key="doc_query"
        )
        
        if doc_query and doc_query.strip() and uploaded_file:
            try:
                with st.spinner("⚡ Analyzing uploaded document and database..."):
                    if uploaded_text:
                        # First analyze the uploaded document
                        doc_definition = assistant.generate_definition(doc_query, uploaded_text[:2000])
                        doc_response = assistant.generate_response(doc_query, uploaded_text)
                        
                        # Then search the database for additional context
                        relevant_docs = assistant.get_relevant_documents(doc_query, k=3)
                        db_context = "\n\n".join([doc["content"] for doc in relevant_docs])
                        
                        # Display document analysis first
                        if doc_definition:
                            st.markdown(f'''
                            <div class="definition-box">
                                <div class="definition-title">◈ Definition from Your Document</div>
                                <div class="definition-content">{doc_definition}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        st.markdown('<h2 class="section-header">◈ Document Analysis Report</h2>', unsafe_allow_html=True)
                        st.markdown(f'''
                        <div class="response-container">
                            <div class="response-content">{doc_response}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Show document source info
                        st.markdown("### ◈ **Your Document Source**")
                        st.info(f"**File:** {uploaded_file.name} | **Size:** {len(uploaded_text):,} characters")
                        
                        # Then show database analysis if relevant docs found
                        if db_context.strip():
                            st.markdown("---")
                            st.markdown('<h2 class="section-header">⚙️ Additional Criminal Code Context</h2>', unsafe_allow_html=True)
                            
                            # Generate response based on database context
                            db_response = assistant.generate_response(doc_query, db_context)
                            st.markdown(f'''
                            <div class="response-container">
                                <div class="response-content">{db_response}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            # Show supporting legal sources from database
                            if relevant_docs:
                                st.markdown("### ⚙️ **Supporting Legal Sources from Database**")
                                
                                for i, doc in enumerate(relevant_docs, 1):
                                    metadata = doc["metadata"]
                                    section_id = metadata.get('id', f'Unknown-{i}')
                                    section_title = metadata.get('title', 'Legal Provision')
                                    category = metadata.get('section_type', 'Unknown').replace('_', ' ').title()
                                    page_num = metadata.get('page_start', 'N/A')
                                    
                                    display_title = f"⬢ Section {section_id}"
                                    if section_title and section_title != 'Legal Provision':
                                        display_title += f" - {section_title[:60]}{'...' if len(section_title) > 60 else ''}"
                                    
                                    with st.expander(display_title, expanded=False):
                                        st.markdown(f"""
                                        <div style="
                                            background: linear-gradient(135deg, rgba(120,119,198,0.1) 0%, rgba(255,119,198,0.05) 100%);
                                            border-radius: 12px;
                                            padding: 1.5rem;
                                            margin-bottom: 1rem;
                                            border: 1px solid rgba(120,119,198,0.2);
                                        ">
                                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                                                <div style="display: flex; align-items: center; gap: 1rem;">
                                                    <div>
                                                        <div style="font-size: 1.3rem; font-weight: 700; color: #7877c6; margin-bottom: 0.3rem;">
                                                            ⬢ Section {section_id}
                                                        </div>
                                                        <div style="font-size: 0.9rem; color: #a0aec0;">
                                                            Category: {category}
                                                        </div>
                                                    </div>
                                                    <div style="
                                                        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                                                        color: #ffffff;
                                                        padding: 0.8rem 1.2rem;
                                                        border-radius: 12px;
                                                        font-size: 1.1rem;
                                                        font-weight: 800;
                                                        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                                                        box-shadow: 0 4px 15px rgba(245,158,11,0.4), inset 0 1px 0 rgba(255,255,255,0.2);
                                                        border: 2px solid rgba(255,255,255,0.2);
                                                        font-family: 'Orbitron', monospace;
                                                    ">
                                                        ◉ Page {page_num}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        st.markdown("**◈ Legal Text:**")
                                        st.markdown(f"""
                                        <div style="
                                            background: rgba(15,15,35,0.6);
                                            border-left: 4px solid #7877c6;
                                            border-radius: 8px;
                                            padding: 1.5rem;
                                            margin: 1rem 0;
                                            font-size: 1.05rem;
                                            line-height: 1.7;
                                            color: #e2e8f0;
                                            font-family: 'Poppins', sans-serif;
                                        ">
                                            {doc['content']}
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        # Combine both responses for history
                        combined_response = f"**Document Analysis:**\n{doc_response}\n\n**Criminal Code Context:**\n{db_response if db_context.strip() else 'No additional database context found.'}"
                        
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.chat_history.append((f"[DOC] {doc_query}", combined_response, timestamp))
                        st.session_state.total_queries += 1
                    else:
                        st.error("❌ No document content available for analysis")
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
        elif doc_query and doc_query.strip() and not uploaded_file:
            st.warning("⚠️ Please upload a document first before asking questions about it.")
    
    with tab3:
        st.markdown("### 🔬 **Criminal Code Database Analytics**")
        
        if st.session_state.chat_history:
            categories = {}
            section_breakdown = {}
            
            for query, _, _ in st.session_state.chat_history:
                try:
                    relevant_docs = assistant.get_relevant_documents(query, k=5)
                    
                    for doc in relevant_docs:
                        metadata = doc.get("metadata", {})
                        category = metadata.get("section_type", "Unknown")
                        
                        if category == "Unknown" or category == "criminal_code_section":
                            title = metadata.get("title", "")
                            
                            if any(word in title.lower() for word in ["property", "theft", "robbery", "fraud", "mischief"]):
                                category = "Offences Against Property"
                            elif any(word in title.lower() for word in ["person", "assault", "murder", "manslaughter"]):
                                category = "Offences Against the Person"
                            elif any(word in title.lower() for word in ["principle", "general", "interpretation"]):
                                category = "General Principles"
                            elif any(word in title.lower() for word in ["procedure", "court", "trial", "evidence"]):
                                category = "Procedure and Evidence"
                            elif any(word in title.lower() for word in ["sentence", "punishment", "penalty"]):
                                category = "Sentencing and Penalties"
                            else:
                                category = "Other Criminal Code Provisions"
                        
                        category = category.replace("_", " ").title()
                        categories[category] = categories.get(category, 0) + 1
                        
                        section_id = metadata.get("id", "")
                        section_key = f"Section {section_id}" if section_id else "Unknown Section"
                        section_breakdown[section_key] = section_breakdown.get(section_key, 0) + 1
                        
                except Exception:
                    categories["General Legal Inquiry"] = categories.get("General Legal Inquiry", 0) + 1
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### ⚙️ **Criminal Code Categories**")
                
                if categories:
                    try:
                        df = pd.DataFrame(list(categories.items()), columns=['Category', 'Frequency'])
                        df = df.sort_values('Frequency', ascending=True)
                        
                        fig = px.bar(
                            df, 
                            x='Frequency', 
                            y='Category',
                            orientation='h',
                            title="Database Category Access Frequency",
                            color='Frequency',
                            color_continuous_scale=['#1e40af', '#3b82f6', '#60a5fa', '#93c5fd'],
                        )
                        
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#e2e8f0',
                            title_font_size=16,
                            title_font_color='#7877c6',
                            height=450
                        )
                        
                        fig.update_xaxes(gridcolor='rgba(120,119,198,0.2)')
                        fig.update_yaxes(gridcolor='rgba(120,119,198,0.2)')
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except ImportError:
                        st.error("📊 Please install: `pip install pandas plotly`")
                        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                            percentage = (count / sum(categories.values())) * 100
                            st.markdown(f"• **{category}**: {count} accesses ({percentage:.1f}%)")
            
            with col2:
                st.markdown("#### ◈ **Category Distribution**")
                
                if categories:
                    try:
                        fig_pie = px.pie(
                            df, 
                            values='Frequency', 
                            names='Category',
                            title="Criminal Code Usage Distribution"
                        )
                        
                        fig_pie.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#e2e8f0',
                            title_font_color='#7877c6',
                            height=450,
                            showlegend=True
                        )
                        
                        fig_pie.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            hovertemplate='<b>%{label}</b><br>Accessed: %{value} times<br>Percentage: %{percent}<extra></extra>'
                        )
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    except ImportError:
                        st.info("Install pandas and plotly for charts")
                
        # Analytics summary at the bottom as a simple container
        if st.session_state.chat_history:
            categories = {}
            
            for query, _, _ in st.session_state.chat_history:
                try:
                    relevant_docs = assistant.get_relevant_documents(query, k=5)
                    
                    for doc in relevant_docs:
                        metadata = doc.get("metadata", {})
                        category = metadata.get("section_type", "Unknown")
                        
                        if category == "Unknown" or category == "criminal_code_section":
                            title = metadata.get("title", "")
                            
                            if any(word in title.lower() for word in ["property", "theft", "robbery", "fraud", "mischief"]):
                                category = "Offences Against Property"
                            elif any(word in title.lower() for word in ["person", "assault", "murder", "manslaughter"]):
                                category = "Offences Against the Person"
                            elif any(word in title.lower() for word in ["principle", "general", "interpretation"]):
                                category = "General Principles"
                            elif any(word in title.lower() for word in ["procedure", "court", "trial", "evidence"]):
                                category = "Procedure and Evidence"
                            elif any(word in title.lower() for word in ["sentence", "punishment", "penalty"]):
                                category = "Sentencing and Penalties"
                            else:
                                category = "Other Criminal Code Provisions"
                        
                        category = category.replace("_", " ").title()
                        categories[category] = categories.get(category, 0) + 1
                        
                except Exception:
                    categories["General Legal Inquiry"] = categories.get("General Legal Inquiry", 0) + 1
            
            if categories:
                total_accesses = sum(categories.values())
                most_accessed = max(categories.items(), key=lambda x: x[1])
                
                st.markdown("### 🔬 Analytics Summary")
                
                # Create a simple container with metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Database Categories",
                        value=len(categories),
                        help="Number of different legal categories accessed"
                    )
                
                with col2:
                    st.metric(
                        label="Total Accesses", 
                        value=total_accesses,
                        help="Total number of database accesses across all queries"
                    )
                
                with col3:
                    st.metric(
                        label="Highest Access Count",
                        value=most_accessed[1],
                        help=f"Most accessed category: {most_accessed[0]}"
                    )
                
                # Show most accessed category info
                st.info(f"⚡ **Most accessed category:** {most_accessed[0]} ({most_accessed[1]} accesses)")
                
        else:
            st.info("🔬 **No database access data available yet.** Start asking questions to see analytics!")
    
    with tab4:
        st.markdown("### 🗂️ **Query History Archive**")
        
        if st.session_state.chat_history:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("⚡ **Clear Archive**", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.total_queries = 0
                    st.success("✅ Archive cleared!")
            
            st.markdown("---")
            
            for i, (q, a, timestamp) in enumerate(reversed(st.session_state.chat_history)):
                operation_num = len(st.session_state.chat_history) - i
                st.markdown(f"**Operation #{operation_num}:** {q}")
                st.markdown(f"*{timestamp}*")
                
                with st.expander(f"⚙️ **View Analysis #{operation_num}**", expanded=False):
                    st.markdown(a)
                
                st.markdown("---")
        else:
            st.info("🗂️ No queries yet. Ask your first question!")

def display_educational_notice():
    """Display educational notice"""
    st.markdown('''
    <div class="educational-notice">
        <p class="educational-text">
            <strong>⚠️ Important Legal Disclaimer:</strong> This application is for educational and informational purposes only. 
            The information provided should not be considered legal advice. Always consult with qualified legal professionals 
            for specific legal matters and official interpretations of Canadian criminal law.
        </p>
    </div>
    ''', unsafe_allow_html=True)

def main():
    """Main application function"""
    initialize_session_state()
    assistant = CriminalCodeAssistant()
    display_main_interface(assistant)
    display_educational_notice()

if __name__ == "__main__":
    main()
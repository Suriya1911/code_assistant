"""
Optimized Canadian Criminal Code Assistant
High-performance Streamlit app with caching and efficient vector search
"""

import streamlit as st
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Canadian Criminal Code Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS - only essential fixes
st.markdown("""
<style>
    /* Only fix critical display issues, no fancy styling */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Ensure text is visible */
    .main .block-container {
        color: white;
    }
    
    /* Basic button styling */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
    }
    
    /* Fix text input visibility */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
        border: 1px solid #404040;
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
            st.error("❌ GOOGLE_API_KEY not found in environment variables")
            st.stop()
        
        genai.configure(api_key=api_key)

    @st.cache_resource
    def load_embeddings_and_db(_self):
        """
        Load embeddings and FAISS database with caching for performance
        This runs only once per session and is cached
        """
        try:
            with st.spinner("🔄 Loading AI models (this may take a moment on first run)..."):
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # Load FAISS database
                index_path = Path("faiss_index")
                if not index_path.exists():
                    st.error("❌ FAISS index not found. Please run 'python build_index.py' first.")
                    st.stop()
                
                db = FAISS.load_local(
                    str(index_path), 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                logger.info("Successfully loaded embeddings and FAISS database")
                return embeddings, db
                
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.stop()

    @st.cache_data
    def get_relevant_documents(_self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents with caching
        """
        try:
            _, db = _self.load_embeddings_and_db()
            retriever = db.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(query)
            
            # Convert to serializable format for caching
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

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using Gemini AI
        """
        prompt = f"""You are a Canadian criminal law assistant. Use the following legal text to answer the question accurately and professionally.

LEGAL CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide clear, accurate legal information based on the context
- Include relevant section numbers when applicable
- Use professional but accessible language
- If the context doesn't contain enough information, state this clearly
- Always remind users to consult with a qualified legal professional

ANSWER:"""

        try:
            model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-001")
            response = model.generate_content([prompt])
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"❌ Error generating response: {str(e)}"

    def process_uploaded_file(self, uploaded_file) -> Optional[str]:
        """
        Process uploaded text file
        """
        try:
            content = uploaded_file.read().decode("utf-8")
            if len(content.strip()) < 100:
                st.warning("⚠️ Uploaded file seems too short. Please ensure it contains substantial legal content.")
            return content
        except Exception as e:
            st.error(f"❌ Error reading uploaded file: {str(e)}")
            return None

def initialize_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

def display_sidebar(assistant: CriminalCodeAssistant):
    """Display sidebar with chat history and file upload"""
    with st.sidebar:
        st.header("⚖️ Criminal Code Assistant")
        
        # Statistics - Using Streamlit metrics instead of custom HTML
        st.subheader("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Questions Asked",
                value=len(st.session_state.chat_history),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Total Sessions",
                value=st.session_state.total_queries,
                delta=None
            )
        
        st.markdown("---")
        
        # File upload
        st.subheader("📄 Custom Legal Context")
        uploaded_file = st.file_uploader(
            "Upload a .txt file to override default Criminal Code context",
            type=["txt"],
            help="Upload your own legal text to search within instead of the Criminal Code"
        )
        
        uploaded_text = None
        if uploaded_file:
            uploaded_text = assistant.process_uploaded_file(uploaded_file)
            if uploaded_text:
                st.success(f"✅ File loaded ({len(uploaded_text)} characters)")
        
        st.markdown("---")
        
        # Chat history - Using native Streamlit components
        st.subheader("💬 Recent Questions")
        if st.session_state.chat_history:
            for i, (q, a, timestamp) in enumerate(st.session_state.chat_history[-5:]):  # Show last 5
                with st.expander(f"Q{len(st.session_state.chat_history)-4+i}: {q[:30]}..."):
                    st.write(f"**Time:** {timestamp}")
                    st.write(f"**Question:** {q}")
                    st.write(f"**Answer Preview:** {a[:200]}...")
        else:
            st.info("No questions asked yet")
        
        # Clear history button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        
        return uploaded_text

def display_main_interface(assistant: CriminalCodeAssistant, uploaded_text: Optional[str]):
    """Display main chat interface"""
    
    # Header - Using native Streamlit components
    st.title("🇨🇦 Canadian Criminal Code Assistant")
    st.subheader("Ask questions about Canadian criminal law and get instant, sourced answers")
    st.divider()
    
    # Query input
    query = st.text_input(
        "❓ Ask your legal question:",
        placeholder="e.g., What constitutes assault under Canadian law?",
        help="Ask specific questions about criminal law, procedures, or penalties"
    )
    
    # Process query
    if query and query.strip():
        with st.spinner("🔍 Searching legal database and generating response..."):
            
            if uploaded_text:
                # Use uploaded file as context
                context = uploaded_text
                st.info("📄 Using uploaded file as context")
                relevant_docs = []
            else:
                # Use FAISS search
                relevant_docs = assistant.get_relevant_documents(query, k=5)
                context = "\n\n".join([doc["content"] for doc in relevant_docs])
            
            # Generate response
            if context.strip():
                response = assistant.generate_response(query, context)
                
                # Display response - Pure Streamlit
                st.subheader("📋 Answer")
                
                # Use Streamlit's built-in success container for styling
                with st.container():
                    st.info(response)
                
                # Display sources (only for FAISS search) - Pure Streamlit
                if relevant_docs:
                    st.subheader("📚 Sources")
                    
                    # Create tabs for sources - using simple string names
                    tab_names = []
                    for doc in relevant_docs:
                        section_id = doc['metadata'].get('id', 'N/A')
                        tab_names.append(f"Section {section_id}")
                    
                    tabs = st.tabs(tab_names)
                    
                    for i, (tab, doc) in enumerate(zip(tabs, relevant_docs)):
                        with tab:
                            metadata = doc["metadata"]
                            
                            # Display metadata using simple columns
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Section ID", metadata.get('id', 'N/A'))
                            with col2:
                                st.metric("Page", metadata.get('page_start', 'N/A'))
                            with col3:
                                category = metadata.get('section_type', 'Unknown').replace('_', ' ').title()
                                st.metric("Category", category)
                            
                            # Display title
                            title = metadata.get('title', 'No title')
                            st.write(f"**Title:** {title}")
                            
                            # Show section content in expandable format
                            with st.expander("📖 Full Section Text", expanded=False):
                                st.text(doc['content'])
                    
                    # Download button with better styling
                    st.divider()
                    download_content = f"""Canadian Criminal Code Assistant - Response

Question: {query}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Answer:
{response}

Sources:
"""
                    for doc in relevant_docs:
                        metadata = doc["metadata"]
                        download_content += f"""
{'='*50}
Section {metadata.get('id', 'N/A')}: {metadata.get('title', 'No title')}
Category: {metadata.get('section_type', 'Unknown').replace('_', ' ').title()}
Page: {metadata.get('page_start', 'N/A')}

{doc['content']}

"""
                    
                    st.download_button(
                        label="📥 Download Complete Response & Sources",
                        data=download_content,
                        file_name=f"criminal_code_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                # Update session state
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history.append((query, response, timestamp))
                st.session_state.total_queries += 1
                
            else:
                st.error("❌ No relevant legal context found for your question")

def display_footer():
    """Display footer with important disclaimers"""
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Important Legal Disclaimer
    
    This assistant provides general information about Canadian criminal law for educational purposes only. 
    It is **NOT** a substitute for professional legal advice. Always consult with a qualified lawyer 
    for specific legal matters or before making any legal decisions.
    
    **Built with:** Streamlit • Google Gemini AI • FAISS Vector Search • LangChain
    """)

def main():
    """Main application function"""
    
    # Initialize
    initialize_session_state()
    assistant = CriminalCodeAssistant()
    
    # Display interface
    uploaded_text = display_sidebar(assistant)
    display_main_interface(assistant, uploaded_text)
    display_footer()

if __name__ == "__main__":
    main()
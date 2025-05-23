"""
Optimized FAISS Index Builder for Legal Document Search
Efficiently creates vector embeddings with progress tracking and error handling
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self):
        load_dotenv()
        self.sections_file = Path("data/processed/sections.json")
        self.index_dir = Path("faiss_index")
        self.embedding_model = "all-MiniLM-L6-v2"
        
    def load_sections(self) -> Optional[List[Dict]]:
        """
        Load sections from JSON file with validation
        """
        logger.info("Loading sections from JSON...")
        
        if not self.sections_file.exists():
            logger.error(f"Sections file not found: {self.sections_file}")
            logger.error("Please run pdf_scraper.py first to extract sections")
            return None
        
        try:
            with open(self.sections_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle both old and new format
            if isinstance(data, dict) and "sections" in data:
                sections = data["sections"]
                metadata = data.get("metadata", {})
                logger.info(f"Loaded {len(sections)} sections with metadata")
                logger.info(f"Source: {metadata.get('source_file', 'Unknown')}")
            else:
                sections = data
                logger.info(f"Loaded {len(sections)} sections (legacy format)")
            
            if not sections:
                logger.error("No sections found in file")
                return None
                
            return sections
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in sections file: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading sections: {e}")
            return None
    
    def create_documents(self, sections: List[Dict]) -> List[Document]:
        """
        Convert sections to LangChain Documents with enhanced metadata
        """
        logger.info("Converting sections to documents...")
        
        documents = []
        
        for i, section in enumerate(sections):
            try:
                # Validate required fields
                if not section.get("content") or not section.get("id"):
                    logger.warning(f"Skipping section {i}: missing content or id")
                    continue
                
                # Create document with rich metadata
                doc = Document(
                    page_content=section["content"],
                    metadata={
                        "id": section["id"],
                        "title": section.get("title", ""),
                        "page_start": section.get("page_start", 0),
                        "word_count": section.get("word_count", len(section["content"].split())),
                        "section_type": self._categorize_section(section["id"])
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Error processing section {i}: {e}")
                continue
        
        logger.info(f"Created {len(documents)} documents from {len(sections)} sections")
        return documents
    
    def _categorize_section(self, section_id: str) -> str:
        """
        Categorize section based on ID for better filtering
        """
        try:
            num = int(section_id.split('.')[0].rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            
            if 1 <= num <= 42:
                return "general_principles"
            elif 43 <= num <= 130:
                return "offences_against_public_order"
            elif 131 <= num <= 200:
                return "offences_against_person_reputation"
            elif 201 <= num <= 370:
                return "offences_against_property"
            elif 371 <= num <= 490:
                return "fraud_related_offences"
            elif 491 <= num <= 672:
                return "procedure_evidence"
            else:
                return "other"
        except:
            return "other"
    
    def initialize_embeddings(self) -> Optional[HuggingFaceEmbeddings]:
        """
        Initialize embedding model with error handling
        """
        logger.info(f"Initializing embeddings model: {self.embedding_model}")
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},  # Explicit CPU usage
                encode_kwargs={'normalize_embeddings': True}  # Better similarity scores
            )
            
            # Test the embeddings
            test_embedding = embeddings.embed_query("test query")
            logger.info(f"Embeddings initialized successfully (dim: {len(test_embedding)})")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return None
    
    def build_faiss_index(self, documents: List[Document], embeddings: HuggingFaceEmbeddings) -> bool:
        """
        Build FAISS index with progress tracking
        """
        logger.info("Building FAISS index...")
        
        try:
            # Create index with progress
            logger.info(f"Processing {len(documents)} documents...")
            
            # Build in batches for large datasets
            batch_size = 100
            if len(documents) > batch_size:
                logger.info(f"Processing in batches of {batch_size}")
                
                # Create initial index with first batch
                first_batch = documents[:batch_size]
                db = FAISS.from_documents(first_batch, embeddings)
                
                # Add remaining documents in batches
                for i in range(batch_size, len(documents), batch_size):
                    batch = documents[i:i+batch_size]
                    batch_db = FAISS.from_documents(batch, embeddings)
                    db.merge_from(batch_db)
                    logger.info(f"Processed {min(i+batch_size, len(documents))}/{len(documents)} documents")
            else:
                db = FAISS.from_documents(documents, embeddings)
            
            # Save index
            logger.info(f"Saving index to {self.index_dir}")
            db.save_local(str(self.index_dir))
            
            # Verify saved index
            test_db = FAISS.load_local(str(self.index_dir), embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Index verification successful - {test_db.index.ntotal} vectors stored")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False
    
    def run(self) -> bool:
        """
        Main execution method
        """
        logger.info("=== FAISS Index Builder ===")
        
        # Load sections
        sections = self.load_sections()
        if not sections:
            return False
        
        # Create documents
        documents = self.create_documents(sections)
        if not documents:
            logger.error("No valid documents created")
            return False
        
        # Initialize embeddings
        embeddings = self.initialize_embeddings()
        if not embeddings:
            return False
        
        # Build index
        success = self.build_faiss_index(documents, embeddings)
        
        if success:
            logger.info("✅ Index building completed successfully!")
            logger.info(f"📁 Index saved to: {self.index_dir}")
            return True
        else:
            logger.error("❌ Index building failed!")
            return False

def main():
    """
    Main function to run the index builder
    """
    builder = IndexBuilder()
    success = builder.run()
    
    if success:
        print("\n🎉 Ready to run the Criminal Code Assistant!")
        print("Next step: Run 'streamlit run app.py'")
        return 0
    else:
        print("\n💡 Troubleshooting:")
        print("1. Make sure you've run 'python pdf_scraper.py' first")
        print("2. Check that the sections.json file exists in data/processed/")
        print("3. Ensure you have sufficient memory for embedding generation")
        return 1

if __name__ == "__main__":
    exit(main())
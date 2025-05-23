"""
Enhanced PDF Scraper for Canadian Criminal Code
Efficiently extracts legal sections with improved accuracy and memory management
"""

import fitz  # PyMuPDF
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CriminalCodeScraper:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path("data/processed")
        self.output_file = self.output_dir / "sections.json"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_pdf(self) -> bool:
        """Check if PDF file exists and is readable"""
        if not self.pdf_path.exists():
            logger.error(f"PDF file not found: {self.pdf_path}")
            return False
            
        try:
            doc = fitz.open(str(self.pdf_path))
            doc.close()
            return True
        except Exception as e:
            logger.error(f"Cannot read PDF file: {e}")
            return False
    
    def extract_sections(self) -> List[Dict[str, str]]:
        """
        Extract sections from Criminal Code PDF with memory-efficient processing
        """
        logger.info("Starting PDF extraction...")
        
        if not self.validate_pdf():
            return []
        
        sections = []
        current_section = None
        section_buffer = []
        
        try:
            doc = fitz.open(str(self.pdf_path))
            total_pages = len(doc)
            logger.info(f"Processing {total_pages} pages...")
            
            for page_num in range(total_pages):
                # Progress indicator every 25 pages
                if page_num % 25 == 0:
                    logger.info(f"Processing page {page_num + 1}/{total_pages}")
                
                page = doc[page_num]
                text = page.get_text()
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Enhanced regex pattern for section detection
                    # Matches: "267", "267.1", "85A", etc. followed by title
                    section_match = re.match(r'^(\d{1,3}[A-Z]?(?:\.\d+)?)\s+(.+)', line)
                    
                    if section_match and self._is_valid_section_start(line):
                        # Save previous section
                        if current_section and section_buffer:
                            current_section["content"] = '\n'.join(section_buffer)
                            if self._is_valid_section(current_section):
                                sections.append(current_section)
                        
                        # Start new section
                        section_id = section_match.group(1)
                        title = section_match.group(2).strip()
                        
                        current_section = {
                            "id": section_id,
                            "title": title,
                            "page_start": page_num + 1
                        }
                        section_buffer = [line]
                        
                    elif current_section:
                        # Add content to current section
                        section_buffer.append(line)
            
            # Don't forget the last section
            if current_section and section_buffer:
                current_section["content"] = '\n'.join(section_buffer)
                if self._is_valid_section(current_section):
                    sections.append(current_section)
            
            doc.close()
            logger.info(f"Successfully extracted {len(sections)} sections")
            
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return []
        
        return sections
    
    def _is_valid_section_start(self, line: str) -> bool:
        """
        Validate if a line is actually a section start
        Helps avoid false positives from page numbers, footnotes, etc.
        """
        # Skip if line is too short
        if len(line.strip()) < 10:
            return False
            
        # Skip common false positives
        false_positives = [
            r'^\d+\s*$',  # Just a number
            r'^\d+\s+Page\s+\d+',  # Page references
            r'^\d+\s+\(\d+\)',  # Subsection references only
        ]
        
        for pattern in false_positives:
            if re.match(pattern, line, re.IGNORECASE):
                return False
        
        return True
    
    def _is_valid_section(self, section: Dict[str, str]) -> bool:
        """
        Validate extracted section quality
        """
        if not section.get("content"):
            return False
            
        content = section["content"].strip()
        
        # Must have minimum content length
        if len(content) < 50:
            return False
            
        # Must have meaningful title
        title = section.get("title", "").strip()
        if len(title) < 5:
            return False
            
        return True
    
    def clean_sections(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Clean and normalize section content
        """
        logger.info("Cleaning extracted sections...")
        
        cleaned_sections = []
        
        for section in sections:
            content = section["content"]
            
            # Clean up content formatting
            content = re.sub(r'\n+', '\n', content)  # Remove multiple newlines
            content = re.sub(r'\s+', ' ', content.replace('\n', ' '))  # Normalize whitespace
            content = content.strip()
            
            # Update section
            section["content"] = content
            section["word_count"] = len(content.split())
            
            cleaned_sections.append(section)
        
        return cleaned_sections
    
    def save_sections(self, sections: List[Dict[str, str]]) -> bool:
        """
        Save sections to JSON file with metadata
        """
        if not sections:
            logger.error("No sections to save")
            return False
        
        try:
            # Add metadata
            output_data = {
                "metadata": {
                    "source_file": str(self.pdf_path),
                    "total_sections": len(sections),
                    "extraction_date": str(Path(__file__).stat().st_mtime)
                },
                "sections": sections
            }
            
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully saved {len(sections)} sections to {self.output_file}")
            
            # Print sample for verification
            if sections:
                sample = sections[0]
                logger.info(f"\nSample section:")
                logger.info(f"ID: {sample['id']}")
                logger.info(f"Title: {sample['title']}")
                logger.info(f"Content preview: {sample['content'][:200]}...")
                logger.info(f"Word count: {sample.get('word_count', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving sections: {e}")
            return False
    
    def run(self) -> bool:
        """
        Main execution method
        """
        logger.info("=== Canadian Criminal Code PDF Scraper ===")
        
        # Extract sections
        sections = self.extract_sections()
        if not sections:
            return False
        
        # Clean sections
        sections = self.clean_sections(sections)
        
        # Save to file
        return self.save_sections(sections)

def main():
    """
    Main function to run the scraper
    """
    pdf_path = "data/C-46.pdf"
    
    scraper = CriminalCodeScraper(pdf_path)
    success = scraper.run()
    
    if success:
        print("\n✅ PDF scraping completed successfully!")
        print(f"📁 Output saved to: {scraper.output_file}")
    else:
        print("\n❌ PDF scraping failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
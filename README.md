# 🇨🇦 Canadian Criminal Code Assistant

An AI-powered legal assistant for Canadian criminal law, built with Streamlit, Google Gemini AI, and FAISS vector search.

## ✨ Features

- **Intelligent Search**: Semantic search through the entire Criminal Code of Canada
- **AI-Powered Responses**: Context-aware answers using Google Gemini AI
- **Source Citations**: Always shows relevant legal sections with full references
- **Custom Context**: Upload your own legal documents for specialized searches
- **Chat History**: Track previous questions and responses
- **Downloadable Reports**: Export questions, answers, and sources
- **Professional UI**: Clean, responsive interface with legal disclaimers

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
Create a `.env` file in the root directory:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Prepare Your Data
Ensure you have the Criminal Code PDF:
```
data/
└── C-46.pdf  # Canadian Criminal Code PDF
```

### 4. Run the Setup Pipeline

#### Step 1: Extract Sections from PDF
```bash
python pdf_scraper.py
```
This will:
- Extract legal sections from the PDF
- Clean and validate content
- Save to `data/processed/sections.json`

#### Step 2: Build Search Index
```bash
python build_index.py
```
This will:
- Create vector embeddings for all sections
- Build FAISS search index
- Save to `faiss_index/` directory

#### Step 3: Launch the App
```bash
streamlit run app.py
```

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── pdf_scraper.py         # PDF extraction and processing
├── build_index.py         # FAISS index creation
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── data/
│   ├── C-46.pdf          # Criminal Code PDF (you provide)
│   └── processed/
│       └── sections.json  # Extracted sections (generated)
└── faiss_index/          # Vector search index (generated)
```

## 🔧 Advanced Configuration

### Embedding Model
The default embedding model is `all-MiniLM-L6-v2`. To use a different model, modify `build_index.py`:

```python
self.embedding_model = "sentence-transformers/all-mpnet-base-v2"  # Higher quality
```

### Search Parameters
Adjust search results in `app.py`:

```python
relevant_docs = assistant.get_relevant_documents(query, k=10)  # More results
```

### Gemini Model
Switch to a different Gemini model in `app.py`:

```python
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-001")  # More powerful
```

## 🐛 Troubleshooting

### Common Issues

1. **"FAISS index not found"**
   - Run `python build_index.py` to create the search index

2. **"Sections file not found"**
   - Run `python pdf_scraper.py` to extract sections from PDF

3. **"GOOGLE_API_KEY not found"**
   - Add your Gemini API key to the `.env` file

4. **Memory issues during indexing**
   - Reduce batch size in `build_index.py`
   - Use a smaller embedding model

5. **PDF extraction errors**
   - Ensure `data/C-46.pdf` exists and is readable
   - Check PDF format compatibility

### Performance Tips

- **First run**: Initial model loading takes 30-60 seconds
- **Subsequent runs**: Models are cached for instant responses
- **Large PDFs**: Extraction may take several minutes
- **Memory usage**: ~2GB RAM recommended for full functionality

## 📋 Usage Examples

### Basic Questions
- "What constitutes assault under Canadian law?"
- "What are the penalties for theft over $5000?"
- "How is murder defined in the Criminal Code?"

### Section-Specific Queries
- "Explain section 265 of the Criminal Code"
- "What does section 322 say about theft?"

### Procedural Questions
- "What is the difference between summary and indictable offences?"
- "How does the bail system work in Canada?"

## ⚖️ Legal Disclaimer

This application provides general information about Canadian criminal law for educational purposes only. It is **NOT** a substitute for professional legal advice. Always consult with a qualified lawyer for specific legal matters.

## 🤝 Contributing

Contributions are welcome! Please ensure all changes maintain:
- Code quality and documentation
- Legal accuracy and appropriate disclaimers
- Performance optimizations
- User experience improvements

## 📄 License

This project is for educational and research purposes. Please respect copyright laws and use responsibly.

---

**Built with:** Streamlit • Google Gemini AI • FAISS • LangChain • PyMuPDF
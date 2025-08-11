# RAG PDF Chat Application

A modern web application that allows users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG) technology.

## Features

- 🔄 **PDF Upload**: Drag-and-drop or click to upload PDF documents
- 💬 **Interactive Chat**: Ask questions about your uploaded PDF content
- 🤖 **AI-Powered Responses**: Get intelligent answers based on document content
- 📚 **Source References**: See which parts of the document were used to generate answers
- 🎨 **Modern UI**: Beautiful, responsive design with smooth animations
- 🔄 **Chat Reset**: Clear conversation history when needed

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /home/anishbawdhankar/Desktop/RAG_v1
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

### Running the Application

You have two options to run the application:

#### Option 1: Using OpenAI API (Recommended for best results)
```bash
python app.py
```

#### Option 2: Using Local/Free Models (No API key required)
```bash
python app_local.py
```

The application will start on `http://localhost:5000`

### Usage

1. **Open your web browser** and navigate to `http://localhost:5000`
2. **Upload a PDF** by dragging and dropping it onto the upload zone or clicking "Choose File"
3. **Wait for processing** - you'll see a success message when ready
4. **Ask questions** about your PDF content in the chat interface
5. **View responses** along with source references from the document

## Project Structure

```
RAG_v1/
├── app.py                 # Main Flask application (OpenAI version)
├── app_local.py          # Local models version (no API key needed)
├── templates/
│   └── index.html        # Frontend HTML template
├── uploads/              # Temporary file upload directory
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

## Technical Details

### Models Used

**OpenAI Version (`app.py`)**:
- **LLM**: GPT-3.5-turbo for question answering
- **Embeddings**: HuggingFace all-MiniLM-L6-v2 for document embeddings
- **Vector Store**: FAISS for similarity search

**Local Version (`app_local.py`)**:
- **LLM**: Google FLAN-T5-base (runs locally)
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Vector Store**: FAISS for similarity search
- **Fallback**: Simple keyword-based search

### Key Features

- **Document Processing**: Extracts text from PDFs and splits into chunks
- **Vector Search**: Uses semantic similarity to find relevant document sections
- **Conversational Memory**: Maintains chat history for context
- **Error Handling**: Graceful fallbacks and user-friendly error messages
- **Responsive Design**: Works on desktop and mobile devices

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenAI API Errors**: Ensure your API key is valid and has credits:
   - Check your `.env` file
   - Verify the API key at https://platform.openai.com

3. **PDF Processing Errors**: 
   - Ensure the PDF is not password-protected
   - Try with a different PDF file
   - Check file size (max 16MB)

4. **Memory Issues**: For large PDFs or limited RAM:
   - Use smaller chunk sizes in the text splitter
   - Consider using the local version with smaller models

### Performance Tips

- **Large PDFs**: May take longer to process initially
- **Local Models**: First run will download models (may take time)
- **Memory Usage**: Close other applications if experiencing slowdowns

## Customization

### Changing Models

Edit the model names in the respective app files:

```python
# For embeddings
embeddings = HuggingFaceEmbeddings(model_name="your-preferred-model")

# For LLM (OpenAI version)
llm = ChatOpenAI(model_name="gpt-4")  # or other OpenAI models
```

### Adjusting Chunk Size

Modify the text splitter parameters:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Increase for more context
    chunk_overlap=300  # Increase for better continuity
)
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.
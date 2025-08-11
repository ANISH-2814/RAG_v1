import os
import tempfile
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
import json
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for RAG components
vectorstore = None
qa_chain = None
chat_history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text

def create_vectorstore(text):
    """Create vector store from text using free/local models"""
    global vectorstore, qa_chain
    
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings using free HuggingFace model
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        # Create LLM using HuggingFace transformers (free)
        # Using a smaller model that works well on CPU
        text_generation_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
            device=-1  # CPU
        )
        
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        return True
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return False

def simple_qa_chain(question, context_chunks):
    """Simple QA function using local processing when HuggingFace models fail"""
    # Simple keyword-based search and response
    relevant_chunks = []
    question_words = question.lower().split()
    
    for chunk in context_chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for word in question_words if word in chunk_lower)
        if score > 0:
            relevant_chunks.append((chunk, score))
    
    # Sort by relevance score
    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
    
    if relevant_chunks:
        # Take top 3 most relevant chunks
        top_chunks = [chunk[0] for chunk in relevant_chunks[:3]]
        context = "\n\n".join(top_chunks)
        
        # Simple response generation
        response = f"Based on the uploaded document:\n\n{context[:500]}..."
        
        return {
            'answer': response,
            'sources': [chunk[:200] + "..." for chunk in top_chunks]
        }
    else:
        return {
            'answer': "I couldn't find relevant information in the document to answer your question.",
            'sources': []
        }

# Store document chunks for fallback
document_chunks = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global chat_history, document_chunks
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from PDF
        text = extract_text_from_pdf(filepath)
        if text is None:
            return jsonify({'error': 'Failed to extract text from PDF'}), 500
        
        # Split text into chunks for fallback
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        document_chunks = text_splitter.split_text(text)
        
        # Try to create vector store with advanced models
        success = create_vectorstore(text)
        
        # Reset chat history
        chat_history = []
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'message': 'PDF uploaded and processed successfully',
            'filename': filename,
            'advanced_mode': success
        })
    
    return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    global qa_chain, chat_history, document_chunks
    
    if not document_chunks:
        return jsonify({'error': 'Please upload a PDF file first'}), 400
    
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    
    try:
        # Try advanced QA chain first
        if qa_chain is not None:
            result = qa_chain({
                "question": question,
                "chat_history": chat_history
            })
            
            # Update chat history
            chat_history.append((question, result['answer']))
            
            # Get source documents for reference
            sources = []
            if 'source_documents' in result:
                for doc in result['source_documents']:
                    sources.append(doc.page_content[:200] + "...")
            
            return jsonify({
                'answer': result['answer'],
                'sources': sources
            })
        else:
            # Fall back to simple QA
            result = simple_qa_chain(question, document_chunks)
            return jsonify(result)
    
    except Exception as e:
        print(f"Error in advanced chat, falling back to simple mode: {e}")
        # Fall back to simple QA
        try:
            result = simple_qa_chain(question, document_chunks)
            return jsonify(result)
        except Exception as e2:
            print(f"Error in simple chat: {e2}")
            return jsonify({'error': 'An error occurred while processing your question'}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    global chat_history
    chat_history = []
    return jsonify({'message': 'Chat history reset successfully'})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

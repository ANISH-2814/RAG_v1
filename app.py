import os
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import json
from dotenv import load_dotenv

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
    """Create vector store from text"""
    global vectorstore, qa_chain
    
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        # Create QA chain
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        return True
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global chat_history
    
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
        
        # Create vector store
        success = create_vectorstore(text)
        if not success:
            return jsonify({'error': 'Failed to process PDF'}), 500
        
        # Reset chat history
        chat_history = []
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'message': 'PDF uploaded and processed successfully',
            'filename': filename
        })
    
    return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    global qa_chain, chat_history
    
    if qa_chain is None:
        return jsonify({'error': 'Please upload a PDF file first'}), 400
    
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    
    try:
        # Get response from QA chain
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
    
    except Exception as e:
        print(f"Error in chat: {e}")
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

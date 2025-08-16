#langgraph code
import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch 
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import ollama_python as ollama
import requests
import json

###Clip Model
import os
from dotenv import load_dotenv
load_dotenv()

### initialize the Clip Model for unified embeddings
clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
### Embedding functions
def embed_image(image_data):
    """Embed image using CLIP"""
    if isinstance(image_data, str):  # If path
        image = Image.open(image_data).convert("RGB")
    else:  # If PIL Image
        image = image_data
    
    inputs=clip_processor(images=image,return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        # Normalize embeddings to unit vector
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    
def embed_text(text):
    """Embed text using CLIP."""
    inputs = clip_processor(
        text=text, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=77  # CLIP's max token length
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        # Normalize embeddings
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    
## Process PDF
pdf_path="test_onlytext.pdf"
doc=fitz.open(pdf_path)
# Storage for all documents and embeddings
all_docs = []
all_embeddings = []
image_data_store = {}  # Store actual image data for LLM

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


for i,page in enumerate(doc):
    ## process text
    text=page.get_text()
    if text.strip():
        ##create temporary document for splitting
        temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
        text_chunks = splitter.split_documents([temp_doc])

        #Embed each chunk using CLIP
        for chunk in text_chunks:
            embedding = embed_text(chunk.page_content)
            all_embeddings.append(embedding)
            all_docs.append(chunk)



    ## process images
    ##Three Important Actions:

    ##Convert PDF image to PIL format
    ##Store as base64 for GPT-4V (which needs base64 images)
    ##Create CLIP embedding for retrieval

    for img_index, img in enumerate(page.get_images(full=True)):
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Create unique identifier
            image_id = f"page_{i}_img_{img_index}"
            
            # Store image as base64 for later use with GPT-4V
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_data_store[image_id] = img_base64
            
            # Embed image using CLIP
            embedding = embed_image(pil_image)
            all_embeddings.append(embedding)
            
            # Create document for image
            image_doc = Document(
                page_content=f"[Image: {image_id}]",
                metadata={"page": i, "type": "image", "image_id": image_id}
            )
            all_docs.append(image_doc)
            
        except Exception as e:
            print(f"Error processing image {img_index} on page {i}: {e}")
            continue

doc.close()



# Create custom FAISS index since we have precomputed embeddings
embeddings_array = np.vstack(all_embeddings)  
vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
    embedding=None,  # We're using precomputed embeddings
    metadatas=[doc.metadata for doc in all_docs]
)
vector_store


# 🆕 NEW: Ollama Client Class
class OllamaLLaVA:
    def __init__(self, model="llava:7b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        
    def generate_with_vision(self, prompt, images=None):
        """Generate response using Ollama's LLaVA model with vision capabilities."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 1000
            }
        }
        
        # Add images if provided (base64 format)
        if images:
            payload["images"] = images
        
        try:
            print(f"🤖 Querying {self.model}...")
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "No response generated")
            
        except requests.exceptions.RequestException as e:
            return f"❌ Ollama Error: {e}"
        except Exception as e:
            return f"❌ Unexpected Error: {e}"
# Initialize GPT-4 Vision model
llm = OllamaLLaVA(model="llava:7b")
def retrieve_multimodal(query, k=5):
    """Unified retrieval using CLIP embeddings for both text and images."""
    # Embed query using CLIP
    query_embedding = embed_text(query)
    
    # Search in unified vector store
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )
    
    return results


# 🆕 NEW: Updated message creation for Ollama LLaVA
def create_ollama_multimodal_message(query, retrieved_docs):
    """Create a message with both text and images for Ollama LLaVA."""
    
    # Separate text and image documents
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    # Build text context
    context_parts = []
    
    # Add text context
    if text_docs:
        text_context = "\n\n".join([
            f"📄 [Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        context_parts.append(f"📝 **TEXT CONTENT:**\n{text_context}")
    
    # Collect images for Ollama
    images_to_analyze = []
    image_descriptions = []
    
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        page = doc.metadata.get('page', '?')
        
        if image_id and image_id in image_data_store:
            # Add base64 image for Ollama
            images_to_analyze.append(image_data_store[image_id])
            image_descriptions.append(f"🖼️ Image from page {page} (analyzing below)")
    
    if image_descriptions:
        context_parts.append(f"📊 **VISUAL CONTENT:**\n" + "\n".join(image_descriptions))
    
    # Create comprehensive prompt for LLaVA
    full_context = "\n\n".join(context_parts)
    
    prompt = f"""🔍 **MULTIMODAL PDF ANALYSIS**

❓ **QUESTION:** {query}

📚 **DOCUMENT CONTEXT:**
{full_context}

🎯 **INSTRUCTIONS:**
1. Carefully read and understand the text content from the PDF
2. Analyze any images/charts/diagrams provided below
3. Provide a comprehensive answer combining insights from BOTH text and visual elements
4. Reference specific page numbers when mentioning information
5. If you see charts, graphs, or tables in images, describe what data they show
6. Be specific and detailed in your analysis

💡 **Your comprehensive analysis:**"""
    
    return {
        "prompt": prompt,
        "images": images_to_analyze
    }


# 🔄 UPDATED: Main pipeline function for Ollama
def multimodal_pdf_rag_pipeline(query):
    """Main pipeline for multimodal RAG using Ollama LLaVA."""
    
    # Retrieve relevant documents (SAME AS BEFORE ✅)
    context_docs = retrieve_multimodal(query, k=5)
    
    # Create Ollama-compatible message (NEW!)
    message_data = create_ollama_multimodal_message(query, context_docs)
    
    # Get response from Ollama LLaVA (NEW!)
    response = llm.generate_with_vision(
        prompt=message_data["prompt"],
        images=message_data["images"]
    )
    
    # Print retrieved context info (SAME AS BEFORE ✅)
    print(f"\n📋 Retrieved {len(context_docs)} documents:")
    for doc in context_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")
        if doc_type == "text":
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  - 📝 Text from page {page}: {preview}")
        else:
            print(f"  - 🖼️ Image from page {page}")
    print("\n")
    
    return response


# 🧪 Testing with enhanced queries
if __name__ == "__main__":
    # 🔧 First, check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in response.json()["models"]]
        print(f"✅ Ollama is running! Available models: {models}")
        
        if "llava:7b" not in models:
            print("❌ LLaVA:7b not found! Run: ollama pull llava:7b")
            exit(1)
        else:
            print("🎉 LLaVA:7b is ready!")
            
    except Exception as e:
        print(f"❌ Ollama not running! Start with: ollama serve\nError: {e}")
        exit(1)
    
    # Example queries optimized for vision capabilities
    queries = [
        "what does this first paragraph of this pdf tells us about ?",
        "who is this person?",
    ]
    
    for query in queries:
        print(f"\n🔥 **Query:** {query}")
        print("-" * 50)
        answer = multimodal_pdf_rag_pipeline(query)
        print(f"**Answer:** {answer}")
        print("=" * 70)
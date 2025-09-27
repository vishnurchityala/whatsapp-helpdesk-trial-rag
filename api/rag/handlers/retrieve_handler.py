from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langdetect import detect
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
pc_api_key = os.getenv("PINECONE_API_KEY")

# Set Google API key as environment variable for Google client
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    logging.info(f"Google API key set: {google_api_key[:10]}...")
else:
    logging.error("GOOGLE_API_KEY not found in environment variables")

# Initialize Pinecone client
pc = Pinecone(api_key=pc_api_key)

# Initialize embeddings once at module level
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    logging.info("GoogleGenerativeAIEmbeddings initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}")
    # Fallback: try without explicit key (rely on environment)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

# Function to retrieve documents using Pinecone
def retrieve_documents(question: str) -> dict:
    """
    Retrieve relevant context from Pinecone vector store using embeddings and similarity search.

    Args:
        question (str): The query string to search for.

    Returns:
        dict: {"context": retrieved_docs, "detected_language": detected_language}
    """
    # Detect language of the question
    detected_language = detect(question)
    logging.debug(f"Detected language: {detected_language}")

    # Connect to existing Pinecone index using global embeddings
    index_name = "rag-english"
    
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    # Perform similarity search
    retrieved_docs = vectorstore.similarity_search(question, k=25)
    return {"context": retrieved_docs, "detected_language": detected_language}

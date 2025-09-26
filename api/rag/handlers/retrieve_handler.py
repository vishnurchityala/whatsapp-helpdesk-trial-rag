from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langdetect import detect
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
hf_token = os.getenv("HF_TOKEN")
pc_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=pc_api_key)

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

    # Always use English documents for retrieval
    path_to_document = "api/rag/data/english_data.txt"
    index_name = "rag-english"

    # Load documents and create embeddings
    loader = TextLoader(path_to_document)
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"token": hf_token}
    )

    # Create or load Pinecone vector store
    if index_name not in [index.name for index in pc.list_indexes()]:
        vectorstore = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=index_name
        )
    else:
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )

    # Perform similarity search
    retrieved_docs = vectorstore.similarity_search(question, k=5)
    return {"context": retrieved_docs, "detected_language": detected_language}

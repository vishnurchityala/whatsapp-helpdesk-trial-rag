from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()


hf_token = os.getenv("HF_TOKEN")

def build_vectorstore(language: str, path: str):
    loader = TextLoader(path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    all_splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"token": hf_token}
    )

    persist_dir = f"api/rag/data/chroma_{language.lower()}"
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    return vectorstore

def retrieve(state: dict):
    """
    Retrieve relevant context from a language-specific document
    using embeddings and similarity search.

    Args:
        state (dict): Dictionary containing at least:
            - 'language': The language of the document (Gujarati, English, Hindi).
            - 'question': The query string to search for.

    Returns:
        dict: A dictionary containing the retrieved context as {"context": retrieved_docs}.
    """

    language_document_map = {
        "Gujarati": "api/rag/data/gujarati_data.txt",
        "English": "api/rag/data/english_data.txt",
        "Hindi": "api/rag/data/hindi_data.txt",
    }

    path_to_document = language_document_map.get(state["language"], None)

    if not path_to_document:
        raise ValueError(f"Unsupported language: {state['language']}")

    persist_dir = f"api/rag/data/chroma_{state['language'].lower()}"
    if not os.path.exists(persist_dir):
        vectorstore = build_vectorstore(state["language"], path_to_document)
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"token": hf_token}
        )
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    retrieved_docs = vectorstore.similarity_search(state["question"], k=10)
    return {"context": retrieved_docs}

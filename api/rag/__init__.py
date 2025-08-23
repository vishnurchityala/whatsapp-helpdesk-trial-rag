from api.rag.handlers.retrieve_handler import retrieve_documents
from api.rag.handlers.generate_handler import generate_response
from langchain_core.runnables import RunnableLambda

def merge_context_with_input(retrieve_output):
    """Merge retrieved context with original input for generate step."""
    def _merge(original_input):
        return {**original_input, **retrieve_output}
    return _merge

def create_rag_chain():
    """Create the complete RAG chain using LangChain runnables."""
    def process_input(input_message):
        question = input_message.get("question", "")
        language = input_message.get("language", "English")
        retrieved_docs = retrieve_documents(question)
        response = generate_response(retrieved_docs["context"], question, language)
        return response

    # Create chain: input -> process -> output
    chain = RunnableLambda(process_input)
    
    return chain

# Create the app as a runnable chain
app = create_rag_chain()
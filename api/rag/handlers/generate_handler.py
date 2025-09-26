from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import os
from dotenv import load_dotenv
from googletrans import Translator
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

# Load environment variables
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_GEMINI_API_KEY
)

# Initialize the translator
translator = Translator()

# Function to generate response using Google Gemini Flash
def generate_response(context: list, question: str, language: str) -> dict:
    """
    Generate an answer using Gemini (via LangChain) based on already retrieved context.

    Args:
        context (list): List of retrieved documents.
        question (str): The original question from the user.
        language (str): The target language for the response.

    Returns:
        dict: {"answer": response.content}
    """
    # Format context into a single string
    context_text = "\n\n".join([doc.page_content for doc in context])

    # Define the prompt template
    template = f"""
    You are a trusted digital assistant for a politician's WhatsApp helpline. 
    Your role is to provide accurate, concise, and easy-to-understand information 
    about government schemes, benefits, and related services. You are also a trusted digital assistant for a politician's WhatsApp helpline. 
    Your role is to provide accurate, concise, and easy-to-understand information 
    about government schemes, benefits, and related services. Listen to the citizen's question and provide a detailed answer. Including links to the official website if available or documents if available.
    If the citizen's question is not related to the context, politely say you don't have that information and suggest contacting the official helpline or visiting the official website.
    If the citizen's question is not clear, ask for more information.
    Remove salutations and signatures.

    Context:
    {context_text}

    Citizen's Question:
    {question}

    Instructions:
    - Strictly following this instruction: Answer in the language {language}
    - Always be polite, respectful, and supportive.
    - Focus only on providing relevant scheme/benefit information from the context.
    - **IMPORTANT: If there are any URLs, links, website addresses, or official portals mentioned in the context, ALWAYS include them in your response.**
    - **IMPORTANT: If there are reference numbers, application IDs, helpline numbers, or contact details in the context, include them in your response.**
    - Provide detailed and comprehensive answers including all available information from the context.
    - Include step-by-step procedures, eligibility criteria, required documents, and application processes if mentioned in the context.
    - If the answer is not in the context, politely say you don't have that information and suggest contacting the official helpline or visiting the official website.
    - Do not make up information or speculate.
    - Format links clearly and make them easily accessible for citizens.
    - **IMPORTANT: if said to list all list titles of all the schemes, list them in a bullet point format.**
    - Remove Hello and Salutation in the message.
    - **IMPORTANT: keep all texts in same weight no bolds and other things.**
    - **IMPORTANT: The final answer should be short, concise, and WhatsApp-friendly (preferably 1–3 sentences).**
    - Always include official scheme links if available, preferably at the end of the answer.
    - The answer should start directly with relevant information without greetings, introductions, or closings.

    Final Answer:
    """

    # Generate response
    response = llm.invoke(template)
    logging.debug(f"Generated response: {response.content}")

    # Translate the response to the target language if necessary
    if language != "English":
        translated_response = translator.translate(response.content, dest=language[:2].lower()).text
        logging.debug(f"Translated response to {language}: {translated_response}")
        return {"answer": translated_response}

    return {"answer": response.content}

def format_context(state: dict):
    """Format context documents into a single string."""
    context = state.get("context", [])
    context_text = "\n\n".join([doc.page_content for doc in context])
    # Use detected language if available, otherwise fall back to manual language or default
    language = state.get("detected_language") or state.get("language", "English")
    return {
        "context": context_text,
        "question": state["question"],
        "language": language
    }
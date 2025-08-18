from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

def generate(state: dict):
    """
    Generate an answer using Gemini (via LangChain) based on already retrieved context.
    """
    context = state["context"]

    context_text = "\n\n".join([doc.page_content for doc in context])

    template = """
    You are a trusted digital assistant for a politician's WhatsApp helpline. 
    Your role is to provide accurate, concise, and easy-to-understand information 
    about government schemes, benefits, and related services.

    Context:
    {context}

    Citizen's Question:
    {question}

    Instructions:
    - Answer in {language}, or in the same language as the question if {language} is not specified.
    - Always be polite, respectful, and supportive.
    - Focus only on providing relevant scheme/benefit information from the context.
    - If the answer is not in the context, politely say you don’t have that information and suggest contacting the official helpline or visiting the official website.
    - Do not make up information or speculate.

    Final Answer:
    """


    prompt = PromptTemplate(
        input_variables=["context", "question", "language"],
        template=template
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_GEMINI_API_KEY
    )

    formatted_prompt = prompt.format(
        context=context_text,
        question=state["question"],
        language=state["language"]
    )

    response = llm.invoke(formatted_prompt)

    return {"answer": response.content}
from api.rag.models import State
from api.rag.handlers import retrieve, generate
from langgraph.graph import StateGraph, END

def retrieve_node(state: State):
    """LangGraph node: retrieve context for the question."""
    result = retrieve(state)
    return {"context": result["context"]}

def generate_node(state: State):
    """LangGraph node: generate answer using Gemini."""
    result = generate(state)
    return {"answer": result["answer"]}

workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
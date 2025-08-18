from typing_extensions import List, TypedDict
from langchain_core.documents import Document

"""
State class used by the LangGraph Agent. 
This object is passed through each node in the LangGraph to maintain context, 
carry forward relevant data, and support prompt generation throughout the workflow.
"""
class State(TypedDict):
    question: str
    language: str
    context: List[Document]
    answer: str
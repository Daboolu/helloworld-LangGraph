from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph
load_dotenv()
import os

llm_model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0,
)

def show_flow(agent: CompiledStateGraph, filename:str):
    """Utility to show the flow graph of an agent"""
    img = agent.get_graph().draw_mermaid_png()
    with open(f"./assets/reproduce/{filename}.png", "wb") as f:
        f.write(img)
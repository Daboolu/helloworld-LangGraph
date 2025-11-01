from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph
import datetime
from dataclasses import is_dataclass, asdict

load_dotenv()
import os
import json

llm_model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0,
)


def get_llm_model():
    """Get LLM model instance"""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
    )
    return llm


def show_flow(agent: CompiledStateGraph, filename: str):
    """Utility to show the flow graph of an agent"""
    img = agent.get_graph().draw_mermaid_png()
    with open(f"./assets/reproduce/{filename}.png", "wb") as f:
        f.write(img)


def print_snapshot_json(snapshot):
    """JSON  StateSnapshot"""

    def safe_serialize(obj):
        # 1. LangChain message
        if hasattr(obj, "content") and hasattr(obj, "__class__"):
            return {
                "type": obj.__class__.__name__,
                "content": getattr(obj, "content", None),
                "name": getattr(obj, "name", None),
                "tool_calls": getattr(obj, "tool_calls", None),
            }

        # 2. pydantic v2
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            try:
                return safe_serialize(obj.model_dump())
            except Exception:
                return safe_serialize(vars(obj))

        # 3. dataclass
        if is_dataclass(obj):
            try:
                return safe_serialize(asdict(obj))
            except Exception:
                return safe_serialize(vars(obj))

        # 4. dict
        if isinstance(obj, dict):
            return {safe_serialize(k): safe_serialize(v) for k, v in obj.items()}

        # 5. list / tuple / set
        if isinstance(obj, (list, tuple, set)):
            return [safe_serialize(v) for v in obj]
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, BaseException):
            return f"{obj.__class__.__name__}: {obj}"
        if hasattr(obj, "__dict__"):
            return {k: safe_serialize(v) for k, v in vars(obj).items()}
        return obj

    data = {
        "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
        "thread_id": snapshot.config["configurable"].get("thread_id"),
        "parent_id": (
            snapshot.parent_config["configurable"].get("checkpoint_id")
            if snapshot.parent_config
            else None
        ),
        "created_at": snapshot.created_at,
        "metadata": safe_serialize(snapshot.metadata),
        "values": safe_serialize(snapshot.values),
        "next_nodes": safe_serialize(snapshot.next),
        "interrupts": safe_serialize(snapshot.interrupts),
        "tasks": safe_serialize(getattr(snapshot, "tasks", [])),
    }

    print(json.dumps(data, indent=2, ensure_ascii=False))

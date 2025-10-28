import os, operator
from typing import Literal
from typing_extensions import TypedDict, Annotated

from reproduce.utils import llm_model as model, print_snapshot_json, show_flow # LLM and tools
from langchain.tools import tool
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

# ====== 1) choose Checkpointer======
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()  


# ====== 2) define Tools======
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`."""
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


# ====== 3) define state and nodes =======
class MessagesState(TypedDict):

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def llm_call(state: dict):

    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

def tool_node(state: dict):

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        func = tools_by_name[tool_call["name"]]
        observation = func.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:

    last_message = state["messages"][-1]
    return "tool_node" if getattr(last_message, "tool_calls", None) else END


# ====== 4) construct and compile graph (add checkpointer)======
def build_agent():
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")
    # key point： input checkpointer to compile → memory checkpoint
    return agent_builder.compile(checkpointer=checkpointer)


def print_history(agent, thread_id: str):
    """print whole checkpoint history (step/ckp_id/parent/values)"""
    print(f"\n🧩 Checkpoint history for thread_id={thread_id}\n" + "="*90)
    history = agent.get_state_history({"configurable": {"thread_id": thread_id}})
    for snap in history:
        cid = snap.config["configurable"].get("checkpoint_id")
        pid = snap.parent_config["configurable"].get("checkpoint_id") if snap.parent_config else None
        step = (snap.metadata or {}).get("step")
        values = snap.values
        print(f"Step={step:<3}  ckpt={cid}  parent={pid}")
        print(f"  values.keys: {list(values.keys()) if isinstance(values, dict) else type(values)}")
        print(f"  created_at : {snap.created_at}")
        print("-"*90)

def replay_history(agent, thread_id: str):
    """
    Replay：print messages by history snapshot 
    """
    print(f"\n🔁 Replay snapshots for thread_id={thread_id}\n" + "="*90)
    history = agent.get_state_history({"configurable": {"thread_id": thread_id}})
    for snap in history:
        step = (snap.metadata or {}).get("step")
        cid = snap.config["configurable"].get("checkpoint_id")
        print(f"\n[Step {step}] checkpoint_id={cid}")
        messages = (snap.values or {}).get("messages", [])
        # 尝试友好打印每条消息
        for i, m in enumerate(messages):
            role = getattr(m, "type", None) or m.__class__.__name__
            content = getattr(m, "content", "")
            print(f"  - msg[{i}] role={role}: {content}")
        print("-"*90)


# ====== 6) demo：first run →  checkpoint → Replay =======
def demo():
    agent = build_agent()

    current_file_name = os.path.basename(__file__)
    show_flow(agent, current_file_name)
    thread_id = "t1"
    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    print("\n▶ Run #1: 'Add 3 and 4.'")
    result = agent.invoke({"messages": [HumanMessage(content="Add 3 and 4.")]}, cfg)
    # print result messages
    for m in result["messages"]:
        try:
            m.pretty_print()
        except Exception:
            print(f"[{m.__class__.__name__}] {getattr(m, 'content', '')}")

    # latest checkpoint
    latest_snap = agent.get_state(cfg)
    print("\n🧾 Latest snapshot (JSON):")
    print_snapshot_json(latest_snap)
    last_ckp_id = latest_snap.config["configurable"]["checkpoint_id"]

    # --- B. history & replay ---
    print_history(agent, thread_id)
    replay_history(agent, thread_id)

    # --- C. parent checkpoint （Branch）---
    #  parent as start point + new input from user
    parent_ckp_id = latest_snap.parent_config["configurable"]["checkpoint_id"]
    # annotation only: you can get grandparent if needed for understandering the checkoutpoint chain
    # cfg_gradparent: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": parent_ckp_id}}
    # gradparent_snap = agent.get_state(cfg_gradparent)
    # gradparent_snap_cptid = gradparent_snap.parent_config["configurable"]["checkpoint_id"]
    # branch_cfg remember to use  gradparent_snap_cptid as checkpoint_id
    print("\n🌿 Branch from parent checkpoint (after node_a): 'Multiply 6 by 7.'")
    branch_cfg: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": parent_ckp_id}}
    branch_result = agent.invoke({"messages": [HumanMessage(content="Multiply it's add result by 7.")]}, branch_cfg)

    for m in branch_result["messages"]:
        try:
            m.pretty_print()
        except Exception:
            print(f"[{m.__class__.__name__}] {getattr(m, 'content', '')}")

    # After branch, print latest snapshot & history
    branch_latest = agent.get_state({"configurable": {"thread_id": thread_id}})
    print("\n🧾 Latest snapshot after branch (JSON):")
    print_snapshot_json(branch_latest)

    print("="*10)
    branch_history = agent.get_state_history({"configurable": {"thread_id": thread_id}})
    for i, branch_snap in enumerate(branch_history):
        print(f"\n--- Branch Snapshot #{i} ---")
        print_snapshot_json(branch_snap)


if __name__ == "__main__":
    demo()
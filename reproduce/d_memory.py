"""
LangGraph + Store Demo:
This example extends the original joke-generation workflow with:
- InMemorySaver:  preserves graph state within threads (short-term memory)
- InMemoryStore:  retains user-specific information across threads (long-term memory)
"""
import os
import uuid
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore, BaseStore
from langchain_core.runnables import RunnableConfig
from reproduce.utils import llm_model as llm, show_flow
from langchain.embeddings import init_embeddings

# Use a lightweight HuggingFace model for local embedding
embed_model = init_embeddings("huggingface:sentence-transformers/all-MiniLM-L6-v2")


# ========== Define State ==========
class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str


# ========== Define Node Functions ==========
def generate_joke(state: State, config: RunnableConfig, *, store: BaseStore):
    """Generate an initial joke, referencing previous user memories if available."""
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")

    # Retrieve previous jokes related to the same topic
    old_jokes = store.search(namespace, query=state["topic"], limit=2)
    context = (
        "\n".join([mem.value["joke"] for mem in old_jokes])
        if old_jokes
        else "No prior jokes found."
    )

    msg = llm.invoke(
        f"Refer to the following previous jokes, then create a new one:\n{context}\nTopic: {state['topic']}"
    )
    return {"joke": msg.content}


def check_punchline(state: State):
    """Check whether the joke contains a punchline."""
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"


def improve_joke(state: State):
    """Enhance the joke with wordplay."""
    msg = llm.invoke(f"Make this joke funnier using clever wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def polish_joke(state: State, config: RunnableConfig, *, store: BaseStore):
    """Polish the improved joke, then store it in long-term memory."""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    final_joke = msg.content

    # Save the final joke as a new user memory
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    store.put(
        namespace,
        memory_id,
        {"joke": final_joke, "topic": state["topic"]},
        index=["joke"],
    )

    return {"final_joke": final_joke}


# ========== Construct Workflow ==========
workflow = StateGraph(State)

workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)


# ========== Define Checkpointer and Store ==========
checkpointer = InMemorySaver()

# Use CPU-based embeddings for demonstration
in_memory_store = InMemoryStore(
    index={"embed": embed_model, "dims": 384, "fields": ["joke", "$"]}
)

# Alternative (requires OpenAI API key)
# in_memory_store = InMemoryStore(
#     index={
#         "embed": "openai:text-embedding-3-small",
#         "dims": 1536,
#         "fields": ["joke", "$"],
#     }
# )

# Compile workflow
graph = workflow.compile(checkpointer=checkpointer, store=in_memory_store)

# Visualize the workflow structure
show_flow(graph, os.path.basename(__file__))


# ========== Run Example ==========
user_id = "user_1"

config1 = {"configurable": {"thread_id": "thread_1", "user_id": user_id}}
state = graph.invoke({"topic": "cats"}, config1)
print("\n🐱 [Run 1] Generated joke:")
print(state["joke"])
if "final_joke" in state:
    print("\n🎭 Final version:")
    print(state["final_joke"])

# Second run (different thread, same user)
config2 = {"configurable": {"thread_id": "thread_2", "user_id": user_id}}
state2 = graph.invoke({"topic": "dogs"}, config2)
print("\n🐶 [Run 2] Generated joke with stored memories:")
print(state2["joke"])
if "final_joke" in state2:
    print("\n🎭 Final version:")
    print(state2["final_joke"])

# Inspect stored memories
memories = in_memory_store.search((user_id, "memories"))
print(f"\n🧠 Total stored memories: {len(memories)}")
for i, mem in enumerate(memories, 1):
    print(f"{i}. {mem.value['topic']} → {mem.value['joke'][:60]}...")

"""
LangGraph Streaming Demo (Dual Model Version)
Demonstrates streaming outputs from two LLMs (story + poem)
and printing each token prefixed with its corresponding model tag.
"""

import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from reproduce.utils import get_llm_model, show_flow
from copy import deepcopy

# 1️⃣ Define the shared State structure
class State(TypedDict):
    topic: str
    story: str
    poem: str


# 2️⃣ Create two tagged model instances
story_model = get_llm_model()
story_model.tags = ["story"]

poem_model = get_llm_model()
poem_model.tags = ["poem"]


async def write_story(state, config):
    topic = state["topic"]
    response = await story_model.ainvoke(
        [{"role": "user", "content": f"Write a short story about {topic}"}],
        config,
    )
    return {"story": response.content}


async def write_poem(state, config):
    topic = state["topic"]
    response = await poem_model.ainvoke(
        [{"role": "user", "content": f"Write a short poem about {topic}"}],
        config,
    )
    return {"poem": response.content}


# 4️⃣ Build LangGraph with two parallel nodes
graph = (
    StateGraph(State)
    .add_node("write_story", write_story)
    .add_node("write_poem", write_poem)
    .add_edge(START, "write_story")
    .add_edge(START, "write_poem")  # run concurrently
    .add_edge("write_story", END)
    .add_edge("write_poem", END)
    .compile()
)

show_flow(graph, os.path.basename(__file__))


import asyncio

async def main():
    print("🎬 Start generating story & poem...\n")

    async for msg_chunk, metadata in graph.astream(
        {"topic": "Spring"},
        stream_mode="messages",
    ):
        if msg_chunk.content:
            tag = metadata.get("tags", ["unknown"])[0]
            print(f"\n[{tag.upper()}]\n {msg_chunk.content}", end="", flush=True)

    print("\n\n✅ Completed!")

if __name__ == "__main__":
    asyncio.run(main())

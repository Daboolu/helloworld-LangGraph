"""
Email triage with LangGraph + your own LLM:
- LLM entry: from reproduce.llm import llm_model as model
- LLM handles: intent/urgency classification (structured JSON) + reply draft generation
- Others: human review interrupt/resume, configurable retry policy
Run: python email_agent_llm.py
"""

from __future__ import annotations
from typing import TypedDict, Literal, List, Dict, Any
import json
import re

# Your LLM (ensure reproduce/llm.py exists and provides llm_model object)
from reproduce.utils import llm_model as model, show_flow

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, RetryPolicy, interrupt
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------
# 1) State definitions
# -----------------------------

class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict, total=False):
    # Raw email data
    email_content: str
    sender_email: str
    email_id: str

    # Classification result
    classification: EmailClassification | None

    # Raw search/API results
    search_results: List[str] | None
    customer_history: Dict[str, Any] | None

    # Generated content
    draft_response: str | None
    messages: List[str] | None

    # Optionals
    current_step: str | None
    customer_id: str | None
    tool_result: str | None

# -----------------------------
# 2) Mock services (minimal runnable)
# -----------------------------

def fake_search_kb(query: str) -> List[str]:
    base = [
        "Reset password via Settings > Security > Change Password",
        "Password must be at least 12 characters",
        "Include uppercase, lowercase, numbers, and symbols",
        "Billing portal: Settings > Billing; Invoices tab shows charges",
        "Refund policy: refunds within 30 days for duplicate charges",
        "Report a bug: Help > Report Issue; attach logs if possible",
    ]
    q = query.lower().split()
    hits = [x for x in base if any(tok in x.lower() for tok in q if tok)]
    return hits or base[:2]

class EmailService:
    @staticmethod
    def send(text: str) -> None:
        print("=== SENDING EMAIL ===")
        print(text)
        print("=====================")

email_service = EmailService()

# -----------------------------
# 3) LLM helpers
# -----------------------------

def _to_text(llm_output: Any) -> str:
    """Compatible with multiple return types: str / object.content / object['content']"""
    if llm_output is None:
        return ""
    if isinstance(llm_output, str):
        return llm_output
    if hasattr(llm_output, "content"):
        return getattr(llm_output, "content")
    if isinstance(llm_output, dict) and "content" in llm_output:
        return llm_output["content"]
    return str(llm_output)

def call_llm(prompt: str) -> str:
    """Call your model.invoke and return text output"""
    out = model.invoke(prompt)
    return _to_text(out).strip()

def call_llm_json(prompt: str) -> Dict[str, Any]:
    """
    Request the LLM to output valid JSON; apply robust parsing (strip fences, fix noise).
    The LLM must return STRICT JSON (no prefix/suffix text).
    """
    out = call_llm(prompt)
    extracted = out
    fence = re.search(r"```json\s*(\{.*?\})\s*```", out, re.S | re.I)
    if fence:
        extracted = fence.group(1)
    else:
        brace = re.search(r"(\{.*\})", out, re.S)
        if brace:
            extracted = brace.group(1)
    try:
        return json.loads(extracted)
    except Exception:
        # Fallback: fix trailing commas/single quotes
        cleaned = extracted.replace("'", '"')
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            # Final fallback: return empty dict
            return {}

# -----------------------------
# 4) Nodes
# -----------------------------

def read_email(state: EmailAgentState) -> Dict[str, Any]:
    msg = f"Processing email from {state.get('sender_email','')} with id {state.get('email_id','')}"
    return {"messages": [msg]}

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """
    Use your LLM to classify intent/urgency, and return a JSON:
    {
        "intent": "question|bug|billing|feature|complex",
        "urgency": "low|medium|high|critical",
        "topic": "<string>",
        "summary": "<string>"
    }
    """
    email = state.get("email_content", "")
    sender = state.get("sender_email", "")
    prompt = f"""
                You are an assistant that classifies customer support emails into a fixed schema.
                Return ONLY a valid JSON object (no extra text) with keys:
                - intent: one of ["question", "bug", "billing", "feature", "complex"]
                - urgency: one of ["low", "medium", "high", "critical"]
                - topic: short string
                - summary: one sentence summary (<= 200 chars)

                Email content:
                {email}

                From: {sender}
            """

    cls = call_llm_json(prompt)
    # Fill missing fields to avoid downstream errors
    intent = cls.get("intent", "complex")
    urgency = cls.get("urgency", "medium")
    topic = cls.get("topic", "general")
    summary = cls.get("summary", (email or "")[:140].replace("\n", " "))

    classification: EmailClassification = {
        "intent": intent, "urgency": urgency, "topic": topic, "summary": summary
    }

    if intent == "billing" or urgency in ["critical"]:
        goto = "human_review"
    elif intent in ["question", "feature"]:
        goto = "search_documentation"
    elif intent == "bug":
        goto = "bug_tracking"
    else:
        goto = "draft_response"
    return Command(update={"classification": classification}, goto=goto)

def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Simple KB retrieval (local mock)"""
    classification = state.get('classification', {}) or {}
    query = f"{classification.get('intent','')} {classification.get('topic','')}".strip()
    try:
        search_results = fake_search_kb(query)
    except Exception as e:
        search_results = [f"Search temporarily unavailable: {e}"]
    return Command(update={"search_results": search_results}, goto="draft_response")

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Simulate ticket creation"""
    ticket_id = "12345"
    return Command(
        update={
            "search_results": [f"Bug ticket BUG-{ticket_id} created"],
            "current_step": "bug_tracked"
        },
        goto="draft_response"
    )

def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Use your LLM to draft an email reply"""
    classification = state.get('classification', {}) or {}

    context_sections = []
    if state.get('search_results'):
        formatted_docs = "\n".join([f"- {doc}" for doc in state['search_results']])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")

    if state.get('customer_history'):
        context_sections.append(
            f"Customer tier: {state['customer_history'].get('tier','standard')}"
        )

    context_text = "\n\n".join(context_sections)

    prompt = f"""
                You are a professional support engineer. Draft a helpful, concise, and polite reply to the customer.
                Follow these rules:
                - Address the user's specific concern clearly
                - If relevant docs are provided, incorporate them
                - Keep it under 180 words
                - No placeholders; ready to send

                Customer email:
                {state.get('email_content','')}

                Classification:
                intent={classification.get('intent','unknown')}, urgency={classification.get('urgency','medium')}, topic={classification.get('topic','general')}

                Context:
                {context_text or '(no extra context)'}

                Reply:
            """

    reply = call_llm(prompt)
    reply = reply.strip()
    if not reply:
        reply = "Hello,\n\nThanks for reaching out. We are looking into your request and will get back to you shortly.\n\nBest regards,\nSupport Team"

    needs_review = (
        classification.get('urgency') in ['high', 'critical'] or
        classification.get('intent') == 'complex'
    )
    goto = "human_review" if needs_review else "send_reply"

    return Command(update={"draft_response": reply}, goto=goto)

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Human review: interrupt to pause, resume with approval result"""
    classification = state.get('classification', {}) or {}

    human_decision = interrupt({
        "email_id": state.get('email_id',''),
        "original_email": state.get('email_content',''),
        "draft_response": state.get('draft_response',''),
        "urgency": classification.get('urgency',''),
        "intent": classification.get('intent',''),
        "action": "Please review and approve/edit this response (set approved: true/false; optional edited_response)"
    })

    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response',''))},
            goto="send_reply"
        )
    else:
        return Command(update={}, goto=END)

def send_reply(state: EmailAgentState) -> Dict[str, Any]:
    """Send email (mock)"""
    email_service.send(state.get("draft_response","(empty)"))
    return {}

# -----------------------------
# 5) Build graph
# -----------------------------

workflow = StateGraph(EmailAgentState)

workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("search_documentation", search_documentation, retry_policy=RetryPolicy(max_attempts=3))
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

# Use in-memory checkpointer to support interrupt/resume across invocations
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
import os
# Show the agent flow graph
current_file_name = os.path.basename(__file__)
show_flow(app, current_file_name)

# -----------------------------
# 6) Demo
# -----------------------------
if __name__ == "__main__":
    print("=== 📨 Email Agent Simulation ===")
    print("Please simulate a customer email (e.g., 'I was charged twice for my subscription!').")
    print("The system will call the LLM to infer intent, urgency, and decide the processing flow.\n")

    email_content = input("Customer Email Content:\n> ").strip()
    if not email_content:
        email_content = "I was charged twice for my subscription! This is urgent! please help."

    sender_email = input("\nSender Email (default: customer@example.com): ").strip() or "customer@example.com"
    email_id = "email_" + str(abs(hash(email_content)))[:6]

    initial_state: EmailAgentState = {
        "email_content": email_content,
        "sender_email": sender_email,
        "email_id": email_id,
        "messages": []
    }

    config = {"configurable": {"thread_id": email_id}}

    print("\n--- 🧠 Running LangGraph Workflow ---\n")

    result = app.invoke(initial_state, config)

    classification = result.get("classification", {})
    print("📋 Classification result:")
    for k, v in classification.items():
        print(f"  - {k}: {v}")

    if "draft_response" in result:
        print("\n--- ✍️ Draft Response ---\n")
        print(result["draft_response"][:600])
    else:
        print("\n(No draft generated — possibly ended early.)")

    for action in app.get_state(config).next:
        if "human_review" == action:
            print("\n--- 🧑‍💼 Human Review Phase ---")
            print("The model marked this email for human review, e.g.,")
            print(" - urgent billing, complex complaint, or ambiguous language.")
            print("You can approve (and optionally edit) or reject auto-reply.")

            approved = input("\nApprove the draft? (y/n): ").strip().lower()
            if approved == "y":
                print("\nOptional: edit the draft (leave empty to keep original)")
                edited = input("Edited Response:\n> ").strip()
                if not edited:
                    edited = result.get("draft_response", "")
                human_response = Command(
                    resume={"approved": True, "edited_response": edited}
                )
            else:
                print("You chose to reject; the email will not be auto-replied.")
                human_response = Command(resume={"approved": False})

            final_result = app.invoke(human_response, config)
            print("\n✅ Email sent successfully (after review)!")
        else:
            print("\n✅ Email processed automatically (no human review needed).")

    print("\n--- 🧾 Workflow Summary ---")
    print(f"Intent: {classification.get('intent', 'unknown')}")
    print(f"Urgency: {classification.get('urgency', 'medium')}")
    print(f"Topic: {classification.get('topic', 'general')}")
    print("\nSimulation complete.\n")
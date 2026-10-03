"""Internal workflow: general questions and conversation that need no action on the computer or web."""
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

MODEL = os.getenv("INTERNAL_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = """
You are the assistant inside a desktop agent. Answer the user's question or reply to their message
directly and concisely. You can't act here: if they ask you to open, find, send or change something,
say that in one line and suggest they ask for it as a request, e.g. "open excel" or "find my resume".
If the message includes results of earlier steps, use them to answer (e.g. summarise what was found).
"""


def run(task):
    """Returns {"output", "failed"}, like every workflow."""
    llm = ChatGroq(model=MODEL, temperature=0.3)
    try:
        answer = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=task)]).content
    except Exception as e:
        return {"output": f"model error: {type(e).__name__}: {str(e)[:200]}", "failed": True}
    return {"output": answer.strip(), "failed": False}

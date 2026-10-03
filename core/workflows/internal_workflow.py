"""Internal workflow: questions answered from knowledge instead of the web, files or apps. Company questions
(people, who reports to whom, clients, policies) come from the company tables in Postgres; anything else
from general knowledge."""
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import db

MODEL = os.getenv("INTERNAL_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = """
You are the assistant inside a company's desktop agent. Other parts of the agent use the web, files and
apps; you answer from knowledge: the company context below and general knowledge.
- Questions about the company, its people, teams, who reports to whom, clients, vendors or policies:
  answer ONLY from the company context. "I", "me" and "my" mean the user it names. If the answer isn't
  there, say you don't have that information. Never invent people, emails or policies.
- General questions: answer briefly from general knowledge.
- If the message includes results of earlier steps, use them.
- If asked to do something (open, send, find a file), say in one line to ask for it as a request.
Be short: answer the question, then stop.

Company context:
{context}
"""


def run(task):
    """Returns {"output", "failed"}, like every workflow. The tables are read on every call, so edits apply at once."""
    context = db.company_context() or "(none available)"
    llm = ChatGroq(model=MODEL, temperature=0)
    try:
        answer = llm.invoke([SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
                             HumanMessage(content=task)]).content
    except db.TRANSIENT:
        raise  # network or rate limit: the orchestrator pauses the task so it can be resumed
    except Exception as e:
        return {"output": f"model error: {type(e).__name__}: {str(e)[:200]}", "failed": True}
    return {"output": answer.strip(), "failed": False}

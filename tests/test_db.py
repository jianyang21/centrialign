"""Checks against the real Postgres container (docker compose up -d). Skips when it isn't reachable.
A "restart" here is a brand-new connection pool and checkpointer reading what the old one saved.
Run: python tests/test_db.py"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # core/
from dotenv import dotenv_values
import groq
import httpx
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt

URL = os.getenv("DATABASE_URL") or dotenv_values(Path(__file__).resolve().parents[2] / ".env").get("DATABASE_URL", "")
os.environ.setdefault("DATABASE_URL", URL)
import db
import orchestrator as o

PREFIX = "test-db-"


def fresh_app():
    """What a new process gets: its own pool and checkpointer over the same database."""
    pool = ConnectionPool(URL, open=False, min_size=1, kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    pool.open(wait=True, timeout=10)
    return o.graph.compile(checkpointer=PostgresSaver(pool)), pool


def cleanup():
    db.execute("DELETE FROM tasks WHERE thread_id LIKE %s", PREFIX + "%")
    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
        db.execute(f"DELETE FROM {table} WHERE thread_id LIKE %s", PREFIX + "%")


def classify_as(*steps):
    class Fake:
        def __init__(self, **kw): pass
        def with_structured_output(self, schema, method): return self
        def invoke(self, messages): return o.IntentClassifier(steps=[{"workflow": w, "task": t} for w, t in steps])
    o.ChatGroq = Fake


def test_all():
    # under pytest the modules are shared with the other test files: swap in, and put everything back after
    saved = (db.URL, o.app, o.ChatGroq, o.desktop_workflow.run, o.browser_workflow.run)
    db.URL = URL
    db.pool.cache_clear()
    o.db.checkpointer().setup()
    cleanup()
    try:
        # schema + seed: the company context renders from the tables
        context = db.company_context()
        assert 'The user ("I", "me"): Arjun Mehta' in context and "- Arjun Mehta | Software Engineer | Platform | Meera Nair |" in context
        assert "Acme Retail (client): contact Jane Doe <jane.doe@acme.example>" in context

        # a network error mid-plan: the task row says interrupted, and a restarted process finishes it
        # without redoing the finished step
        steps_run = []
        def flaky(task):
            steps_run.append(task)
            if len(steps_run) == 2:
                raise groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))
            return {"output": f"did {task.splitlines()[0]}", "failed": False}
        o.desktop_workflow.run = o.browser_workflow.run = flaky
        classify_as(("Desktop_Workflow", "take a screenshot"), ("Browser_Workflow", "email it to John"))
        o.app, pool = fresh_app()
        tid = PREFIX + "network"
        db.task_started(tid, "take a screenshot and email it to John")
        assert o.drive(tid, {"user_request": "x"}).startswith("Paused by a network error (APIConnectionError)")
        row = db.execute("SELECT status, error FROM tasks WHERE thread_id = %s", tid)[0]
        assert row["status"] == "interrupted" and row["error"].startswith("APIConnectionError"), row
        assert [t["thread_id"] for t in db.unfinished() if t["thread_id"].startswith(PREFIX)] == [tid]
        pool.close()

        o.app, pool = fresh_app()  # "restart"
        # resume only this test's task, exactly as --resume does; never resume_all(): that would pick up
        # the user's own unfinished tasks and finish them with this test's fakes
        print(o.drive(tid, None))
        row = db.execute("SELECT status, results FROM tasks WHERE thread_id = %s", tid)[0]
        assert row["status"] == "done", row
        assert [r["output"] for r in row["results"]] == ["did take a screenshot", "did email it to John"], row
        assert len(steps_run) == 3  # screenshot once, email cut off once, email again: nothing redone
        pool.close()

        # a question waiting for an answer survives a restart and resumes with the answer
        def asks(task):
            return {"output": f"answered {interrupt('Close Calculator? (yes/no)')}", "failed": False}
        o.desktop_workflow.run = asks
        classify_as(("Desktop_Workflow", "close the calculator"))
        o.app, pool = fresh_app()
        config = {"configurable": {"thread_id": PREFIX + "question"}}
        assert o.app.invoke({"user_request": "x"}, config)["__interrupt__"][0].value == "Close Calculator? (yes/no)"
        pool.close()
        o.app, pool = fresh_app()  # "restart"
        assert o.app.get_state(config).interrupts[0].value == "Close Calculator? (yes/no)"
        assert o.app.invoke(Command(resume="yes"), config)["results"][0]["output"] == "answered yes"
        pool.close()
    finally:
        cleanup()
        db.URL, o.app, o.ChatGroq, o.desktop_workflow.run, o.browser_workflow.run = saved
        db.pool.cache_clear()


if __name__ == "__main__":
    try:
        if not URL:
            raise RuntimeError("DATABASE_URL is not set")
        db.URL = URL
        db.execute("SELECT 1")
    except Exception as e:
        print(f"skipped: database not reachable ({type(e).__name__}); start it with: docker compose up -d")
        sys.exit()
    test_all()
    print("all checks passed")

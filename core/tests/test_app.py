"""Offline checks for app.py, with a fake agent: the page and the worker thread, no LLM, browser or database.
Run: python tests/test_app.py  (or: python -m pytest tests)"""
import sys
import threading
import time
import types
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
threads = []  # the thread each task ran on: must always be the same one


def fake_modules():
    db = types.ModuleType("db")
    db.TRANSIENT = (ConnectionError,)
    db.unfinished = lambda: []
    db.task_started = lambda tid, request: True
    db.task_update = lambda *a, **k: None
    db.start_database = lambda say=print: None
    o = types.ModuleType("orchestrator")
    def drive(thread_id, first, ask):
        threads.append(threading.current_thread())
        print("  [browser] goto https://mail.google.com -> ok")  # a workflow's progress line
        if first["user_request"] == "who is my boss":
            return "Meera Nair"
        return f"answered {ask('Proceed? (yes / no / or say what to change)')}"
    o.drive = drive
    o.resume = lambda t, ask: "resumed"
    sys.modules.update(db=db, orchestrator=o)


def until(at, check, seconds=10):
    """Rerun the page until check(at) holds: the worker answers on its own thread."""
    for _ in range(int(seconds * 10)):
        at.run()
        if check(at):
            return at
        time.sleep(0.1)
    raise AssertionError(f"timed out: {[m.markdown[0].value for m in at.chat_message if m.markdown]}")


def said(at):
    return [m.markdown[0].value for m in at.chat_message if m.markdown]


def test_app():
    real = {k: sys.modules.get(k) for k in ("db", "orchestrator")}
    fake_modules()
    try:
        check()
    finally:  # under pytest the other test files share sys.modules
        for k, m in real.items():
            if m:
                sys.modules[k] = m
            else:
                sys.modules.pop(k, None)


def check():
    at = until(AppTest.from_file(APP, default_timeout=10), lambda at: "Ready" in [c.value for c in at.caption])
    # a question-free request: the reply appears, no answer buttons
    at.chat_input[0].set_value("who is my boss").run()
    at = until(at, lambda at: "Meera Nair" in said(at))
    assert not [b for b in at.button if b.key and b.key.startswith("a-")]
    assert "[browser] goto https://mail.google.com -> ok" in [t.value for t in at.text]  # progress shows

    # a request that asks: the question and the Yes/No/Done/Cancel buttons appear, the input box answers it
    at.chat_input[0].set_value("email my boss").run()
    at = until(at, lambda at: any("Proceed?" in t.value for t in at.text))
    assert "Answer the question" in at.chat_input[0].placeholder
    at.button(key="a-yes").click().run()
    at = until(at, lambda at: "answered yes" in said(at))
    assert said(at).count("answered yes") == 1 and "Ask the agent" in at.chat_input[0].placeholder

    # a typed answer goes to the task, not to a new task
    at.chat_input[0].set_value("email my boss").run()
    at = until(at, lambda at: "Answer the question" in at.chat_input[0].placeholder)
    at.chat_input[0].set_value("make it shorter").run()
    at = until(at, lambda at: "answered make it shorter" in said(at))

    # every task ran on one and the same thread (Playwright and COM need that), never a page thread
    assert len(set(threads)) == 1 and threads[0].name != threading.main_thread().name


if __name__ == "__main__":
    test_app()
    print("all checks passed")

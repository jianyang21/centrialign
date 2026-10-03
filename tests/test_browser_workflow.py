"""Offline checks for browser_workflow. Run: python tests/test_browser_workflow.py  (needs: playwright install chromium)"""
import itertools
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # core/, where workflows/ and guards.py live
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from playwright.sync_api import sync_playwright
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
from workflows import browser_workflow as bw

PAGE = """<label>To <input></label> <label>Password <input type=password></label>
<button onclick="document.title='sent'">Send</button>
<button onclick="document.title='saved'">Save draft</button>"""

seen = []  # what the fake LLM was shown, per call

def script(*actions):
    """Fake LLM: returns these actions in order, then repeats the last one forever."""
    seen.clear()
    it, last = iter(actions), {}
    class Fake:
        def __init__(self, **kw): pass
        def with_structured_output(self, schema, method): return self
        def invoke(self, messages):
            nonlocal last
            seen.append(messages[-1].content)
            last = next(it, last)
            try:
                return bw.Action(**last)
            except ValidationError as e:  # the real json_mode parser wraps it like this
                raise OutputParserException(str(e))
    bw.ChatGroq = Fake

app = bw.graph.compile(checkpointer=InMemorySaver())
_ids = itertools.count()

def run(task, *answers):
    bw.current_page().set_content(PAGE)
    config = {"configurable": {"thread_id": f"b{next(_ids)}"}, "recursion_limit": bw.RECURSION_LIMIT}
    r = app.invoke({"task": task}, config)
    for a in answers:
        assert "__interrupt__" in r, r
        r = app.invoke(Command(resume=a), config)
    return r

def test_all():
    bw._ctx = sync_playwright().start().chromium.launch().new_context()  # headless, no profile
    page = bw.current_page
    done = {"tool": "done", "text": "finished"}

    # the LLM sees roles and names; actions find elements by them
    script({"tool": "type", "role": "textbox", "name": "To", "text": "john@example.com"}, done)
    assert run("fill in John")["output"] == "finished"
    assert 'button "Send"' in seen[0], seen[0]
    assert page().get_by_role("textbox", name="To").input_value() == "john@example.com"

    # Send asks first even when the LLM didn't flag it; nothing happens while waiting
    send = {"tool": "click", "role": "button", "name": "Send"}
    script(send, done)
    assert "Proceed?" in run("send it")["__interrupt__"][0].value and page().title() != "sent"
    # "no" sends nothing and tells the LLM why
    script(send, done)
    r = run("send it", "no")
    assert page().title() != "sent" and "declined by user" in r["history"][0] and "declined" in seen[-1]
    script(send, done)
    run("send it", "yes")
    assert page().title() == "sent"
    # the question shows what was typed and the summary; feedback revises the draft and asks again
    reply = ({"tool": "type", "role": "textbox", "name": "To", "text": "Hi John, long draft"},
             dict(send, summary="Send reply to John"), {"tool": "type", "role": "textbox", "name": "To", "text": "Hi John"},
             dict(send, summary="Send reply to John"), done)
    script(*reply)
    q = run("reply to John")["__interrupt__"][0].value
    assert "Send reply to John" in q and "To: Hi John, long draft" in q, q
    script(*reply)
    r = run("reply to John", "make it shorter")
    q = r["__interrupt__"][0].value
    assert "declined by user: make it shorter" in seen[-2] and "To: Hi John\n" in q and "long draft" not in q, q
    assert page().title() != "sent"
    # an action the LLM flags as irreversible asks too, whatever its name
    script({"tool": "press", "text": "Enter", "irreversible": True}, done)
    assert "Proceed?" in run("submit")["__interrupt__"][0].value
    # a harmless button runs without asking
    script({"tool": "click", "role": "button", "name": "Save draft"}, done)
    assert "__interrupt__" not in run("save") and page().title() == "saved"

    # ask_human pauses; the answer reaches the LLM and earlier actions are not repeated
    script({"tool": "type", "role": "textbox", "name": "To", "text": "a"}, {"tool": "ask_human", "text": "Which John?"}, done)
    r = run("email John", "John Smith")
    assert len(r["history"]) == 2 and "user: John Smith" in r["history"][1] and "John Smith" in seen[-1], r
    # a question put in name instead of text still reaches the user
    script({"tool": "ask_human", "name": "Please log in"}, done)
    assert run("log in")["__interrupt__"][0].value == "Please log in"

    # an open dialog is shown first, even when it is outside the cut; plain text can be clicked to reveal a field
    bw.current_page().set_content("<p>" + "x" * (bw.MAX_PAGE_CHARS + 100) + """</p>
        <div role=dialog aria-label="New Message"><span onclick="this.nextElementSibling.hidden=false">Recipients</span>
        <input aria-label="To recipients" hidden></div>""")
    assert "New Message" in bw.page_tree(bw.current_page()) and "Recipients" in bw.page_tree(bw.current_page())
    script({"tool": "click", "role": "text", "name": "Recipients"},
           {"tool": "type", "role": "textbox", "name": "To recipients", "text": "jane@acme.com"}, done)
    config = {"configurable": {"thread_id": "dialog"}, "recursion_limit": bw.RECURSION_LIMIT}
    r = app.invoke({"task": "address it"}, config)
    assert page().get_by_label("To recipients").input_value() == "jane@acme.com", r["history"]
    # the LLM never types a password, even when it tries
    script({"tool": "type", "role": "textbox", "name": "Password", "text": "hunter2"}, done)
    assert "refused: passwords" in run("log in")["history"][0]
    assert page().get_by_label("Password").input_value() == ""
    # file:// and other schemes are refused
    script({"tool": "goto", "url": "file:///C:/Windows/win.ini"}, done)
    assert "refused" in run("read win.ini")["history"][0]
    # a missing element is reported back to the LLM, not a crash
    script({"tool": "click", "role": "button", "name": "Nope"}, done)
    assert run("click nope")["history"][0].startswith("click 'Nope' -> error")
    # upload: the picker the button opens gets the file (no OS dialog), without a question of its own;
    # the Send confirmation then lists the attachment next to the typed text
    from guards import ALLOWED_ROOTS
    docs = next(r for r in ALLOWED_ROOTS if r.is_dir())
    attach_page = """<button onclick="document.getElementById('f').click()">Attach files</button>
        <input id=f type=file hidden><button onclick="document.title='sent'">Send</button>"""
    with tempfile.NamedTemporaryFile(dir=docs, suffix=".pdf", delete=False) as f:
        f.write(b"x")
    try:
        bw.current_page().set_content(attach_page)
        script({"tool": "upload", "role": "button", "name": "Attach files", "path": f.name},
               dict(send, summary="Send the resume"), done)
        q = app.invoke({"task": "attach"}, {"configurable": {"thread_id": "up"}})["__interrupt__"][0].value
        assert f"Attached {Path(f.name).name}: {bw.resolve(f.name)}" in q, q
        assert page().evaluate("document.getElementById('f').files[0].name") == Path(f.name).name
        assert page().title() != "sent"
        # outside the allowed folders: refused before any picker opens; a missing file: said so
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as outside:  # the temp folder, under AppData
            outside.write(b"x")
        for path, why in [(outside.name, "refused: outside allowed folders"), (str(docs / "nope.pdf"), "error: no such file")]:
            bw.current_page().set_content(attach_page)
            script({"tool": "upload", "role": "button", "name": "Attach files", "path": path}, done)
            r = app.invoke({"task": "attach"}, {"configurable": {"thread_id": f"up-{why}"}})
            assert why in r["history"][0], r["history"]
            assert page().evaluate("document.getElementById('f').files.length") == 0
        os.remove(outside.name)
    finally:
        os.remove(f.name)

    # a misnamed field is shown back to the LLM, which retries
    script({"tool": "goto", "link": "https://example.com"}, done)
    r = run("open example")
    assert "invalid reply" in r["history"][0] and r["output"] == "finished", r
    # a looping LLM is stopped
    script({"tool": "press", "text": "Tab"})
    assert run("loop")["output"] == f"gave up after {bw.MAX_ACTIONS} actions"

if __name__ == "__main__":
    test_all()
    print("all checks passed")

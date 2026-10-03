"""Offline checks for desktop_workflow: fake model, apps and windows; nothing is launched or closed.
Run: python tests/test_desktop_workflow.py"""
import itertools
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # core/, where workflows/ and guards.py live
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from workflows import desktop_workflow as dw

APPS = [("Word", "word-id"), ("WordPad", "wordpad-id"), ("Google Chrome", "Chrome"), ("Calculator", "calc-id"),
        ("Visual Studio Code", "vscode-id")]
open_windows = {}  # hwnd -> title
started, closed, focused = [], [], []


class FakeWin32:
    IsWindow = staticmethod(lambda h: h in open_windows)
    def PostMessage(self, h, msg, *_):
        closed.append(open_windows.pop(h))


dw.installed_apps = lambda: APPS
dw.windows = lambda: list(open_windows.items())
dw.win32gui = lambda: FakeWin32()
dw.focus = focused.append
dw.wait_for_new_window = lambda before, seconds=8: "New window"
dw.time.sleep = lambda s: None
os.startfile = started.append


def model(action, target=""):
    class Fake:
        def __init__(self, **kw): pass
        def with_structured_output(self, schema, method): return self
        def invoke(self, messages): return dw.Action(action=action, target=target)
    dw.ChatGroq = Fake


app = dw.graph.compile(checkpointer=InMemorySaver())
_ids = itertools.count()

def run(action, target="", *answers):
    model(action, target)
    config = {"configurable": {"thread_id": f"d{next(_ids)}"}}
    r = app.invoke({"task": f"{action} {target}"}, config)
    for a in answers:
        assert "__interrupt__" in r, r
        r = app.invoke(Command(resume=a), config)
    return r


def test_all():
    # open by exact name: the Start-menu launch id is used, never a path the model made up
    r = run("open", "calculator")
    assert started[-1] == "shell:AppsFolder\\calc-id" and r["output"] == "Opened Calculator (window: 'New window').", r
    # exact beats starts-with: "word" is Word, not WordPad
    run("open", "Word")
    assert started[-1] == "shell:AppsFolder\\word-id"
    # several equally good matches: asks, by number or name
    r = run("open", "goo")
    assert started[-1] == "shell:AppsFolder\\Chrome"  # only one starts with "goo": no question
    APPS.append(("Google Earth", "earth-id"))
    q = run("open", "google")["__interrupt__"][0].value
    assert "1. Google Chrome" in q and "2. Google Earth" in q, q
    assert run("open", "google", "2")["output"].startswith("Opened Google Earth") and started[-1].endswith("earth-id")
    assert run("open", "google", "earth")["output"].startswith("Opened Google Earth")
    n = len(started)
    assert run("open", "google", "firefox")["failed"] and len(started) == n  # no match: nothing opened
    # misspelt: closest name; unknown: says so and suggests
    run("open", "calculater")
    assert started[-1] == "shell:AppsFolder\\calc-id"
    r = run("open", "Photoshop")
    assert r["failed"] and r["output"].startswith("No installed app called 'Photoshop'."), r

    # files and folders open by path; programs and scripts by path are refused
    with tempfile.TemporaryDirectory() as d:
        doc, exe = Path(d, "report.pdf"), Path(d, "setup.exe")
        doc.write_text("x"); exe.write_text("x")
        assert run("open", str(doc))["output"].startswith("Opened") and started[-1] == str(doc)
        n = len(started)
        r = run("open", str(exe))
        assert r["failed"] and r["output"].startswith("refused: won't run programs") and len(started) == n, r

    # switch: by title words; no window means it says so
    open_windows.update({101: "Instagram - Google Chrome", 102: "main.py - Visual Studio Code", 103: "Calculator"})
    assert run("switch", "visual studio code")["output"] == "Switched to 'main.py - Visual Studio Code'." and focused == [102]
    assert run("switch", "excel")["output"] == "No open window matching 'excel'."

    # close asks first; "no" leaves it open, "yes" closes it
    q = run("close", "calculator")["__interrupt__"][0].value
    assert "About to: close the window 'Calculator'" in q, q
    r = run("close", "calculator", "no")
    assert r["failed"] and r["output"] == "declined by user" and 103 in open_windows and closed == []
    assert run("close", "calculator", "yes")["output"] == "Closed 'Calculator'." and closed == ["Calculator"]

    # two matching windows: choose, then confirm; the choice comes from the saved list,
    # so a window opening while you decide can't shift which one "2" means
    open_windows.update({104: "Docs - Google Chrome"})
    config = {"configurable": {"thread_id": "pick"}}
    model("close", "google chrome")
    q = app.invoke({"task": "close chrome"}, config)["__interrupt__"][0].value
    assert "1. Instagram - Google Chrome" in q and "2. Docs - Google Chrome" in q, q
    open_windows.update({100: "Mail - Google Chrome"})  # appears while the question is open
    q = app.invoke(Command(resume="2"), config)["__interrupt__"][0].value
    assert "'Docs - Google Chrome'" in q, q
    r = app.invoke(Command(resume="yes"), config)
    assert r["output"] == "Closed 'Docs - Google Chrome'." and 101 in open_windows and 100 in open_windows

    # a window that closed in the meantime is reported, not acted on
    model("switch", "mail")
    config = {"configurable": {"thread_id": "gone"}}
    open_windows.pop(100)
    assert app.invoke({"task": "switch to mail"}, config)["output"] == "No open window matching 'mail'."

    # list, and what isn't built yet says so
    out = run("list")["output"]
    assert out.startswith("Open windows:") and "Instagram - Google Chrome" in out, out
    assert run("other", "")["output"].startswith("not supported yet")


if __name__ == "__main__":
    test_all()
    print("all checks passed")

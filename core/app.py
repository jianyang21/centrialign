"""centrialign in the browser. Run from core/:  streamlit run app.py

Streamlit reruns this script on a fresh thread for every click, but the agent must stay on ONE thread:
Playwright's sync API and the Chrome connection belong to the thread that opened them, and COM (Windows
Search, window focus) is set up per thread. So one worker thread, created once and shared by every rerun
and tab, runs every task one at a time; the page only shows its state and hands it input."""
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
import streamlit as st

# core/ first: an unrelated "db" package in site-packages must never shadow core/db.py
if (CORE := str(Path(__file__).resolve().parent)) not in sys.path[:1]:
    sys.path.insert(0, CORE)
WORKING = "Working… the agent may use your windows and Chrome"


class Agent:
    def __init__(self, failed=None):  # failed: an agent that couldn't start; its chat and queued requests carry over
        self.chat = failed.chat if failed else []  # (role, text): "user", "assistant", or "log" (progress lines)
        self.question = None    # what a paused task is asking, while it waits for an answer
        self.status = "Starting…"
        self.tasks = []         # unfinished tasks from the database, for the sidebar
        self.requests = failed.requests if failed else queue.Queue()  # page -> worker: ("run", text), ("resume", task), ("drop", task)
        self.answers = queue.Queue()   # page -> worker: the answer to self.question
        self.thread = threading.Thread(target=self.work, daemon=True)
        sys.stdout = Progress(self, sys.stdout)
        self.thread.start()

    def ask(self, question):
        """drive() calls this instead of input(): the page shows the question, the worker waits for the answer."""
        self.question = question
        return self.answers.get()

    def answer(self, text):
        if self.question:  # cleared first, so a double click can't answer twice
            self.question = None
            self.chat.append(("user", text))
            self.answers.put(text)

    def start(self):
        import pythoncom
        pythoncom.CoInitialize()
        from dotenv import load_dotenv
        load_dotenv()  # before db: it reads DATABASE_URL when imported
        import db
        db.start_database(say=lambda s: setattr(self, "status", s))  # Docker Desktop + the Postgres container
        while True:  # just in case it's up but not accepting connections yet
            try:
                import orchestrator  # opens the database and compiles the graph
                return orchestrator, db
            except db.TRANSIENT as e:
                self.status = f"Waiting for the database ({type(e).__name__}): run `docker compose up -d` in core"
                time.sleep(5)

    def work(self):
        try:
            o, db = self.start()
        except Exception as e:  # a real bug or a missing package: retrying won't fix it, so show it
            self.status = f"Couldn't start: {type(e).__name__}: {e}"
            raise
        while True:
            try:
                self.tasks = db.unfinished()
            except db.TRANSIENT:
                pass  # keep the last list; the next task says if the database is down
            self.status = "Ready"
            kind, arg = self.requests.get()
            self.status = WORKING
            try:
                if kind == "run":
                    tid = str(uuid.uuid4())  # unique per task: saved checkpoints must never mix two tasks
                    reply = (o.drive(tid, {"user_request": arg}, self.ask) if db.task_started(tid, arg) else
                             "The database is unreachable: start it with `docker compose up -d` in core.")
                elif kind == "resume":
                    reply = o.resume(arg, self.ask)
                else:
                    db.task_update(arg["thread_id"], "cancelled")
                    reply = f"Dropped: {arg['request']}"
            except Exception as e:  # the worker must outlive any one task
                reply = f"Error: {type(e).__name__}: {str(e)[:300]}"
            self.question = None
            self.chat.append(("assistant", reply))


class Progress:
    """The workflows' print() lines ("[browser] click 'Compose' -> ok") from the worker go into the chat too."""
    def __init__(self, agent, real):
        self.agent, self.real = agent, real

    def write(self, s):
        if threading.current_thread() is self.agent.thread and s.strip():
            self.agent.chat.append(("log", s.strip()))
        return self.real.write(s)

    def flush(self):
        self.real.flush()


@st.cache_resource  # one agent per server process, whatever the number of reruns and tabs
def get_agent(_failed=None):  # leading underscore: not part of the cache key
    return Agent(_failed)


st.set_page_config(page_title="centrialign", page_icon="🤖")
agent = get_agent()
st.session_state.shown = (agent.question, agent.status)

with st.sidebar:
    st.subheader("Unfinished tasks")
    busy = agent.status != "Ready"
    for t in agent.tasks:
        st.write(f"**{t['request']}**  \n{t['status']}" + (f": {t['error']}" if t["error"] else ""))
        resume, drop = st.columns(2)
        if resume.button("Resume", key=f"r{t['thread_id']}", disabled=busy):
            agent.chat.append(("user", f"Resume: {t['request']}"))
            agent.requests.put(("resume", t))
            st.rerun()
        if drop.button("Drop", key=f"d{t['thread_id']}", disabled=busy):
            agent.requests.put(("drop", t))
            st.rerun()
    if not agent.tasks:
        st.caption("None.")


@st.fragment(run_every=1)  # live: progress lines appear while a task runs
def conversation():
    for role, text in agent.chat:
        if role == "log":
            st.text(text)
        else:
            st.chat_message(role).markdown(text.replace("\n", "  \n"))
    if agent.question:
        with st.chat_message("assistant"):
            st.text(agent.question)
            for col, word in zip(st.columns(4), ("yes", "no", "done", "cancel")):
                col.button(word.capitalize(), key=f"a-{word}", on_click=agent.answer, args=(word,),
                           use_container_width=True)
    else:
        st.caption(agent.status)
    if agent.status.startswith("Couldn't start") and st.button("Retry"):  # after fixing it: no server restart
        get_agent.clear()
        get_agent(agent)
        st.rerun()
    if (agent.question, agent.status) != st.session_state.shown:
        st.rerun()  # whole page: the input box and the sidebar change with these


conversation()
if agent.question:
    text = st.chat_input("Answer the question above (yes / no / what to change / a number)…")
    if text:
        agent.answer(text)
        st.rerun()
else:
    text = st.chat_input("Ask the agent to do something…")
    if text:
        agent.chat.append(("user", text + ("" if agent.status == "Ready" else "  _(queued)_")))
        agent.requests.put(("run", text))
        st.rerun()

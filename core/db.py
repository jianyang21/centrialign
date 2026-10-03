"""Postgres: company context, task records and LangGraph checkpoints. Start it with `docker compose up -d`.
Without DATABASE_URL everything still runs, but in memory: company context is empty and nothing can be resumed."""
import atexit
import functools
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse
import groq
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout
from langgraph.checkpoint.memory import InMemorySaver

URL = os.getenv("DATABASE_URL", "")
# Failures worth pausing for and resuming later, instead of failing the task: the network, the model
# provider being down or rate-limited, the database being unreachable.
TRANSIENT = (groq.APIConnectionError, groq.APITimeoutError, groq.RateLimitError, groq.InternalServerError,
             psycopg.OperationalError, PoolTimeout)


DOCKER_DESKTOP = Path(os.getenv("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe"


def start_database(say=print, timeout=180):
    """Bring up the local Postgres: start Docker Desktop if its engine is down, then `docker compose up -d --wait`
    (returns once the container's healthcheck passes; nothing to do if it's already up). Skipped without
    DATABASE_URL or when it points at another machine. Raises RuntimeError with what to fix."""
    if urlparse(URL).hostname not in ("127.0.0.1", "localhost"):
        return
    def docker(*args, wait=60):
        try:
            return subprocess.run(["docker", *args], cwd=Path(__file__).resolve().parent, capture_output=True,
                                  text=True, timeout=wait, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except FileNotFoundError:
            raise RuntimeError("Docker isn't installed (no `docker` command). Install Docker Desktop.") from None
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(args, 1, "", f"docker {args[0]} timed out")
    deadline = time.monotonic() + timeout
    if docker("info").returncode != 0:  # the engine is down: Docker Desktop isn't running
        if not DOCKER_DESKTOP.exists():
            raise RuntimeError(f"Docker's engine isn't running and Docker Desktop wasn't found at {DOCKER_DESKTOP}.")
        say("Starting Docker Desktop (the first start can take a minute)…")
        subprocess.Popen([str(DOCKER_DESKTOP)])
        while docker("info").returncode != 0:
            if time.monotonic() > deadline:
                raise RuntimeError(f"Docker Desktop didn't start within {timeout}s. Open it yourself, then retry.")
            time.sleep(2)
    say("Starting the database…")
    up = docker("compose", "up", "-d", "--wait", wait=max(30, deadline - time.monotonic()))
    if up.returncode != 0:
        raise RuntimeError(f"`docker compose up` failed: {(up.stderr or up.stdout).strip()[-500:]}")


@functools.cache
def pool():
    # a pool, not one connection: it replaces connections that die when the network or the container drops.
    # Use 127.0.0.1, not localhost, in DATABASE_URL: localhost tries IPv6 first and hangs the pool's workers.
    p = ConnectionPool(URL, open=False, min_size=1, max_size=4, timeout=10, check=ConnectionPool.check_connection,
                       kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    p.open(wait=True, timeout=10)
    atexit.register(p.close)
    return p


def checkpointer():
    if not URL:
        return InMemorySaver()
    from langgraph.checkpoint.postgres import PostgresSaver
    saver = PostgresSaver(pool())
    saver.setup()  # creates or migrates LangGraph's checkpoint tables; safe to run every start
    return saver


def execute(sql, *params):
    if not URL:
        return []
    with pool().connection() as conn:
        cur = conn.execute(sql, params)
        return cur.fetchall() if cur.description else []


def record(sql, *params):
    """Task bookkeeping. A database outage must not crash the task it tracks (it is often the very error being
    recorded): a missed update leaves the row in its last unfinished state, so --resume still finds it.
    False if it wasn't written."""
    try:
        execute(sql, *params)
        return True
    except TRANSIENT as e:
        print(f"[db] task record not saved: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return False


def task_started(thread_id, request):
    return record("INSERT INTO tasks (thread_id, request, status) VALUES (%s, %s, 'running')", thread_id, request)


def task_update(thread_id, status, question=None, error=None, results=None):
    record("UPDATE tasks SET status = %s, question = %s, error = %s, results = %s, updated_at = now() "
           "WHERE thread_id = %s", status, question, error, json.dumps(results) if results is not None else None,
           thread_id)


def unfinished():
    return execute("SELECT thread_id, request, status, question, error FROM tasks "
                   "WHERE status IN ('running', 'waiting', 'interrupted') ORDER BY created_at")


def company_context():
    """The company tables as text for the Internal workflow's prompt. "" without a database."""
    people = execute("SELECT p.name, p.role, p.team, p.email, p.is_user, m.name AS manager "
                     "FROM people p LEFT JOIN people m ON m.id = p.manager_id ORDER BY p.id")
    orgs = execute("SELECT o.name, o.kind, o.contact_name, o.contact_email, o.portal_url, o.notes, p.name AS owner "
                   "FROM organizations o LEFT JOIN people p ON p.id = o.owner_id ORDER BY o.name")
    policies = execute("SELECT topic, body FROM policies ORDER BY topic")
    if not (people or orgs or policies):
        return ""
    user = next((p for p in people if p["is_user"]), None)
    lines = [f"The user (\"I\", \"me\"): {user['name']}, {user['role']}, {user['team']} team, {user['email']}." if user else ""]
    lines += ["", "People (name | role | team | reports to | email):"]
    lines += [f"- {p['name']} | {p['role']} | {p['team']} | {p['manager'] or '-'} | {p['email']}" for p in people]
    lines += ["", "Clients and vendors:"]
    lines += [f"- {o['name']} ({o['kind']})" + (f": contact {o['contact_name']} <{o['contact_email']}>" if o["contact_name"] else "")
              + (f"; looked after by {o['owner']}" if o["owner"] else "") + (f"; portal {o['portal_url']}" if o["portal_url"] else "")
              + (f". {o['notes']}" if o["notes"] else "") for o in orgs]
    lines += ["", "Policies and facts:"] + [f"- {p['topic']}: {p['body']}" for p in policies]
    return "\n".join(lines).strip()

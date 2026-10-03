"""Postgres: company context, task records and LangGraph checkpoints. Start it with `docker compose up -d`.
Without DATABASE_URL everything still runs, but in memory: company context is empty and nothing can be resumed."""
import atexit
import functools
import json
import os
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


def task_started(thread_id, request):
    execute("INSERT INTO tasks (thread_id, request, status) VALUES (%s, %s, 'running')", thread_id, request)


def task_update(thread_id, status, question=None, error=None, results=None):
    execute("UPDATE tasks SET status = %s, question = %s, error = %s, results = %s, updated_at = now() "
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

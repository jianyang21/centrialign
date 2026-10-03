# centrialign

A desktop agent for company work on Windows. You type a request in plain English. It plans the steps, then uses the browser, your files, your apps and the company database to get them done. It asks you before anything that cannot be undone, checks its own work, and keeps a screenshot as proof.

## How it works

1. **Plan.** An LLM splits the request into ordered steps and gives each one to a workflow.
2. **Act.** Each workflow uses the LLM only to decide. Plain code does the work and enforces the safety rules.
3. **Ask.** Sending, paying, deleting or closing anything waits for your yes.
4. **Verify.** When the browser agent says it is done, a second LLM call checks the final page. If the proof is missing, the agent gets one more try. Every browser task saves a screenshot.
5. **Remember.** Every step is saved in Postgres, so a task stopped by a network error can be resumed.

| Workflow | What it does |
|---|---|
| Browser | Works on any website through your Chrome, with no site specific code |
| Files | Finds files by name |
| Desktop | Opens, switches to, lists and closes apps |
| Internal | Answers company questions from the database |

## Setup

You need Windows, Python 3.13, Google Chrome, Docker Desktop and a Groq API key.

```
cd core
pip install -r requirements.txt
playwright install chromium
docker compose up -d
```

Create `core/.env`:

```
GROQ_API_KEY=your_key
DATABASE_URL=postgresql://centrialign:centrialign@127.0.0.1:5432/centrialign
```

## Run

From the core folder:

```
streamlit run app.py
```

Or in the terminal, with a request or with the built in demo list:

```
python orchestrator.py "Who's my boss?"
python orchestrator.py --resume
```

Tests run offline with a fake LLM:

```
python tests/test_browser_workflow.py
```

## Models

All models are served by Groq. `openai/gpt-oss-120b` runs the browser agent and its verifier. `openai/gpt-oss-20b` runs the planner and the Files, Desktop and Internal workflows.

## Assumptions

* The company data is simulated. Northwind Freight lives in `schema.sql` and its emails use the `.example` domain.
* Demos use a test Gmail account, never real company credentials.
* We did not build mock company portals. The browser agent has no site specific code, so a real portal such as Odoo only needs its address and a login.

## Limitations

* Files can only search. Copy, move and rename are not built yet.
* Desktop can only open, switch to, list and close apps.
* If a step fails, the plan stops. It does not plan again yet.
* Only browser steps are verified.

## Next steps

* Plan again when a step fails, with a limit on attempts.
* Add a table of company procedures for the planner to follow.
* Add file copy, move and rename behind the same approval check.
* Add an evaluation suite with DeepEval.

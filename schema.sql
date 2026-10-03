-- The centrialign database: company context, task records, and SIMULATED sample company data.
-- Docker loads this once, when the container starts on an empty volume (docker compose up -d).
-- LangGraph creates and migrates its own checkpoint tables (checkpoints, checkpoint_blobs, checkpoint_writes,
-- checkpoint_migrations) when the orchestrator starts, so they are not defined here.

-- ---------------------------------------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------------------------------------

-- Company context for the Internal workflow: who works here, who reports to whom, clients, vendors, policies.
CREATE TABLE people (
    id         serial PRIMARY KEY,
    name       text NOT NULL UNIQUE,
    role       text NOT NULL,
    team       text,
    email      text UNIQUE,
    manager_id int REFERENCES people (id),
    is_user    boolean NOT NULL DEFAULT false  -- the person using the assistant: "me", "my boss"
);
CREATE UNIQUE INDEX one_user ON people (is_user) WHERE is_user;

CREATE TABLE organizations (
    id            serial PRIMARY KEY,
    name          text NOT NULL UNIQUE,
    kind          text NOT NULL CHECK (kind IN ('client', 'vendor', 'partner')),
    contact_name  text,
    contact_email text,
    owner_id      int REFERENCES people (id),  -- who looks after the relationship
    portal_url    text,
    notes         text
);

CREATE TABLE policies (
    topic text PRIMARY KEY,
    body  text NOT NULL
);

-- One row per request, so unfinished ones can be listed and resumed. thread_id is the checkpoint thread.
CREATE TABLE tasks (
    thread_id  text PRIMARY KEY,
    request    text NOT NULL,
    status     text NOT NULL CHECK (status IN ('running', 'waiting', 'interrupted', 'done', 'cancelled')),
    question   text,   -- what it is waiting for you to answer
    error      text,   -- why it was interrupted
    results    jsonb,  -- each step's output, once done
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX unfinished_tasks ON tasks (created_at) WHERE status IN ('running', 'waiting', 'interrupted');

-- ---------------------------------------------------------------------------------------------------------
-- SIMULATED sample company (Northwind Freight). Replace with the real company's data.
-- Emails use the reserved .example domain so nothing here can reach a real inbox.
-- ---------------------------------------------------------------------------------------------------------

INSERT INTO people (name, role, team, email, manager_id, is_user) VALUES
    ('Priya Raman',  'CEO',                 'Leadership', 'priya.raman@northwind.example',  NULL, false),
    ('Karthik Iyer', 'CTO',                 'Leadership', 'karthik.iyer@northwind.example', 1,    false),
    ('Meera Nair',   'Engineering Manager', 'Platform',   'meera.nair@northwind.example',   2,    false),
    ('Arjun Mehta',  'Software Engineer',   'Platform',   'arjun.mehta@northwind.example',  3,    true),
    ('Divya Shah',   'Software Engineer',   'Platform',   'divya.shah@northwind.example',   3,    false),
    ('Rahul Verma',  'Head of Operations',  'Operations', 'rahul.verma@northwind.example',  1,    false),
    ('Sneha Pillai', 'HR Manager',          'People',     'sneha.pillai@northwind.example', 1,    false),
    ('Vikram Rao',   'Finance Manager',     'Finance',    'vikram.rao@northwind.example',   1,    false);

INSERT INTO organizations (name, kind, contact_name, contact_email, owner_id, portal_url, notes) VALUES
    ('Acme Retail',        'client', 'Jane Doe', 'jane.doe@acme.example',  6, NULL, NULL),
    ('Globex Stores',      'client', 'Sam Lee',  'sam.lee@globex.example', 6, NULL, NULL),
    ('Swiftline Supplies', 'vendor', NULL, NULL, 8, 'https://vendors.swiftline.example',
     'Packaging vendor. Invoices go to Vikram Rao.');

INSERT INTO policies (topic, body) VALUES
    ('Company',           'Northwind Freight Pvt Ltd, a logistics company. Head office: Chennai.'),
    ('Office hours',      '9:30 to 18:30 IST, Monday to Friday.'),
    ('Leave',             '24 days a year. Apply in the HR portal; your manager approves it.'),
    ('Reimbursements',    'Send receipts to Vikram Rao (Finance) within 30 days of the expense.'),
    ('IT help',           'it-help@northwind.example'),
    ('Working from home', 'Up to 2 days a week, agreed with your manager.');

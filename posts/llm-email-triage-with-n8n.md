---
title: An LLM email triage workflow in n8n
date: 2026-08-24
summary: A self-hosted n8n workflow that classifies incoming email with an LLM, routes it by category, drafts replies for human approval and logs every decision
tags: Automation, Agents
---

Email triage suits an LLM well. Most messages fall into a few categories, the cost of a wrong label is small if a human still approves the reply, and the time saved adds up every day. It is also a task where letting a model act unsupervised is a bad idea, so the design matters more than the prompt.

This post builds the workflow in n8n: trigger on new mail, classify and extract fields into JSON, route by category, draft replies for a human to approve, log everything, and handle failures. It ends with self-hosting n8n in Docker.

## The shape of the workflow

```text
Gmail Trigger
  -> Basic LLM Chain (+ chat model, + Structured Output Parser)
  -> Switch (by category)
       billing  -> Gmail: create draft  -> Google Sheets: append row
       support  -> Gmail: create draft  -> Google Sheets: append row
       spam     -> Gmail: add label     -> Google Sheets: append row
       fallback -> Gmail: add label "needs-human" -> Google Sheets: append row
Error workflow: Error Trigger -> notify me
```

The rule I hold to: the workflow never sends an email. It creates drafts and applies labels, and a person presses send.

## Step 1: trigger on new email

The **Gmail Trigger** node has one event, *Message Received*, and works by polling on a schedule you set. Settings worth knowing:

- **Simplify** is on by default and returns IDs, labels and headers such as From, To and Subject. Turn it off if you need the full message body for classification.
- **Max Emails per Poll** defaults to 10, up to a maximum of 50. Extra unread mail waits for the next poll.
- **Filters**: Label Names or IDs, a **Search** box that takes Gmail search operators (for example `-category:promotions`), Read Status (unread only by default) and Sender.

If you are not on Gmail, the core **Email Trigger (IMAP)** node fills the same role.

## Step 2: classify and extract with an LLM

n8n has several AI root nodes that could do this:

- **Text Classifier** takes a list of categories with descriptions and gives each one its own output branch. Its *When No Clear Match* option can send unmatched items to an extra "Other" branch instead of dropping them. That is useful if all you need is routing.
- **Information Extractor** pulls structured fields out of text.
- **Basic LLM Chain** is a single prompt plus a model. Turn on **Require Specific Output Format** and attach an output parser.
- **AI Agent** adds tools and multi-step reasoning, which triage does not need.

I use the Basic LLM Chain because I want the category *and* the extracted fields in one call. Attach a chat model sub-node (for example OpenAI Chat Model, or another provider's chat model) and a **Structured Output Parser**.

Set the chain's prompt to *Define below* and build it from the trigger's fields. The field names depend on whether Simplify is on, so check them in the trigger's output panel first.

```text
You triage email for a small software business.
Classify the email and extract fields. Use "other" if unsure.

From: {{ $json.From }}
Subject: {{ $json.Subject }}
Body:
{{ $json.text ?? $json.snippet }}
```

The Structured Output Parser offers two schema types. *Generate From JSON Example* infers a schema from a sample object, ignores the sample values and makes every field mandatory. *Define using JSON Schema* lets you write the schema by hand. Note that `$ref` is not supported. I prefer the explicit schema because it lets me use enums:

```json
{
  "type": "object",
  "properties": {
    "category": { "type": "string", "enum": ["billing", "support", "sales", "spam", "other"] },
    "urgency": { "type": "string", "enum": ["low", "normal", "high"] },
    "summary": { "type": "string", "description": "One sentence" },
    "customer_name": { "type": ["string", "null"] },
    "order_id": { "type": ["string", "null"] },
    "needs_reply": { "type": "boolean" },
    "draft_reply": { "type": ["string", "null"], "description": "Polite reply, no promises about refunds or dates" }
  },
  "required": ["category", "urgency", "summary", "needs_reply"]
}
```

Keep it flat and small. Every extra field is another chance for the model to get something wrong, and nested objects make the downstream expressions harder to read.

If the model sometimes returns output the parser rejects, the Basic LLM Chain also supports the **Auto-fixing Output Parser**. It wraps the structured parser and sends the parsing error back to a model to correct. The Text Classifier has the same idea built in as *Enable Auto-Fixing*. Auto-fixing costs an extra model call, so I turn it on only when I can see parse failures in the execution log.

## Step 3: route with a Switch node

Add a **Switch** node in Rules mode with one rule per category, comparing the string `{{ $json.output.category }}` to `billing`, `support` and so on. (With an output parser attached, the parsed object normally appears under `output`. Confirm the exact path in the chain's output panel before wiring up expressions.) Turn on the fallback output so anything unexpected goes to a "needs-human" branch rather than vanishing.

An **If** node works when there are only two paths, for example `needs_reply` true or false. For more categories, Switch keeps the canvas readable.

## Step 4: draft, do not send

On each branch that needs a reply, use the **Gmail** node with the **Draft → Create** operation. Set the message to `{{ $json.output.draft_reply }}` and the subject to `Re: ` plus the original subject. A **Message → Add Label** step (for example `triage/billing`) makes the drafts easy to find. The Gmail node also has Message → Reply and Send, but I leave those out of the workflow on purpose.

Why drafts rather than auto-send:

- A misclassified email gets a wrong but harmless draft, not a wrong reply to a customer.
- The draft prompt can be cautious ("no promises about refunds or dates") because a human adds the specifics.
- You collect a record of model drafts against what was actually sent, which is the evidence you would need before automating any category further.

For spam and newsletters, apply a label and stop. No draft needed.

## Step 5: log every decision

End every branch with a **Google Sheets** node that appends a row (or a Postgres or other database node if you prefer). I log:

```text
timestamp | message_id | from | subject | category | urgency | summary | draft_created | workflow_execution_id
```

`{{ $execution.id }}` gives you the execution ID, which links a sheet row back to the full run in n8n. Logging every branch, including spam, lets you audit the classifier later. You can sample rows weekly and see where the labels were wrong.

## Step 6: errors and retries

Two layers:

1. **Per-node retries.** In a node's settings you can turn on *Retry On Fail* and set the number of tries and the wait between them. I enable this on the chat model and Gmail nodes, which fail occasionally on rate limits or timeouts. The node's *On Error* setting controls what happens once retries run out: stop the workflow, continue, or continue through a separate error output that you can route to the "needs-human" label.
2. **An error workflow.** Create a separate workflow that starts with the **Error Trigger** node, and select it under the main workflow's **Settings → Error workflow**. The Error Trigger receives the execution ID and URL, the error message and stack, the last node executed, and the workflow name and ID. Send that to yourself by email or chat. One error workflow can serve all your workflows. If you want a run to fail on purpose, for example when the LLM returns `other` for something marked high urgency, a **Stop and Error** node forces the failure and the error workflow picks it up.

## Credentials

n8n stores credentials encrypted in its database and reuses them across nodes. The Gmail nodes use a Google credential (OAuth2). You create it once in n8n's Credentials section and select it in each node. Practical habits:

- Use a dedicated Google Cloud OAuth client for n8n, with only the Gmail and Sheets scopes you need.
- Never paste API keys into prompts, Code nodes or expressions. Use the credential fields.
- On a self-hosted instance, set `N8N_ENCRYPTION_KEY` yourself and back it up. If you lose it, the stored credentials cannot be decrypted.

## Self-hosting n8n with Docker

n8n's Docker guide creates a named volume for its data (the SQLite database and encryption key live in `/home/node/.n8n`) and runs the `n8nio/n8n` image on port 5678:

```bash
docker volume create n8n_data

docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -e GENERIC_TIMEZONE="Europe/London" \
  -e TZ="Europe/London" \
  -e N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true \
  -v n8n_data:/home/node/.n8n \
  n8nio/n8n
```

`GENERIC_TIMEZONE` sets the time zone for n8n's scheduling, and `TZ` sets the container's system clock. The guide also notes that `N8N_RUNNERS_ENABLED` is deprecated from n8n 2.0 and no longer needs setting.

For anything long-running, n8n now recommends Docker Compose. Its official compose file includes several supporting services, so use that for production. A minimal compose file of my own, with Postgres for the database, looks like this:

```yaml
services:
  postgres:
    image: postgres:18
    restart: always
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: n8n
      PGDATA: /var/lib/postgresql/data
    volumes:
      - db-storage:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -h localhost -U ${POSTGRES_USER} -d n8n"]
      interval: 10s
      retries: 5

  n8n:
    image: n8nio/n8n:${N8N_VERSION}
    restart: always
    ports:
      - "5678:5678"
    environment:
      DB_TYPE: postgresdb
      DB_POSTGRESDB_HOST: postgres
      DB_POSTGRESDB_PORT: "5432"
      DB_POSTGRESDB_DATABASE: n8n
      DB_POSTGRESDB_USER: ${POSTGRES_USER}
      DB_POSTGRESDB_PASSWORD: ${POSTGRES_PASSWORD}
      N8N_ENCRYPTION_KEY: ${N8N_ENCRYPTION_KEY}
      GENERIC_TIMEZONE: Europe/London
      TZ: Europe/London
    volumes:
      - n8n-data:/home/node/.n8n
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  db-storage:
  n8n-data:
```

Put the variables in a `.env` file next to it (generate the key with `openssl rand -base64 32`), pin `N8N_VERSION` rather than using `latest`, and update with `docker compose pull`, then `docker compose down` and `docker compose up -d`. Before exposing the instance beyond localhost, put it behind a reverse proxy with TLS. The Gmail OAuth callback needs a stable HTTPS URL anyway.

## What I would change before trusting it more

Start with drafts for everything. Watch the log for a few weeks. Only consider auto-sending for a category where the drafts consistently needed no edits, and even then keep the label and the log so you can see what went out.

## Sources

- [n8n: Gmail Trigger node](https://docs.n8n.io/integrations/builtin/trigger-nodes/n8n-nodes-base.gmailtrigger/)
- [n8n: Gmail node](https://docs.n8n.io/integrations/builtin/app-nodes/n8n-nodes-base.gmail)
- [n8n: Basic LLM Chain node](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.chainllm)
- [n8n: Structured Output Parser node](https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.outputparserstructured/)
- [n8n: Text Classifier node](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.text-classifier)
- [n8n: Information Extractor node](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.information-extractor)
- [n8n: Error handling](https://docs.n8n.io/flow-logic/error-handling/)
- [n8n: Install with Docker](https://docs.n8n.io/deploy/host-n8n/install-options/install-with-docker)
- [n8n: Install using Docker Compose](https://docs.n8n.io/deploy/host-n8n/install-options/install-using-docker-compose)

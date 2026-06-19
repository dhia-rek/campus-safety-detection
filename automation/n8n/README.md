# n8n automation

This folder contains a local n8n runtime setup and the contract between this project and a no-code workflow.

## What the app sends

The webhook action posts JSON like this when the decision severity matches `ESCALATE_SEVERITIES`:

```json
{
  "event": "campus_safety_incident",
  "scene": "05_0019",
  "frame": 123,
  "decision": {
    "is_incident": true,
    "severity": "high",
    "modalities": ["vision", "sound"],
    "reasoning": "...",
    "action": "...",
    "report": "...",
    "source": "llm"
  },
  "observations": [
    {
      "agent": "vision",
      "modality": "Visual (CLIP+YOLO)",
      "score": 2.14,
      "triggered": true,
      "detail": "..."
    }
  ]
}
```

## Run n8n locally

Docker is the most repeatable option, but it is optional.

```bash
cd automation/n8n
docker compose up -d
```

Then open `http://localhost:5678`.

If you prefer not to use Docker, install Node.js 18+ and run:

```bash
npx n8n start
```

## Suggested workflow

1. Add a **Webhook** node with the path `campus-incident` and method `POST`.
2. Add an **IF** node to route only `high` or `medium` severity events.
3. Add one or more actions, such as:
   - Telegram notification
   - e-mail
   - Slack or Discord
   - Google Sheets / database log
   - HTTP call to another incident system

## Connect the app

Set the webhook URL in your `.env` file:

```bash
INCIDENT_WEBHOOK_URL=http://localhost:5678/webhook/campus-incident
ESCALATE_SEVERITIES=high
```

Restart the FastAPI backend after changing the environment.
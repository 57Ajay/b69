# Cab Booking Slack Bot

AI-powered cab booking assistant integrated with Slack using LangGraph.

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set environment variables:
```bash
export SLACK_BOT_TOKEN="xoxb-your-slack-token"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export PORT=8000
```

3. Run the bot:
```bash
uv run python main.py
```

## Slack Setup

1. Create a Slack app at https://api.slack.com/apps
2. Add bot token scopes: `chat:write`, `im:history`, `im:read`
3. Install app to workspace
4. Set Event URL: `https://your-url/slack/events`
5. Subscribe to events: `message.channels`, `message.im`
6. Create slash command `/cab` pointing to `https://your-url/slack/commands`

## Usage

- Direct message: "I need drivers in Jaipur"
- Slash command: `/cab show drivers in Delhi`
- Filters: "Show me drivers who speak Hindi"
- Reset: Type "reset" to start over

## Docker

```bash
docker build -t cab-bot .
docker run -p 8000:8000 --env-file .env cab-bot
```

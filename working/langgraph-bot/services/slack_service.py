# services/slack_service.py
"""Slack integration service"""

import logging
from typing import Dict, Any, Set
from datetime import datetime

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

import config

logger = logging.getLogger(__name__)


class SlackService:
    def __init__(self):
        self.client = WebClient(token=config.SLACK_BOT_TOKEN)
        self.processed_messages: Set[str] = set()

    def is_duplicate_message(self, event: dict) -> bool:
        """Check if message was already processed"""
        user_id = event.get("user")
        text = event.get("text", "").strip()
        timestamp = event.get("ts", "")

        # Create unique identifier
        identifier = f"{user_id}:{text}:{timestamp}"

        if identifier in self.processed_messages:
            return True

        self.processed_messages.add(identifier)

        # Keep only last 200 messages
        if len(self.processed_messages) > 200:
            self.processed_messages = set(list(self.processed_messages)[-100:])

        return False

    def send_message(self, channel: str, text: str) -> bool:
        """Send message to Slack channel"""
        try:
            self.client.chat_postMessage(channel=channel, text=text)
            return True
        except SlackApiError as e:
            logger.error(f"Slack error: {e}")
            # Try DM if channel fails
            try:
                dm_response = self.client.conversations_open(users=[channel])
                if dm_response["ok"]:
                    dm_channel = dm_response["channel"]["id"]
                    self.client.chat_postMessage(channel=dm_channel, text=text)
                    return True
            except:
                pass

            return False

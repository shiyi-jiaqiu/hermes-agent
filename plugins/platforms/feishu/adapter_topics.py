"""Distinguish a topic title from replies without dropping ordinary group messages."""
from typing import Any


def is_topic_root(message: Any, *, chat_mode: str | None, event_chat_type: str) -> bool:
    """A topic-group root has no ancestor distinct from itself.

    `thread_id` identifies both roots and replies, so its presence alone cannot
    distinguish them. Require the chat API's topic mode: ordinary group messages
    also lack root/parent IDs and must continue to reach the agent.
    """
    if event_chat_type == "p2p" or chat_mode != "topic":
        return False
    message_id = str(getattr(message, "message_id", "") or "")
    return not any(
        str(getattr(message, field, "") or "") not in ("", message_id)
        for field in ("root_id", "parent_id", "upper_message_id")
    )

"""
State Manager for the cab booking agent.
Handles state persistence, session management, and state recovery.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError
import pickle

from src.models.agent_state_model import AgentState

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages conversation state with support for:
    1. Redis-based persistence for production
    2. In-memory fallback for development
    3. State compression and optimization
    4. Session timeout handling
    5. State recovery mechanisms
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        use_redis: bool = True,
        session_ttl: int = 1800,  # 30 minutes
        state_version: str = "1.0",
    ):
        """
        Initialize the state manager.

        Args:
            redis_host: Redis host address
            redis_port: Redis port
            redis_db: Redis database number
            use_redis: Whether to use Redis (False uses in-memory)
            session_ttl: Session timeout in seconds
            state_version: State schema version for migrations
        """
        self.use_redis = use_redis
        self.session_ttl = session_ttl
        self.state_version = state_version
        self.redis_client = None
        self.memory_store = {}  # Fallback in-memory store

        if use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False,  # We'll handle encoding
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis for state management")
            except RedisError as e:
                logger.warning(
                    f"Redis connection failed: {e}. Using in-memory storage."
                )
                self.use_redis = False
                self.redis_client = None

    async def get_state(self, session_id: str) -> Optional[AgentState]:
        """
        Retrieve state for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Agent state or None if not found
        """
        try:
            key = self._get_state_key(session_id)

            if self.use_redis and self.redis_client:
                # Get from Redis
                state_data = self.redis_client.get(key)
                if state_data:
                    # Extend TTL on access
                    self.redis_client.expire(key, self.session_ttl)
                    return self._deserialize_state(state_data)
            else:
                # Get from memory
                if key in self.memory_store:
                    state_entry = self.memory_store[key]
                    # Check if expired
                    if datetime.now() < state_entry["expires"]:
                        # Extend expiry
                        state_entry["expires"] = datetime.now() + timedelta(
                            seconds=self.session_ttl
                        )
                        return state_entry["state"]
                    else:
                        # Expired, remove it
                        del self.memory_store[key]

            return None

        except Exception as e:
            logger.error(f"Error retrieving state for {session_id}: {e}")
            return None

    async def save_state(self, session_id: str, state: AgentState) -> bool:
        """
        Save state for a session.

        Args:
            session_id: Unique session identifier
            state: Agent state to save

        Returns:
            Success status
        """
        try:
            key = self._get_state_key(session_id)

            # Add metadata
            state["last_updated"] = datetime.now().isoformat()
            state["state_version"] = self.state_version

            # Compress state for storage
            compressed_state = self._compress_state(state)

            if self.use_redis and self.redis_client:
                # Save to Redis
                serialized = self._serialize_state(compressed_state)
                self.redis_client.setex(key, self.session_ttl, serialized)
            else:
                # Save to memory
                self.memory_store[key] = {
                    "state": compressed_state,
                    "expires": datetime.now() + timedelta(seconds=self.session_ttl),
                }

            logger.debug(f"Saved state for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving state for {session_id}: {e}")
            return False

    async def delete_state(self, session_id: str) -> bool:
        """
        Delete state for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Success status
        """
        try:
            key = self._get_state_key(session_id)

            if self.use_redis and self.redis_client:
                self.redis_client.delete(key)
            else:
                if key in self.memory_store:
                    del self.memory_store[key]

            logger.debug(f"Deleted state for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting state for {session_id}: {e}")
            return False

    async def list_active_sessions(self) -> List[str]:
        """
        List all active sessions.

        Returns:
            List of active session IDs
        """
        try:
            if self.use_redis and self.redis_client:
                pattern = self._get_state_key("*")
                keys = self.redis_client.keys(pattern)
                # Extract session IDs from keys
                prefix = "cabbot:state:"
                return [key.decode().replace(prefix, "") for key in keys]
            else:
                # From memory store
                now = datetime.now()
                active = []
                for key, entry in list(self.memory_store.items()):
                    if now < entry["expires"]:
                        session_id = key.replace("cabbot:state:", "")
                        active.append(session_id)
                    else:
                        # Clean up expired
                        del self.memory_store[key]
                return active

        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []

    async def extend_session(
        self, session_id: str, additional_seconds: int = None
    ) -> bool:
        """
        Extend session timeout.

        Args:
            session_id: Session to extend
            additional_seconds: Additional time (default: session_ttl)

        Returns:
            Success status
        """
        try:
            key = self._get_state_key(session_id)
            ttl = additional_seconds or self.session_ttl

            if self.use_redis and self.redis_client:
                if self.redis_client.exists(key):
                    self.redis_client.expire(key, ttl)
                    return True
            else:
                if key in self.memory_store:
                    self.memory_store[key]["expires"] = datetime.now() + timedelta(
                        seconds=ttl
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Error extending session {session_id}: {e}")
            return False

    def _get_state_key(self, session_id: str) -> str:
        """Generate Redis key for session state."""
        return f"cabbot:state:{session_id}"

    def _compress_state(self, state: AgentState) -> AgentState:
        """
        Compress state by removing unnecessary data.

        Args:
            state: Original state

        Returns:
            Compressed state
        """
        compressed = state.copy()

        # Limit message history
        if "messages" in compressed and len(compressed["messages"]) > 20:
            # Keep first 5 and last 15 messages
            compressed["messages"] = (
                compressed["messages"][:5] + compressed["messages"][-15:]
            )

        # Limit driver history
        if "driver_history" in compressed and len(compressed["driver_history"]) > 10:
            compressed["driver_history"] = compressed["driver_history"][-10:]

        # Limit error history
        if "error_history" in compressed and len(compressed["error_history"]) > 5:
            compressed["error_history"] = compressed["error_history"][-5:]

        # Remove temporary data
        temp_keys = ["retry_count", "last_error", "failed_node"]
        for key in temp_keys:
            if key in compressed and compressed.get(key) is None:
                del compressed[key]

        return compressed

    def _serialize_state(self, state: AgentState) -> bytes:
        """Serialize state for storage."""
        try:
            # Use pickle for complex objects
            return pickle.dumps(state)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            # Fallback to JSON for simple data
            return json.dumps(state, default=str).encode()

    def _deserialize_state(self, data: bytes) -> AgentState:
        """Deserialize state from storage."""
        try:
            # Try pickle first
            return pickle.loads(data)
        except Exception:
            # Fallback to JSON
            try:
                return json.loads(data.decode())
            except Exception as e:
                logger.error(f"Deserialization error: {e}")
                return None

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned
        """
        try:
            if self.use_redis and self.redis_client:
                # Redis handles expiry automatically
                return 0
            else:
                # Manual cleanup for memory store
                now = datetime.now()
                expired = []
                for key, entry in self.memory_store.items():
                    if now >= entry["expires"]:
                        expired.append(key)

                for key in expired:
                    del self.memory_store[key]

                return len(expired)

        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session metadata without full state.

        Args:
            session_id: Session ID

        Returns:
            Session info or None
        """
        try:
            state = await self.get_state(session_id)
            if state:
                return {
                    "session_id": session_id,
                    "last_updated": state.get("last_updated"),
                    "user_id": state.get("user_id"),
                    "conversation_language": state.get(
                        "conversation_language", "english"
                    ),
                    "message_count": len(state.get("messages", [])),
                    "has_active_booking": state.get("booking_status") == "confirmed",
                }
            return None

        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return None

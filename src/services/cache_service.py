import redis.asyncio
import os
import json
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class RedisService:
    """A service for interacting with Redis for caching."""

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """
        Initializes the RedisService.

        In a production environment (like Google Cloud Run), these values would
        typically be pulled from environment variables.
        """
        if os.environ.get('GOOGLE_CLOUD_PROJECT'):
            # Production environment on Google Cloud
            self.redis_host = os.environ.get('REDIS_HOST', 'localhost')
            self.redis_port = int(os.environ.get('REDIS_PORT', 6379))
        else:
            # Local development environment (e.g., with Docker)
            self.redis_host = host
            self.redis_port = port

        self.redis_client = redis.asyncio.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=db,
            decode_responses=True  # Decode responses to UTF-8
        )

    async def set(self, key: str, value: Any, expiration_seconds: int = 300):
        """
        Sets a value in the Redis cache.

        Args:
            key: The cache key.
            value: The value to cache (will be JSON serialized).
            expiration_seconds: The expiration time for the key in seconds.
        """
        try:
            serialized_value = json.dumps(value)
            await self.redis_client.setex(key, expiration_seconds, serialized_value)
        except redis.RedisError as e:
            logger.error(f"Error setting cache for key {key}: {e}")
            # Depending on the desired error handling, you might want to log this
            # or raise an exception.

    async def get(self, key: str) -> Optional[Any]:
        """
        Gets a value from the Redis cache.

        Args:
            key: The cache key.

        Returns:
            The deserialized value if the key exists, otherwise None.
        """
        try:
            cached_value = await self.redis_client.get(key)
            if cached_value:
                return json.loads(cached_value)
            return None
        except redis.RedisError as e:
            logger.error(f"Error getting cache for key {key}: {e}")
            return None

    async def delete(self, key: str):
        """
        Deletes a key from the Redis cache.

        Args:
            key: The cache key to delete.
        """
        try:
            await self.redis_client.delete(key)
        except redis.RedisError as e:
            logger.error(f"Error deleting cache for key {key}: {e}")

    async def clear_all(self):
        """
        Clears the entire Redis database.
        Use with caution, especially in production.
        """
        try:
            await self.redis_client.flushdb()
        except redis.RedisError as e:
            logger.error(f"Error clearing Redis cache: {e}")

    async def ping(self) -> bool:
        """
        Pings the Redis server to check if it's alive.

        Returns:
            True if the server is alive, False otherwise.
        """
        try:
            return await self.redis_client.ping()
        except redis.RedisError:
            return False

    async def exists(self, key: str) -> bool:
        """
        Checks if a key exists in the Redis cache.

        Args:
            key: The cache key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        try:
            return await self.redis_client.exists(key) > 0
        except redis.RedisError as e:
            logger.error(f"Error checking if key exists: {e}")
        return False

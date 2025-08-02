"""
Builder for the cab booking agent.
Assembles all components into a complete working agent.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI

from src.models.agent_state_model import AgentState
from src.services.api_service import DriversAPIClient
from src.lngraph.tools.driver_tools import DriverTools
from src.lngraph.graph import CabBookingGraph
from src.lngraph.state_manager import StateManager

logger = logging.getLogger(__name__)


class CabBookingAgentBuilder:
    """
    Builder class for creating and managing the cab booking agent.
    Handles initialization, configuration, and lifecycle management.
    """

    def __init__(
        self,
        # LLM Configuration
        llm_model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        # API Configuration
        api_session_id: Optional[str] = None,
        cache_duration_minutes: int = 5,
        # State Management
        use_redis: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        session_ttl: int = 1800,
        # Agent Configuration
        default_page_size: int = 10,
        max_retry_attempts: int = 3,
        enable_logging: bool = True,
    ):
        """
        Initialize the agent builder with configuration.

        Args:
            llm_model: Vertex AI model name
            temperature: LLM temperature for response generation
            max_output_tokens: Maximum tokens in LLM response
            api_session_id: Session ID for API client
            cache_duration_minutes: API cache duration
            use_redis: Whether to use Redis for state
            redis_host: Redis host address
            redis_port: Redis port
            session_ttl: Session timeout in seconds
            default_page_size: Default results per page
            max_retry_attempts: Maximum retry attempts for errors
            enable_logging: Enable detailed logging
        """
        self.config = {
            "llm": {
                "model": llm_model,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
            "api": {
                "session_id": api_session_id or str(uuid.uuid4()),
                "cache_duration": cache_duration_minutes,
            },
            "state": {
                "use_redis": use_redis,
                "redis_host": redis_host,
                "redis_port": redis_port,
                "session_ttl": session_ttl,
            },
            "agent": {
                "page_size": default_page_size,
                "max_retries": max_retry_attempts,
            },
        }

        # Configure logging
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        # Component instances
        self.llm = None
        self.api_client = None
        self.driver_tools = None
        self.state_manager = None
        self.graph = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all agent components."""
        try:
            # Initialize LLM
            logger.info(f"Initializing LLM: {self.config['llm']['model']}")
            self.llm = ChatVertexAI(
                model=self.config["llm"]["model"],
                temperature=self.config["llm"]["temperature"],
                max_output_tokens=self.config["llm"]["max_output_tokens"],
            )

            # Initialize API Client
            logger.info("Initializing API client")
            self.api_client = DriversAPIClient(
                session_id=self.config["api"]["session_id"],
                cache_duration_minutes=self.config["api"]["cache_duration"],
            )

            # Initialize Driver Tools
            logger.info("Initializing driver tools")
            self.driver_tools = DriverTools(self.api_client)

            # Initialize State Manager
            logger.info("Initializing state manager")
            self.state_manager = StateManager(
                redis_host=self.config["state"]["redis_host"],
                redis_port=self.config["state"]["redis_port"],
                use_redis=self.config["state"]["use_redis"],
                session_ttl=self.config["state"]["session_ttl"],
            )

            # Initialize Graph
            logger.info("Building conversation graph")
            self.graph = CabBookingGraph(self.llm, self.driver_tools)

            logger.info("Agent initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise

    async def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.

        Args:
            user_id: Optional user identifier

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())

        # Create initial state
        initial_state: AgentState = {
            "session_id": session_id,
            "user_id": user_id,
            "messages": [],
            "conversation_language": "english",
            "current_page": 1,
            "page_size": self.config["agent"]["page_size"],
            "total_results": 0,
            "has_more_results": False,
            "active_filters": {},
            "previous_filters": [],
            "current_drivers": [],
            "driver_history": [],
            "booking_status": "none",
            "retry_count": 0,
            "cache_keys_used": [],
            "user_preferences": {},
            "awaiting_user_input": True,
            "created_at": datetime.now().isoformat(),
            "next_node": "wait_for_user_input",  # Add this
        }

        # Send welcome message
        welcome_msg = self._get_welcome_message()
        initial_state["messages"].append(AIMessage(content=welcome_msg))

        # Save initial state
        await self.state_manager.save_state(session_id, initial_state)

        logger.info(f"Created new session: {session_id}")
        return session_id

    async def process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Process a user message in a session.

        Args:
            session_id: Session identifier
            message: User message

        Returns:
            Response with updated state and messages
        """
        try:
            # Retrieve current state
            state = await self.state_manager.get_state(session_id)
            if not state:
                return {
                    "error": "Session not found or expired",
                    "session_id": session_id,
                    "action": "create_new_session",
                }

            # Check for special commands
            command_result = await self.graph.handle_special_commands(message, state)
            if command_result:
                await self.state_manager.save_state(session_id, command_result)
                return self._format_response(command_result)

            # Process message through graph
            logger.info(f"Processing message in session {session_id}")
            updated_state = await self.graph.process_message(message, state, session_id)

            # Save updated state
            await self.state_manager.save_state(session_id, updated_state)

            # Format response
            return self._format_response(updated_state)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "error": "Failed to process message",
                "details": str(e),
                "session_id": session_id,
            }

    async def get_session_state(self, session_id: str) -> Optional[AgentState]:
        """
        Get current state for a session.

        Args:
            session_id: Session identifier

        Returns:
            Current state or None
        """
        return await self.state_manager.get_state(session_id)

    async def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions.

        Returns:
            List of session information
        """
        session_ids = await self.state_manager.list_active_sessions()
        sessions = []

        for session_id in session_ids:
            info = await self.state_manager.get_session_info(session_id)
            if info:
                sessions.append(info)

        return sessions

    async def end_session(self, session_id: str) -> bool:
        """
        End a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        return await self.state_manager.delete_state(session_id)

    def _format_response(self, state: AgentState) -> Dict[str, Any]:
        """
        Format state into API response.

        Args:
            state: Current state

        Returns:
            Formatted response
        """
        # Get last AI message
        last_message = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                last_message = msg.content
                break

        # Extract relevant information
        response = {
            "session_id": state.get("session_id"),
            "message": last_message,
            "awaiting_input": state.get("awaiting_user_input", True),
            "context": {
                "city": state.get("pickup_city"),
                "destination": state.get("destination_city"),
                "drivers_shown": len(state.get("current_drivers", [])),
                "selected_driver": state.get("selected_driver", {}).get("name")
                if state.get("selected_driver")
                else None,
                "booking_status": state.get("booking_status", "none"),
                "active_filters": state.get("active_filters", {}),
                "language": state.get("conversation_language", "english"),
            },
        }

        # Add quick actions if available
        if state.get("quick_city_options"):
            response["quick_actions"] = {
                "type": "city_selection",
                "options": state["quick_city_options"],
            }
        elif state.get("current_drivers"):
            response["quick_actions"] = {
                "type": "driver_selection",
                "count": len(state["current_drivers"]),
            }

        return response

    def _get_welcome_message(self) -> str:
        """Get welcome message for new sessions."""
        return (
            "Hello! I'm your cab booking assistant. I can help you find and book "
            "verified drivers in any Indian city. Just tell me where you need a ride, "
            "and I'll show you the best available options."
        )

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Close API client
            if self.api_client:
                await self.api_client.close()

            # Cleanup expired sessions
            if self.state_manager:
                cleaned = await self.state_manager.cleanup_expired_sessions()
                logger.info(f"Cleaned up {cleaned} expired sessions")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent configuration and status information."""
        return {
            "version": "1.0.0",
            "llm_model": self.config["llm"]["model"],
            "components": {
                "llm": self.llm is not None,
                "api_client": self.api_client is not None,
                "driver_tools": self.driver_tools is not None,
                "state_manager": self.state_manager is not None,
                "graph": self.graph is not None,
            },
            "configuration": {
                "session_ttl": self.config["state"]["session_ttl"],
                "cache_duration": self.config["api"]["cache_duration"],
                "page_size": self.config["agent"]["page_size"],
                "using_redis": self.config["state"]["use_redis"],
            },
        }

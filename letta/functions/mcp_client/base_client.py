import asyncio
from typing import List, Optional, Tuple

from mcp import ClientSession
from mcp.types import TextContent

from letta.functions.mcp_client.exceptions import MCPTimeoutError
from letta.functions.mcp_client.types import BaseServerConfig, MCPTool
from letta.log import get_logger
from letta.settings import tool_settings
from letta.utils import run_async_task

logger = get_logger(__name__)


class ClosedResourceError(Exception):
    """Exception raised when trying to use a closed MCP resource"""
    pass


class BaseMCPClient:
    def __init__(self, server_config: BaseServerConfig):
        self.server_config = server_config
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        self.initialized = False
        self.cleanup_funcs = []
        self.max_reconnect_attempts = 3
        self.reconnect_delay = 1.0


    def connect_to_server(self):
        success = self._initialize_connection(self.server_config, timeout=tool_settings.mcp_connect_to_server_timeout)

        if success:
            try:
                run_async_task(
                    asyncio.wait_for(self.session.initialize(), timeout=tool_settings.mcp_connect_to_server_timeout)
                )
                self.initialized = True
            except asyncio.TimeoutError:
                raise MCPTimeoutError("initializing session", self.server_config.server_name, tool_settings.mcp_connect_to_server_timeout)
        else:
            raise RuntimeError(
                f"Connecting to MCP server failed. Please review your server config: {self.server_config.model_dump_json(indent=4)}"
            )

    def _initialize_connection(self, server_config: BaseServerConfig, timeout: float) -> bool:
        raise NotImplementedError("Subclasses must implement _initialize_connection")

    def list_tools(self) -> List[MCPTool]:
        self._check_initialized()
        
        def _list_tools():
            try:
                response = run_async_task(
                    asyncio.wait_for(self.session.list_tools(), timeout=tool_settings.mcp_list_tools_timeout)
                )
                return response.tools
            except asyncio.TimeoutError:
                logger.error(
                    f"Timed out while listing tools for MCP server {self.server_config.server_name} (timeout={tool_settings.mcp_list_tools_timeout}s)."
                )
                raise MCPTimeoutError("listing tools", self.server_config.server_name, tool_settings.mcp_list_tools_timeout)
        
        return self._execute_with_retry("list_tools", _list_tools)

    def execute_tool(self, tool_name: str, tool_args: dict) -> Tuple[str, bool]:
        self._check_initialized()
        
        def _execute_tool():
            try:
                result = run_async_task(
                    asyncio.wait_for(self.session.call_tool(tool_name, tool_args), timeout=tool_settings.mcp_execute_tool_timeout)
                )

                parsed_content = []
                for content_piece in result.content:
                    if isinstance(content_piece, TextContent):
                        parsed_content.append(content_piece.text)
                        print("parsed_content (text)", parsed_content)
                    else:
                        parsed_content.append(str(content_piece))
                        print("parsed_content (other)", parsed_content)

                if len(parsed_content) > 0:
                    final_content = " ".join(parsed_content)
                else:
                    # TODO move hardcoding to constants
                    final_content = "Empty response from tool"

                return final_content, result.isError
            except asyncio.TimeoutError:
                logger.error(
                    f"Timed out while executing tool '{tool_name}' for MCP server {self.server_config.server_name} (timeout={tool_settings.mcp_execute_tool_timeout}s)."
                )
                raise MCPTimeoutError(f"executing tool '{tool_name}'", self.server_config.server_name, tool_settings.mcp_execute_tool_timeout)
        
        return self._execute_with_retry(f"execute_tool({tool_name})", _execute_tool)

    def _check_initialized(self):
        if not self.initialized:
            logger.error("MCPClient has not been initialized")
            raise RuntimeError("MCPClient has not been initialized")

    def _is_connection_closed(self, exception) -> bool:
        """Check if the exception indicates a closed connection"""
        error_msg = str(exception).lower()
        return any(keyword in error_msg for keyword in [
            'closed', 'connection closed', 'stream closed', 'transport closed',
            'closedresourceerror', 'connectionclosed', 'broken pipe'
        ])

    def _reconnect(self) -> bool:
        """Attempt to reconnect to the MCP server"""
        logger.info(f"Attempting to reconnect to MCP server: {self.server_config.server_name}")
        
        # Clean up existing connection
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup before reconnect: {e}")
        
        # Reset state
        self.initialized = False
        self.session = None
        self.stdio = None
        self.write = None
        self.cleanup_funcs = []
        
        # Attempt reconnection
        try:
            self.connect_to_server()
            logger.info(f"Successfully reconnected to MCP server: {self.server_config.server_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reconnect to MCP server {self.server_config.server_name}: {e}")
            return False

    def _execute_with_retry(self, operation_name: str, operation_func):
        """Execute an operation with automatic retry on connection errors"""
        for attempt in range(self.max_reconnect_attempts):
            try:
                return operation_func()
            except Exception as e:
                if self._is_connection_closed(e):
                    logger.warning(f"Connection closed during {operation_name} (attempt {attempt + 1}/{self.max_reconnect_attempts}): {e}")
                    
                    if attempt < self.max_reconnect_attempts - 1:
                        # Sleep before retry
                        import time
                        time.sleep(self.reconnect_delay * (attempt + 1))
                        
                        # Attempt reconnection
                        if self._reconnect():
                            continue
                        else:
                            logger.error(f"Reconnection failed for {operation_name}")
                    else:
                        logger.error(f"Max reconnection attempts reached for {operation_name}")
                        raise RuntimeError(f"MCP server connection failed after {self.max_reconnect_attempts} attempts")
                else:
                    # Re-raise non-connection errors immediately
                    raise

    def is_connection_healthy(self) -> bool:
        """Check if the connection is healthy by attempting to list tools"""
        if not self.initialized:
            return False
        
        try:
            # Quick health check - list tools with shorter timeout
            run_async_task(
                asyncio.wait_for(self.session.list_tools(), timeout=5.0)
            )
            return True
        except Exception as e:
            if self._is_connection_closed(e):
                logger.debug(f"Health check failed - connection closed: {e}")
                return False
            else:
                logger.warning(f"Health check failed with non-connection error: {e}")
                return False

    def cleanup(self):
        try:
            for cleanup_func in self.cleanup_funcs:
                cleanup_func()
            self.initialized = False
        except Exception as e:
            logger.warning(e)
        finally:
            logger.info("Cleaned up MCP clients on shutdown.")

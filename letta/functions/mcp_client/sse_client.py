import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client

from letta.functions.mcp_client.base_client import BaseMCPClient
from letta.functions.mcp_client.types import SSEServerConfig
from letta.log import get_logger

# see: https://modelcontextprotocol.io/quickstart/user
MCP_CONFIG_TOPLEVEL_KEY = "mcpServers"

logger = get_logger(__name__)


class SSEMCPClient(BaseMCPClient):
    def _initialize_connection(self, server_config: SSEServerConfig, timeout: float) -> bool:
        try:
            sse_cm = sse_client(url=server_config.server_url)
            sse_transport = self.loop.run_until_complete(asyncio.wait_for(sse_cm.__aenter__(), timeout=timeout))
            self.stdio, self.write = sse_transport
            
            # Store context managers for proper cleanup
            self.sse_cm = sse_cm
            self.cleanup_funcs.append(self._safe_sse_cleanup)

            session_cm = ClientSession(self.stdio, self.write)
            self.session = self.loop.run_until_complete(asyncio.wait_for(session_cm.__aenter__(), timeout=timeout))
            
            # Store session context manager for proper cleanup  
            self.session_cm = session_cm
            self.cleanup_funcs.append(self._safe_session_cleanup)
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timed out while establishing SSE connection (timeout={timeout}s).")
            return False
        except Exception:
            logger.exception("Exception occurred while initializing SSE client session.")
            return False
    
    def _safe_sse_cleanup(self):
        """Safely cleanup SSE connection, handling ClosedResourceError"""
        try:
            if hasattr(self, 'sse_cm') and self.sse_cm:
                self.loop.run_until_complete(self.sse_cm.__aexit__(None, None, None))
        except Exception as e:
            if self._is_connection_closed(e):
                logger.debug(f"SSE connection already closed during cleanup: {e}")
            else:
                logger.warning(f"Error during SSE cleanup: {e}")
    
    def _safe_session_cleanup(self):
        """Safely cleanup session, handling ClosedResourceError"""
        try:
            if hasattr(self, 'session_cm') and self.session_cm:
                self.loop.run_until_complete(self.session_cm.__aexit__(None, None, None))
        except Exception as e:
            if self._is_connection_closed(e):
                logger.debug(f"Session already closed during cleanup: {e}")
            else:
                logger.warning(f"Error during session cleanup: {e}")

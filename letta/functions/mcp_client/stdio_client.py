import asyncio
import sys
from contextlib import asynccontextmanager

import anyio
import anyio.lowlevel
import mcp.types as types
from anyio.streams.text import TextReceiveStream
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment

from letta.functions.mcp_client.base_client import BaseMCPClient
from letta.functions.mcp_client.types import StdioServerConfig
from letta.log import get_logger
from letta.utils import run_async_task

logger = get_logger(__name__)


class StdioMCPClient(BaseMCPClient):
    def _initialize_connection(self, server_config: StdioServerConfig, timeout: float) -> bool:
        try:
            server_params = StdioServerParameters(command=server_config.command, args=server_config.args, env=server_config.env)
            stdio_cm = forked_stdio_client(server_params)
            stdio_transport = run_async_task(asyncio.wait_for(stdio_cm.__aenter__(), timeout=timeout))
            self.stdio, self.write = stdio_transport
            
            # Store context managers for proper cleanup
            self.stdio_cm = stdio_cm
            self.cleanup_funcs.append(self._safe_stdio_cleanup)

            session_cm = ClientSession(self.stdio, self.write)
            self.session = run_async_task(asyncio.wait_for(session_cm.__aenter__(), timeout=timeout))
            
            # Store session context manager for proper cleanup
            self.session_cm = session_cm
            self.cleanup_funcs.append(self._safe_session_cleanup)
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timed out while establishing stdio connection (timeout={timeout}s).")
            return False
        except Exception:
            logger.exception("Exception occurred while initializing stdio client session.")
            return False

    def _safe_stdio_cleanup(self):
        """Safely cleanup stdio connection, handling ClosedResourceError"""
        try:
            if hasattr(self, 'stdio_cm') and self.stdio_cm:
                run_async_task(self.stdio_cm.__aexit__(None, None, None))
        except Exception as e:
            if self._is_connection_closed(e):
                logger.debug(f"Stdio connection already closed during cleanup: {e}")
            else:
                logger.warning(f"Error during stdio cleanup: {e}")
    
    def _safe_session_cleanup(self):
        """Safely cleanup session, handling ClosedResourceError"""
        try:
            if hasattr(self, 'session_cm') and self.session_cm:
                run_async_task(self.session_cm.__aexit__(None, None, None))
        except Exception as e:
            if self._is_connection_closed(e):
                logger.debug(f"Session already closed during cleanup: {e}")
            else:
                logger.warning(f"Error during session cleanup: {e}")


@asynccontextmanager
async def forked_stdio_client(server: StdioServerParameters):
    """
    Client transport for stdio: this will connect to a server by spawning a
    process and communicating with it over stdin/stdout.
    """
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    try:
        process = await anyio.open_process(
            [server.command, *server.args],
            env=server.env or get_default_environment(),
            stderr=sys.stderr,  # Consider logging stderr somewhere instead of silencing it
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to spawn process: {server.command} {server.args}") from exc

    async def stdout_reader():
        assert process.stdout, "Opened process is missing stdout"
        buffer = ""
        try:
            async with read_stream_writer:
                async for chunk in TextReceiveStream(
                    process.stdout,
                    encoding=server.encoding,
                    errors=server.encoding_error_handler,
                ):
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()
                    for line in lines:
                        try:
                            message = types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:
                            await read_stream_writer.send(exc)
                            continue
                        await read_stream_writer.send(message)
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def stdin_writer():
        assert process.stdin, "Opened process is missing stdin"
        try:
            async with write_stream_reader:
                async for message in write_stream_reader:
                    json = message.model_dump_json(by_alias=True, exclude_none=True)
                    await process.stdin.send(
                        (json + "\n").encode(
                            encoding=server.encoding,
                            errors=server.encoding_error_handler,
                        )
                    )
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def watch_process_exit():
        returncode = await process.wait()
        if returncode != 0:
            raise RuntimeError(f"Subprocess exited with code {returncode}. Command: {server.command} {server.args}")

    async with anyio.create_task_group() as tg, process:
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)
        tg.start_soon(watch_process_exit)

        with anyio.move_on_after(0.2):
            await anyio.sleep_forever()

        yield read_stream, write_stream

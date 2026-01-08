"""
Parlant Agent Factory for CAE System.

Creates and manages the Scribe Agent subprocess with Parlant framework.
Adapted from existing parlant_subprocess.py pattern.
"""

import os
import sys
import asyncio
import subprocess
import tempfile
from typing import Optional
import threading
import time

import httpx

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    PARLANT_PORT,
    PARLANT_STARTUP_TIMEOUT,
    PARLANT_RESPONSE_TIMEOUT,
)
from utils.logging import get_logger
from utils.exceptions import ParlantError, ParlantNotReadyError, ParlantTimeoutError

logger = get_logger(__name__)


# =============================================================================
# PARLANT SERVER SCRIPT
# =============================================================================

PARLANT_SERVER_SCRIPT = '''
"""Parlant Scribe Agent Server - Runs in subprocess with HTTP server"""
import os
import sys
import asyncio

# Set environment variables BEFORE importing parlant
os.environ["OLLAMA_MODEL"] = "{ollama_model}"
os.environ["OLLAMA_BASE_URL"] = "{ollama_base_url}"
os.environ["OLLAMA_EMBEDDING_MODEL"] = "nomic-embed-text"
os.environ["OLLAMA_API_TIMEOUT"] = "300"

async def main():
    from parlant.bin.server import start_parlant, StartupParameters
    from parlant.core.loggers import LogLevel

    # Create async NLP service factory for Ollama
    async def create_ollama_service(container):
        from parlant.core.loggers import Logger
        from parlant.adapters.nlp.ollama_service import OllamaService
        return OllamaService(container[Logger])

    port = {port}

    params = StartupParameters(
        port=port,
        nlp_service=create_ollama_service,
        log_level=LogLevel.WARNING,
        modules=[],
        migrate=False,
    )

    print("PARLANT_STARTING", flush=True)

    async with start_parlant(params) as container:
        # Create agent BEFORE serve_app runs
        from parlant.core.agents import AgentStore

        agent_store = container[AgentStore]

        # Check for existing agent
        agents = await agent_store.list_agents()
        agent_id = None

        for agent in agents:
            if agent.name == "CAEScribeAgent":
                agent_id = agent.id
                break

        if not agent_id:
            # Create new Scribe Agent
            agent = await agent_store.create_agent(
                name="CAEScribeAgent",
                description="Clinical Admin Edge Scribe Agent. Administrative assistant for medical documentation. Does NOT provide clinical diagnoses.",
            )
            agent_id = agent.id

        # Signal agent ready
        print(f"AGENT_ID={{agent_id}}", flush=True)
        print("SERVER_READY", flush=True)

        # Context manager will run serve_app() which blocks

if __name__ == "__main__":
    asyncio.run(main())
'''


# =============================================================================
# PARLANT SUBPROCESS MANAGER
# =============================================================================

class ParlantSubprocess:
    """Manages Parlant as a subprocess with HTTP API."""

    def __init__(self, port: int = PARLANT_PORT):
        self.port = port
        self._process: Optional[subprocess.Popen] = None
        self._agent_id: Optional[str] = None
        self._script_file: Optional[str] = None

    @property
    def is_running(self) -> bool:
        """Check if subprocess is running."""
        return self._process is not None and self._process.poll() is None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    async def start(self, timeout: float = None) -> bool:
        """Start Parlant subprocess and wait for HTTP API."""
        timeout = timeout or PARLANT_STARTUP_TIMEOUT

        if self.is_running:
            logger.info("Parlant subprocess already running")
            return True

        # Create script content
        script_content = PARLANT_SERVER_SCRIPT.format(
            ollama_model=OLLAMA_MODEL,
            ollama_base_url=OLLAMA_BASE_URL,
            port=self.port,
        )

        # Write to temp file
        fd, self._script_file = tempfile.mkstemp(suffix='.py', prefix='parlant_cae_')
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)

        logger.info("Starting Parlant subprocess", port=self.port)

        # Build environment
        env = os.environ.copy()
        env["OLLAMA_MODEL"] = OLLAMA_MODEL
        env["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
        env["OLLAMA_EMBEDDING_MODEL"] = "nomic-embed-text"
        env["OLLAMA_API_TIMEOUT"] = "300"

        # Start process
        self._process = subprocess.Popen(
            [sys.executable, self._script_file],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Wait for ready signal
        ready_event = threading.Event()
        agent_id_container = [None]
        error_lines = []

        def read_stdout():
            try:
                for line in iter(self._process.stdout.readline, ''):
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue

                    if any(x in line for x in ['PARLANT', 'AGENT', 'SERVER', 'Error']):
                        logger.debug("Parlant output", line=line)

                    if line.startswith("AGENT_ID="):
                        agent_id_container[0] = line.split("=", 1)[1]
                    elif line == "SERVER_READY":
                        ready_event.set()
            except Exception as e:
                logger.debug("Stdout reader error", error=str(e))

        def read_stderr():
            try:
                for line in iter(self._process.stderr.readline, ''):
                    if not line:
                        break
                    line = line.strip()
                    if line and 'DeprecationWarning' not in line:
                        error_lines.append(line)
                        if 'error' in line.lower():
                            logger.warning("Parlant stderr", line=line)
            except Exception:
                pass

        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self._process.poll() is not None:
                logger.error("Parlant subprocess exited", code=self._process.returncode)
                if error_lines:
                    logger.error("Errors", lines=error_lines[-5:])
                return False

            if ready_event.is_set():
                self._agent_id = agent_id_container[0]
                await asyncio.sleep(2.0)  # Wait for HTTP server
                if await self.health_check():
                    logger.info("Parlant ready", url=self.base_url, agent_id=self._agent_id)
                    return True

            await asyncio.sleep(2.0)

        logger.warning("Parlant startup timeout", timeout=timeout)
        await self.stop()
        return False

    async def stop(self):
        """Stop Parlant subprocess."""
        if self._process:
            logger.info("Stopping Parlant subprocess")
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
            except Exception as e:
                logger.debug("Stop error", error=str(e))
            self._process = None

        if self._script_file and os.path.exists(self._script_file):
            try:
                os.unlink(self._script_file)
            except Exception:
                pass
            self._script_file = None

        self._agent_id = None

    async def health_check(self) -> bool:
        """Check if Parlant HTTP is responding."""
        if not self.is_running:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/agents")
                return resp.status_code == 200
        except Exception:
            return False


# =============================================================================
# SCRIBE AGENT
# =============================================================================

class ScribeAgent:
    """CAE Scribe Agent - wraps Parlant for document generation."""

    def __init__(self, subprocess: ParlantSubprocess):
        self._subprocess = subprocess
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_ready(self) -> bool:
        return self._subprocess.is_running

    @property
    def agent_id(self) -> Optional[str]:
        return self._subprocess.agent_id

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._subprocess.base_url,
                timeout=httpx.Timeout(PARLANT_RESPONSE_TIMEOUT),
            )
        return self._client

    async def generate_response(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Generate response from Scribe Agent.

        Args:
            prompt: User prompt
            session_id: Optional Parlant session ID

        Returns:
            Generated response text
        """
        if not self.is_ready:
            raise ParlantNotReadyError()

        try:
            client = await self._get_client()

            # Create session if needed
            if not session_id:
                session_resp = await client.post(
                    "/sessions",
                    json={
                        "agent_id": self.agent_id,
                        "customer_id": "cae_user",
                    },
                )
                session_resp.raise_for_status()
                session_id = session_resp.json()["session_id"]

            # Post message
            msg_resp = await client.post(
                f"/sessions/{session_id}/events",
                json={
                    "kind": "message",
                    "source": "customer",
                    "message": prompt,
                },
            )
            msg_resp.raise_for_status()

            # Wait for response
            await asyncio.sleep(0.5)

            # Get agent response
            events_resp = await client.get(f"/sessions/{session_id}/events")
            events_resp.raise_for_status()
            events = events_resp.json()

            # Find latest agent message
            for event in reversed(events):
                if event.get("source") == "agent" and event.get("kind") == "message":
                    return event.get("message", "")

            return ""

        except httpx.TimeoutException:
            raise ParlantTimeoutError(PARLANT_RESPONSE_TIMEOUT)
        except Exception as e:
            raise ParlantError(f"Generation failed: {str(e)}")

    async def close(self):
        """Close client connection."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# SINGLETON MANAGEMENT
# =============================================================================

_parlant_subprocess: Optional[ParlantSubprocess] = None
_scribe_agent: Optional[ScribeAgent] = None


async def create_scribe_agent() -> ScribeAgent:
    """Create and initialize the Scribe Agent."""
    global _parlant_subprocess, _scribe_agent

    if _scribe_agent is not None and _scribe_agent.is_ready:
        return _scribe_agent

    # Create subprocess
    if _parlant_subprocess is None:
        _parlant_subprocess = ParlantSubprocess()

    # Start if not running
    if not _parlant_subprocess.is_running:
        success = await _parlant_subprocess.start()
        if not success:
            raise ParlantError("Failed to start Parlant subprocess")

    # Create agent wrapper
    _scribe_agent = ScribeAgent(_parlant_subprocess)
    return _scribe_agent


async def shutdown_parlant():
    """Shutdown Parlant subprocess."""
    global _parlant_subprocess, _scribe_agent

    if _scribe_agent:
        await _scribe_agent.close()
        _scribe_agent = None

    if _parlant_subprocess:
        await _parlant_subprocess.stop()
        _parlant_subprocess = None

    logger.info("Parlant shutdown complete")


async def get_parlant_status() -> str:
    """Get Parlant service status."""
    global _parlant_subprocess
    if _parlant_subprocess is None:
        return "not_initialized"
    if not _parlant_subprocess.is_running:
        return "stopped"
    if await _parlant_subprocess.health_check():
        return "ready"
    return "starting"

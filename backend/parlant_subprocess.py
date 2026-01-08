"""
Parlant Subprocess Server - Run Parlant HTTP server in a separate process

The Parlant SDK 3.0's Server class yields control BEFORE start_parlant() calls
serve_app(). This means the HTTP server never starts when using the SDK directly.

This module spawns a subprocess that:
1. Uses start_parlant() properly (with serve_app blocking)
2. Creates an agent and outputs its ID
3. Runs until terminated
"""

import os
import sys
import asyncio
import subprocess
import logging
import tempfile
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    PARLANT_PORT,
)

PARLANT_STARTUP_TIMEOUT = 120  # seconds
PARLANT_HEALTH_CHECK_INTERVAL = 2  # seconds


# Server script template - runs in subprocess
# Uses start_parlant() which properly blocks on serve_app()
PARLANT_SERVER_SCRIPT = '''
"""Parlant Server Script - Runs in subprocess with HTTP server"""
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
        # Create agent BEFORE serve_app runs (we're still in the yield)
        from parlant.core.agents import AgentStore

        agent_store = container[AgentStore]

        # Check for existing agent
        agents = await agent_store.list_agents()
        agent_id = None

        for agent in agents:
            if agent.name == "MedicalTriageAgent":
                agent_id = agent.id
                break

        if not agent_id:
            # Create new agent
            agent = await agent_store.create_agent(
                name="MedicalTriageAgent",
                description="Medical triage assistant for emergency department. Supports EN (Manchester) and FR (SFMU).",
            )
            agent_id = agent.id

        # Signal agent ready
        print(f"AGENT_ID={{agent_id}}", flush=True)
        print("SERVER_READY", flush=True)

        # The context manager will now run serve_app() which blocks
        # This keeps the HTTP server running until the process is killed

if __name__ == "__main__":
    asyncio.run(main())
'''


class ParlantSubprocess:
    """
    Manages Parlant as a subprocess with HTTP API.

    This approach ensures the HTTP server actually starts by using
    start_parlant() in its intended blocking mode.
    """

    def __init__(self, port: int = None):
        self.port = port or PARLANT_PORT
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

        # Create script content with configuration
        script_content = PARLANT_SERVER_SCRIPT.format(
            ollama_model=OLLAMA_MODEL,
            ollama_base_url=OLLAMA_BASE_URL,
            port=self.port,
        )

        # Write script to temp file
        fd, self._script_file = tempfile.mkstemp(suffix='.py', prefix='parlant_server_')
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)

        logger.info(f"Starting Parlant subprocess on port {self.port}...")
        logger.debug(f"Script file: {self._script_file}")

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
            bufsize=1,  # Line buffered
        )

        # Wait for ready signal with background reader
        import threading
        import time

        ready_event = threading.Event()
        agent_id_container = [None]
        error_lines = []

        def read_stdout():
            """Read subprocess stdout in background thread."""
            try:
                for line in iter(self._process.stdout.readline, ''):
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue

                    # Log important messages
                    if any(x in line for x in ['PARLANT', 'AGENT', 'SERVER', 'Error', 'error']):
                        logger.info(f"Parlant: {line}")

                    if line.startswith("AGENT_ID="):
                        agent_id_container[0] = line.split("=", 1)[1]
                        logger.info(f"Parlant agent ID: {agent_id_container[0]}")
                    elif line == "SERVER_READY":
                        ready_event.set()
                        logger.info("Parlant subprocess signaled ready")
            except Exception as e:
                logger.debug(f"Stdout reader error: {e}")

        def read_stderr():
            """Read subprocess stderr in background thread."""
            try:
                for line in iter(self._process.stderr.readline, ''):
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        # Filter out noisy warnings
                        if 'DeprecationWarning' not in line and 'websockets' not in line:
                            error_lines.append(line)
                            if 'error' in line.lower() or 'exception' in line.lower():
                                logger.warning(f"Parlant stderr: {line}")
            except Exception:
                pass

        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        start_time = time.time()
        while (time.time() - start_time) < timeout:
            # Check if process crashed
            if self._process.poll() is not None:
                logger.error(f"Parlant subprocess exited with code {self._process.returncode}")
                if error_lines:
                    logger.error(f"Last errors: {error_lines[-5:]}")
                return False

            # Check ready event
            if ready_event.is_set():
                self._agent_id = agent_id_container[0]
                # Wait a moment for HTTP server to fully start
                await asyncio.sleep(2.0)
                # Verify HTTP is actually working
                if await self.health_check():
                    logger.info(f"Parlant HTTP API ready at {self.base_url}")
                    return True
                else:
                    logger.warning("Ready signal received but HTTP not responding, waiting...")

            await asyncio.sleep(PARLANT_HEALTH_CHECK_INTERVAL)

        logger.warning(f"Parlant subprocess not ready after {timeout}s")
        if error_lines:
            logger.warning(f"Stderr output: {error_lines[-10:]}")
        await self.stop()
        return False

    async def stop(self):
        """Stop Parlant subprocess."""
        if self._process:
            logger.info("Stopping Parlant subprocess...")
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
            except Exception as e:
                logger.debug(f"Stop error: {e}")
            self._process = None

        # Clean up temp script
        if self._script_file and os.path.exists(self._script_file):
            try:
                os.unlink(self._script_file)
            except Exception:
                pass
            self._script_file = None

        self._agent_id = None
        logger.info("Parlant subprocess stopped")

    async def health_check(self) -> bool:
        """Check if Parlant HTTP is responding."""
        if not self.is_running:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Parlant doesn't have /health - use /agents endpoint
                resp = await client.get(f"{self.base_url}/agents")
                return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False


# Singleton
_subprocess_instance: Optional[ParlantSubprocess] = None


def get_parlant_subprocess() -> ParlantSubprocess:
    """Get singleton Parlant subprocess instance."""
    global _subprocess_instance
    if _subprocess_instance is None:
        _subprocess_instance = ParlantSubprocess()
    return _subprocess_instance

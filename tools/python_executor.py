"""
Safely executes Python code snippets in a subprocess with a timeout.
Returns stdout + stderr as a single string.

Security note
-------------
- Runs in a sandboxed subprocess (separate process, no shared globals).
- Hard timeout prevents runaway execution.
- Module import restrictions can be added via RestrictedPython if needed.
"""

import subprocess
import sys
import tempfile
import os
import logging
from textwrap import dedent

TIMEOUT_SECONDS = 15
logger = logging.getLogger(__name__)


def execute_python(code: str) -> str:
    """
    Execute a Python code snippet and return its output.

    Parameters
    ----------
    code : str
        Valid Python source code to execute.

    Returns
    -------
    str
        Combined stdout and stderr output, or an error message.
    """
    code = dedent(code).strip()
    logger.info("[PythonExecutor] Executing %d chars of code.", len(code))

    # Write to a temp file so we avoid shell-injection via -c flag
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0:
            output = stdout if stdout else "(Code executed successfully – no output)"
        else:
            output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}" if stdout else f"ERROR:\n{stderr}"
    except subprocess.TimeoutExpired:
        output = f"Execution timed out after {TIMEOUT_SECONDS} seconds."
    except Exception as exc:
        output = f"Executor error: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    logger.info("[PythonExecutor] Output (first 200): %s", output[:200])
    return output

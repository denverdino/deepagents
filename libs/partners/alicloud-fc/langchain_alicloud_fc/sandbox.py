"""Alibaba Cloud Function Compute sandbox backend implementation."""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from agentrun.sandbox.code_interpreter_sandbox import CodeInterpreterSandbox

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
"""Default command timeout in seconds (FC gateway hard limit)."""

_TIMEOUT_EXIT_CODE = 124
"""Exit code returned when a command times out."""

_INLINE_B64_LIMIT = 65536
"""Maximum base64 characters to embed in a *single* execute() call."""

_UPLOAD_CHUNK_SIZE = 65536
"""Base64 characters per chunk when writing large files via execute()."""

_GLOB_FC_TEMPLATE = """python3 -c "
import glob
import os
import json
import sys
import base64

path = base64.b64decode('{path_b64}').decode('utf-8')
pattern = base64.b64decode('{pattern_b64}').decode('utf-8')

os.chdir(path)
kw = dict(recursive=True)
if sys.version_info >= (3, 11):
    kw['include_hidden'] = True
matches = sorted(glob.glob(pattern, **kw))
for m in matches:
    stat = os.stat(m)
    result = {{
        'path': m,
        'size': stat.st_size,
        'mtime': stat.st_mtime,
        'is_dir': os.path.isdir(m)
    }}
    print(json.dumps(result))
" 2>&1"""
"""FC-specific glob template that passes ``include_hidden=True`` on
Python 3.11+ so that patterns like ``.*`` correctly match dot-files."""


class AlicloudFCSandbox(BaseSandbox):
    """Alibaba Cloud FC sandbox conforming to SandboxBackendProtocol.

    Wraps the `agentrun-sdk` ``CodeInterpreterSandbox`` and delegates
    command execution and file transfer to the FC sandbox data-plane API.

    This implementation inherits all file operation methods from
    `BaseSandbox` and only implements the core transport methods.
    """

    def __init__(
        self,
        *,
        sandbox: CodeInterpreterSandbox,
        working_dir: str = "/home/user",
    ) -> None:
        """Create a backend wrapping an existing FC sandbox.

        Args:
            sandbox: An already-created ``CodeInterpreterSandbox`` instance
                from the ``agentrun-sdk`` package.
            working_dir: Default working directory for command execution.
        """
        self._sandbox = sandbox
        self._working_dir = working_dir

    @property
    def id(self) -> str:
        """Return the FC sandbox id."""
        return self._sandbox.sandbox_id or ""

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the FC sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command
                to complete. If None, uses the default timeout (30s).

        Returns:
            `ExecuteResponse` containing output, exit code, and truncation
            flag.
        """
        effective_timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT

        try:
            result: dict[str, Any] = self._sandbox.process.cmd(
                command=command,
                cwd=self._working_dir,
                timeout=effective_timeout,
            )
        except Exception:  # noqa: BLE001 — SDK may raise any exception type on timeout/network failure
            logger.debug("FC sandbox command failed", exc_info=True)
            msg = f"Command timed out after {effective_timeout} seconds"
            return ExecuteResponse(
                output=msg,
                exit_code=_TIMEOUT_EXIT_CODE,
                truncated=False,
            )

        # agentrun-sdk wraps the actual output in a nested "result" dict:
        # {"executionId": ..., "status": ..., "result": {"stdout": ..., "exitCode": ...}}
        inner = result.get("result", result)
        stdout = inner.get("stdout", "") or ""
        stderr = inner.get("stderr", "") or ""
        exit_code = inner.get("exitCode", 1)

        output = stdout
        if stderr.strip():
            output += f"\n<stderr>{stderr.strip()}</stderr>"

        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=False,
        )

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload files into the FC sandbox.

        All files are written via ``execute()`` calls to guarantee
        filesystem visibility.  The SDK's ``file.write()`` API stores
        files in a namespace that may not be visible to ``process.cmd()``
        (especially for dot-files, binary content, or async contexts).

        Small payloads are inlined in a single ``execute()`` call; large
        payloads are base64-encoded, written in chunks to a temporary
        file, then decoded server-side.

        Args:
            files: List of ``(remote_path, content_bytes)`` tuples.

        Returns:
            List of `FileUploadResponse` for each file.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(
                    FileUploadResponse(path=path, error="invalid_path")
                )
                continue
            try:
                # Ensure parent directory exists.
                parent = "/".join(path.rsplit("/", 1)[:-1]) or "/"
                if parent != "/":
                    self.execute(f"mkdir -p {parent}")

                if len(content) == 0:
                    # Empty content: create the file server-side.
                    path_b64 = base64.b64encode(
                        path.encode("utf-8")
                    ).decode("ascii")
                    result = self.execute(
                        "python3 -c \""
                        "import base64; "
                        f"open(base64.b64decode('{path_b64}').decode(), 'w').close()"
                        "\""
                    )
                    if result.exit_code != 0:
                        msg = f"Failed to create empty file: {result.output}"
                        responses.append(
                            FileUploadResponse(path=path, error=msg)
                        )
                        continue
                else:
                    error = self._upload_via_execute(path, content)
                    if error:
                        responses.append(
                            FileUploadResponse(path=path, error=error)
                        )
                        continue

                responses.append(FileUploadResponse(path=path, error=None))
            except Exception as exc:  # noqa: BLE001 — SDK may raise any exception type on I/O failure
                logger.debug(
                    "Failed to upload %s to FC sandbox", path, exc_info=True
                )
                responses.append(
                    FileUploadResponse(path=path, error=str(exc))
                )
        return responses

    def _upload_via_execute(self, path: str, content: bytes) -> str | None:
        """Write *content* to *path* using only ``execute()`` calls.

        Small payloads (base64 length <= ``_INLINE_B64_LIMIT``) are
        written in a single ``execute()`` call.  Larger payloads are
        split into chunks, written to a temporary file, then decoded
        server-side.

        Args:
            path: Absolute target path inside the sandbox.
            content: Raw bytes to write.

        Returns:
            ``None`` on success, or an error message string on failure.
        """
        b64_text = base64.b64encode(content).decode("ascii")
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")

        if len(b64_text) <= _INLINE_B64_LIMIT:
            # Small payload — single execute() call.
            cmd = (
                "python3 -c \""
                "import base64; "
                f"p = base64.b64decode('{path_b64}').decode(); "
                f"open(p, 'wb').write(base64.b64decode('{b64_text}'))"
                "\""
            )
            result = self.execute(cmd)
            if result.exit_code != 0:
                return f"Failed to write file: {result.output}"
            return None

        # Large payload — write base64 chunks to a temp file, then decode.
        tmp_suffix = ".__b64__"
        tmp_b64 = base64.b64encode(
            f"{path}{tmp_suffix}".encode("utf-8")
        ).decode("ascii")

        for i in range(0, len(b64_text), _UPLOAD_CHUNK_SIZE):
            chunk = b64_text[i : i + _UPLOAD_CHUNK_SIZE]
            mode = "w" if i == 0 else "a"
            write_cmd = (
                "python3 -c \""
                "import base64; "
                f"p = base64.b64decode('{tmp_b64}').decode(); "
                f"open(p, '{mode}').write('{chunk}')"
                "\""
            )
            result = self.execute(write_cmd)
            if result.exit_code != 0:
                return f"Failed to write chunk: {result.output}"

        decode_cmd = (
            "python3 -c \""
            "import base64, os; "
            f"p = base64.b64decode('{path_b64}').decode(); "
            f"t = base64.b64decode('{tmp_b64}').decode(); "
            "open(p, 'wb').write(base64.b64decode(open(t).read())); "
            "os.remove(t)"
            "\""
        )
        result = self.execute(decode_cmd)
        if result.exit_code != 0:
            return f"Failed to decode file content: {result.output}"
        return None

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Glob with ``include_hidden=True`` on Python 3.11+.

        The upstream ``BaseSandbox.glob()`` uses the default
        ``include_hidden=False``, which prevents patterns like ``.*``
        from matching dot-files on Python 3.11+.  This override uses a
        modified server-side script that enables hidden-file matching.

        Args:
            pattern: Glob pattern (e.g. ``"*.py"`` or ``".*"``).
            path: Directory to search in.

        Returns:
            `GlobResult` with matching entries.
        """
        pattern_b64 = base64.b64encode(
            pattern.encode("utf-8")
        ).decode("ascii")
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")

        cmd = _GLOB_FC_TEMPLATE.format(
            path_b64=path_b64, pattern_b64=pattern_b64
        )
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return GlobResult(matches=[])

        file_infos: list[FileInfo] = []
        for line in output.split("\n"):
            try:
                data = json.loads(line)
                file_infos.append(
                    {"path": data["path"], "is_dir": data["is_dir"]}
                )
            except (json.JSONDecodeError, KeyError):
                continue

        return GlobResult(matches=file_infos)

    def download_files(
        self, paths: list[str]
    ) -> list[FileDownloadResponse]:
        """Download files from the FC sandbox.

        Uses the agentrun-sdk ``file.read()`` API which returns content
        as a dict with ``content`` and ``encoding`` keys.  A server-side
        pre-check detects missing files, directories, and permission errors
        before attempting the download.

        Args:
            paths: List of absolute file paths to download.

        Returns:
            List of `FileDownloadResponse` for each path.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error="invalid_path"
                    )
                )
                continue
            try:
                # Pre-check: verify the path exists, is a regular file, and
                # is readable before calling file.read().
                path_b64 = base64.b64encode(
                    path.encode("utf-8")
                ).decode("ascii")
                check_cmd = (
                    "python3 -c \""
                    "import os, base64; "
                    f"p = base64.b64decode('{path_b64}').decode(); "
                    "e = os.path.exists(p); "
                    "d = os.path.isdir(p) if e else False; "
                    "r = os.access(p, os.R_OK) if e else False; "
                    "print(f'exists:{e},dir:{d},read:{r}')"
                    "\""
                )
                check = self.execute(check_cmd)
                if check.exit_code == 0:
                    out = check.output.strip()
                    if "exists:False" in out:
                        responses.append(
                            FileDownloadResponse(
                                path=path, content=None, error="file_not_found"
                            )
                        )
                        continue
                    if "dir:True" in out:
                        responses.append(
                            FileDownloadResponse(
                                path=path, content=None, error="is_directory"
                            )
                        )
                        continue
                    if "read:False" in out:
                        responses.append(
                            FileDownloadResponse(
                                path=path,
                                content=None,
                                error="permission_denied",
                            )
                        )
                        continue

                result = self._sandbox.file.read(path=path)
                content_bytes = _extract_content_bytes(result)
                responses.append(
                    FileDownloadResponse(
                        path=path, content=content_bytes, error=None
                    )
                )
            except Exception as exc:  # noqa: BLE001 — SDK may raise any exception type on I/O failure
                logger.debug(
                    "Failed to download %s from FC sandbox",
                    path,
                    exc_info=True,
                )
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error=str(exc)
                    )
                )
        return responses


def _extract_content_bytes(result: Any) -> bytes:
    """Convert an agentrun file-read response to raw bytes.

    The API returns a dict with ``content`` (str) and ``encoding``
    (``"utf-8"`` or ``"base64"``).

    Args:
        result: Response dict from ``file.read()``.

    Returns:
        File content as raw bytes.

    Raises:
        ValueError: If the encoding is not recognized.
    """
    if isinstance(result, dict):
        content = result.get("content", "")
        encoding = result.get("encoding", "utf-8")
        if encoding == "base64":
            return base64.b64decode(content)
        return str(content).encode("utf-8")
    # Fallback: treat as string
    return str(result).encode("utf-8")

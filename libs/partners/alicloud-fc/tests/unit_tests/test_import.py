from __future__ import annotations

import base64
from unittest.mock import MagicMock

import langchain_alicloud_fc
from langchain_alicloud_fc.sandbox import AlicloudFCSandbox, _extract_content_bytes

COMMAND_TIMEOUT_EXIT_CODE = 124


def _make_sandbox(
    *,
    working_dir: str = "/home/user",
) -> tuple[AlicloudFCSandbox, MagicMock]:
    mock_sdk = MagicMock()
    mock_sdk.sandbox_id = "sb-fc-123"
    # Default success response for process.cmd() so that internal
    # execute() calls (mkdir -p, pre-checks, etc.) succeed.
    mock_sdk.process.cmd.return_value = {
        "executionId": "mock-id",
        "status": "completed",
        "result": {"exitCode": 0, "stdout": "", "stderr": ""},
    }
    sb = AlicloudFCSandbox(sandbox=mock_sdk, working_dir=working_dir)
    return sb, mock_sdk


def test_import_alicloud_fc() -> None:
    assert langchain_alicloud_fc is not None


def test_id_property() -> None:
    sb, _mock = _make_sandbox()
    assert sb.id == "sb-fc-123"


def test_id_property_none() -> None:
    mock_sdk = MagicMock()
    mock_sdk.sandbox_id = None
    sb = AlicloudFCSandbox(sandbox=mock_sdk)
    assert sb.id == ""


def test_execute_returns_stdout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.cmd.return_value = {
        "executionId": "test-id",
        "status": "completed",
        "result": {
            "exitCode": 0,
            "stdout": "hello world",
            "stderr": "",
        },
    }

    result = sb.execute("echo hello world")

    assert result.output == "hello world"
    assert result.exit_code == 0
    assert result.truncated is False
    mock_sdk.process.cmd.assert_called_once_with(
        command="echo hello world",
        cwd="/home/user",
        timeout=30,
    )


def test_execute_appends_stderr() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.cmd.return_value = {
        "executionId": "test-id",
        "status": "completed",
        "result": {
            "exitCode": 1,
            "stdout": "partial output",
            "stderr": "something went wrong",
        },
    }

    result = sb.execute("bad-command")

    assert result.exit_code == 1
    assert "partial output" in result.output
    assert "<stderr>something went wrong</stderr>" in result.output


def test_execute_empty_stderr_not_appended() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.cmd.return_value = {
        "executionId": "test-id",
        "status": "completed",
        "result": {
            "exitCode": 0,
            "stdout": "output only",
            "stderr": "   ",
        },
    }

    result = sb.execute("some-cmd")

    assert result.output == "output only"
    assert "<stderr>" not in result.output


def test_execute_custom_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.cmd.return_value = {
        "executionId": "test-id",
        "status": "completed",
        "result": {
            "exitCode": 0,
            "stdout": "done",
            "stderr": "",
        },
    }

    sb.execute("long-cmd", timeout=10)

    mock_sdk.process.cmd.assert_called_once_with(
        command="long-cmd",
        cwd="/home/user",
        timeout=10,
    )


def test_execute_custom_working_dir() -> None:
    sb, mock_sdk = _make_sandbox(working_dir="/workspace")
    mock_sdk.process.cmd.return_value = {
        "executionId": "test-id",
        "status": "completed",
        "result": {
            "exitCode": 0,
            "stdout": "",
            "stderr": "",
        },
    }

    sb.execute("ls")

    mock_sdk.process.cmd.assert_called_once_with(
        command="ls",
        cwd="/workspace",
        timeout=30,
    )


def test_execute_exception_returns_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.cmd.side_effect = TimeoutError("gateway timeout")

    result = sb.execute("sleep 999", timeout=15)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert "timed out" in result.output
    assert "15" in result.output


def test_execute_none_stdout_stderr() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.cmd.return_value = {
        "executionId": "test-id",
        "status": "completed",
        "result": {
            "exitCode": 0,
            "stdout": None,
            "stderr": None,
        },
    }

    result = sb.execute("true")

    assert result.output == ""
    assert result.exit_code == 0


def test_upload_files_success() -> None:
    sb, mock_sdk = _make_sandbox()

    responses = sb.upload_files([
        ("/home/user/test.py", b"print('hello')"),
        ("/home/user/data.txt", b"some data"),
    ])

    assert len(responses) == 2  # noqa: PLR2004 — expected count for two uploaded files
    assert all(r.error is None for r in responses)
    assert responses[0].path == "/home/user/test.py"
    assert responses[1].path == "/home/user/data.txt"
    # Small text files are written via execute() (process.cmd), not file.write().
    assert mock_sdk.file.write.call_count == 0


def test_upload_files_invalid_path() -> None:
    sb, mock_sdk = _make_sandbox()

    responses = sb.upload_files([("relative/path.txt", b"content")])

    assert len(responses) == 1
    assert responses[0].error == "invalid_path"
    assert responses[0].path == "relative/path.txt"
    mock_sdk.file.write.assert_not_called()


def test_upload_files_partial_failure() -> None:
    sb, mock_sdk = _make_sandbox()
    # Simulate execute() failure on the second file's write command.
    call_count = 0

    def cmd_side_effect(**kwargs: object) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        # Call sequence: mkdir -p (file1), write (file1), mkdir -p (file2),
        # write (file2).  Fail the 4th call (second file's write).
        if call_count == 4:  # noqa: PLR2004
            return {
                "executionId": "mock-id",
                "status": "completed",
                "result": {"exitCode": 1, "stdout": "disk full", "stderr": ""},
            }
        return {
            "executionId": "mock-id",
            "status": "completed",
            "result": {"exitCode": 0, "stdout": "", "stderr": ""},
        }

    mock_sdk.process.cmd.side_effect = cmd_side_effect

    responses = sb.upload_files([
        ("/home/user/ok.txt", b"ok"),
        ("/home/user/fail.txt", b"fail"),
    ])

    assert len(responses) == 2  # noqa: PLR2004 — expected count for two uploaded files
    assert responses[0].error is None
    assert responses[1].error is not None
    assert "disk full" in responses[1].error


def test_download_files_utf8() -> None:
    sb, mock_sdk = _make_sandbox()
    # Pre-check returns: file exists, not a directory, readable.
    mock_sdk.process.cmd.return_value = {
        "executionId": "mock-id",
        "status": "completed",
        "result": {
            "exitCode": 0,
            "stdout": "exists:True,dir:False,read:True",
            "stderr": "",
        },
    }
    mock_sdk.file.read.return_value = {
        "content": "hello world",
        "encoding": "utf-8",
    }

    responses = sb.download_files(["/home/user/test.txt"])

    assert len(responses) == 1
    assert responses[0].error is None
    assert responses[0].content == b"hello world"
    assert responses[0].path == "/home/user/test.txt"


def test_download_files_base64() -> None:
    sb, mock_sdk = _make_sandbox()
    original_bytes = b"\x00\x01\x02binary data"
    encoded = base64.b64encode(original_bytes).decode("ascii")
    # Pre-check returns: file exists, not a directory, readable.
    mock_sdk.process.cmd.return_value = {
        "executionId": "mock-id",
        "status": "completed",
        "result": {
            "exitCode": 0,
            "stdout": "exists:True,dir:False,read:True",
            "stderr": "",
        },
    }
    mock_sdk.file.read.return_value = {
        "content": encoded,
        "encoding": "base64",
    }

    responses = sb.download_files(["/home/user/binary.bin"])

    assert len(responses) == 1
    assert responses[0].error is None
    assert responses[0].content == original_bytes


def test_download_files_invalid_path() -> None:
    sb, mock_sdk = _make_sandbox()

    responses = sb.download_files(["relative/path.txt"])

    assert len(responses) == 1
    assert responses[0].error == "invalid_path"
    assert responses[0].content is None
    mock_sdk.file.read.assert_not_called()


def test_download_files_failure() -> None:
    sb, mock_sdk = _make_sandbox()
    # Pre-check returns: file does not exist.
    mock_sdk.process.cmd.return_value = {
        "executionId": "mock-id",
        "status": "completed",
        "result": {
            "exitCode": 0,
            "stdout": "exists:False,dir:False,read:False",
            "stderr": "",
        },
    }

    responses = sb.download_files(["/nonexistent.txt"])

    assert len(responses) == 1
    assert responses[0].error == "file_not_found"
    assert responses[0].content is None
    mock_sdk.file.read.assert_not_called()


def test_download_files_directory() -> None:
    sb, mock_sdk = _make_sandbox()
    # Pre-check returns: path is a directory.
    mock_sdk.process.cmd.return_value = {
        "executionId": "mock-id",
        "status": "completed",
        "result": {
            "exitCode": 0,
            "stdout": "exists:True,dir:True,read:True",
            "stderr": "",
        },
    }

    responses = sb.download_files(["/home/user/testdir"])

    assert len(responses) == 1
    assert responses[0].content is None
    assert responses[0].error == "is_directory"
    mock_sdk.file.read.assert_not_called()


def test_download_files_permission_denied() -> None:
    sb, mock_sdk = _make_sandbox()
    # Pre-check returns: file exists but is not readable.
    mock_sdk.process.cmd.return_value = {
        "executionId": "mock-id",
        "status": "completed",
        "result": {
            "exitCode": 0,
            "stdout": "exists:True,dir:False,read:False",
            "stderr": "",
        },
    }

    responses = sb.download_files(["/home/user/secret.txt"])

    assert len(responses) == 1
    assert responses[0].content is None
    assert responses[0].error == "permission_denied"
    mock_sdk.file.read.assert_not_called()


def test_upload_files_empty_content() -> None:
    sb, mock_sdk = _make_sandbox()

    responses = sb.upload_files([("/home/user/empty.txt", b"")])

    assert len(responses) == 1
    assert responses[0].error is None
    assert responses[0].path == "/home/user/empty.txt"
    # Empty content is created via execute(), not file.write().
    mock_sdk.file.write.assert_not_called()


def test_extract_content_bytes_utf8() -> None:
    result = _extract_content_bytes({"content": "text data", "encoding": "utf-8"})
    assert result == b"text data"


def test_extract_content_bytes_base64() -> None:
    original = b"binary\x00data"
    encoded = base64.b64encode(original).decode("ascii")
    result = _extract_content_bytes({"content": encoded, "encoding": "base64"})
    assert result == original


def test_extract_content_bytes_fallback_string() -> None:
    result = _extract_content_bytes("plain string response")
    assert result == b"plain string response"


def test_extract_content_bytes_missing_encoding() -> None:
    result = _extract_content_bytes({"content": "data"})
    assert result == b"data"

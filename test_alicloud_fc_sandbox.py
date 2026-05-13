"""Comprehensive test script for Alicloud FC Sandbox with Deep Agents.

Exercises all four sandbox capability areas:
  1. Planning     — write_todos for task breakdown and progress tracking
  2. Filesystem   — read_file, write_file, edit_file, ls, glob, grep
  3. Shell access  — execute for running commands
  4. Sub-agents   — task for delegating work with isolated context windows

Usage:
    cd libs/partners/alicloud-fc
    uv run --with python-dotenv --with langchain-openai \
        python ../../../test_alicloud_fc_sandbox.py

Requires:
    - .env at project root with AGENTRUN_* credentials
    - OPENAI_API_KEY and OPENAI_BASE_URL set in .env or shell environment
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

missing = []
for var in (
    "AGENTRUN_ACCESS_KEY_ID",
    "AGENTRUN_ACCESS_KEY_SECRET",
    "AGENTRUN_ACCOUNT_ID",
    "AGENTRUN_REGION",
    "DASHSCOPE_API_KEY",
):
    if not os.environ.get(var):
        missing.append(var)
if missing:
    print(f"Error: missing required environment variables: {', '.join(missing)}")
    print(f"Set them in {env_path} or export in shell.")
    sys.exit(1)

from langchain_openai import ChatOpenAI  # noqa: E402

model = ChatOpenAI(
    model="glm-5.1",
    #model="qwen3.6-plus",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    use_responses_api=False,
)

from agentrun.sandbox import Sandbox, TemplateInput, TemplateType

from deepagents import create_deep_agent
from langchain_alicloud_fc import AlicloudFCSandbox

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"


def print_section(name: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}\n")


def run_test(
    name: str,
    agent: object,
    prompt: str,
    keywords: list[str],
) -> tuple[str, str]:
    """Invoke *agent* with *prompt* and validate the response.

    Args:
        name: Human-readable test name.
        agent: Compiled agent graph returned by ``create_deep_agent``.
        prompt: The user message to send.
        keywords: Phrases that should appear in the response (case-insensitive).

    Returns:
        ``(status, detail)`` where status is PASS / FAIL / ERROR.
    """
    print_section(name)
    print(f"Prompt:\n{prompt}\n")
    start = time.time()

    result = agent.invoke(  # type: ignore[union-attr]
        {"messages": [{"role": "user", "content": prompt}]}
    )
    elapsed = time.time() - start

    raw_content = result["messages"][-1].content
    # content may be a plain string or a list of content blocks
    if isinstance(raw_content, list):
        content = "\n".join(
            block.get("text", "") for block in raw_content if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        content = str(raw_content)
    print(f"Response ({elapsed:.1f}s):\n{content}\n")

    lower = content.lower()
    missing_kw = [kw for kw in keywords if kw.lower() not in lower]
    if missing_kw:
        detail = f"Missing expected keywords: {missing_kw}"
        print(f"[FAIL] {detail}")
        return FAIL, detail

    print("[PASS]")
    return PASS, "OK"


def print_summary(results: list[tuple[str, str, str]]) -> None:
    """Print a final results table."""
    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for name, status, detail in results:
        print(f"  [{status}] {name}: {detail}")
    total = len(results)
    passed = sum(1 for _, s, _ in results if s == PASS)
    print(f"\n  {passed}/{total} tests passed.")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Sandbox setup
# ---------------------------------------------------------------------------

template_name = "test-alicloud-fc-sandbox"
try:
    Sandbox.create_template(
        input=TemplateInput(
            template_name=template_name,
            template_type=TemplateType.CODE_INTERPRETER,
        )
    )
except Exception:  # noqa: BLE001, S110 — template may already exist
    pass

sandbox = Sandbox.create(
    template_type=TemplateType.CODE_INTERPRETER,
    template_name=template_name,
)
backend = AlicloudFCSandbox(sandbox=sandbox)

results: list[tuple[str, str, str]] = []

try:
    # ------------------------------------------------------------------
    # Agent for tests 1-3 (shared)
    # ------------------------------------------------------------------
    agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You are a Python coding assistant with sandbox access. "
            "Always report the results of your actions clearly."
        ),
        backend=backend,
    )

    # ==================================================================
    # Test 1: Planning (write_todos)
    # ==================================================================
    try:
        status, detail = run_test(
            name="Planning (write_todos)",
            agent=agent,
            prompt=(
                "You MUST use your write_todos tool to plan and track "
                "the following 3-step task. Mark each step in_progress "
                "before starting and completed after finishing.\n\n"
                "Steps:\n"
                "1. Create a file /home/user/plan_test/greeting.txt "
                "containing 'Hello, Deep Agents!'\n"
                "2. Read that file back and confirm the content\n"
                "3. Delete the file using a shell command\n\n"
                "After all steps, confirm everything is done."
            ),
            keywords=["greeting", "Hello"],
        )
        results.append(("Planning (write_todos)", status, detail))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        results.append(("Planning (write_todos)", ERROR, str(exc)))

    # ==================================================================
    # Test 2: Filesystem operations
    # ==================================================================
    try:
        status, detail = run_test(
            name="Filesystem operations",
            agent=agent,
            prompt=(
                "Perform the following filesystem operations and report "
                "the result of each step:\n\n"
                "1. Use write_file to create /home/user/fs_test/hello.py "
                "containing:\n"
                "   def greet(name):\n"
                "       return f'Hello, {name}!'\n\n"
                "2. Use write_file to create /home/user/fs_test/utils.py "
                "containing:\n"
                "   def add(a, b):\n"
                "       return a + b\n\n"
                "3. Use ls to list /home/user/fs_test/\n"
                "4. Use read_file to read /home/user/fs_test/hello.py\n"
                "5. Use edit_file to change 'Hello' to 'Hi' in hello.py\n"
                "6. Use glob with pattern '*.py' to find all Python files "
                "under /home/user/fs_test/\n"
                "7. Use grep to search for 'def ' in /home/user/fs_test/ and show the full matching lines\n\n"
                "Report the output of every step clearly."
            ),
            keywords=["hello.py", "utils.py", "Hi", "greet", "add"],
        )
        results.append(("Filesystem operations", status, detail))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        results.append(("Filesystem operations", ERROR, str(exc)))

    # ==================================================================
    # Test 3: Shell execution (execute)
    # ==================================================================
    try:
        status, detail = run_test(
            name="Shell execution (execute)",
            agent=agent,
            prompt=(
                "Run the following shell commands using the execute tool "
                "and report the output of each:\n\n"
                "1. python3 --version\n"
                "2. echo 'Hello from sandbox'\n"
                "3. python3 -c \"print(2 + 2)\"\n\n"
                "Report each command's output clearly."
            ),
            keywords=["Python 3", "Hello from sandbox", "4"],
        )
        results.append(("Shell execution (execute)", status, detail))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        results.append(("Shell execution (execute)", ERROR, str(exc)))

    # ==================================================================
    # Test 4: Sub-agents (task)
    # ==================================================================
    try:
        agent_with_subagents = create_deep_agent(
            model=model,
            system_prompt=(
                "You are a task coordinator. When asked to delegate work, "
                "you MUST use the task tool to dispatch to the appropriate "
                "subagent. After the subagent finishes, verify its output."
            ),
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "model": model,
                    "description": (
                        "A research agent that investigates the sandbox "
                        "environment and writes reports to files."
                    ),
                    "system_prompt": (
                        "You are a research assistant. Investigate topics "
                        "using available tools (execute, write_file, etc.) "
                        "and write concise findings to the specified file."
                    ),
                }
            ],
        )

        status, detail = run_test(
            name="Sub-agents (task)",
            agent=agent_with_subagents,
            prompt=(
                "Delegate the following task to the 'researcher' subagent:\n\n"
                "  Determine which Python version is installed in the sandbox "
                "  by running 'python3 --version', then write a short report "
                "  to /home/user/subagent_test/report.txt that includes the "
                "  Python version.\n\n"
                "After the subagent completes, read "
                "/home/user/subagent_test/report.txt and summarize the result."
            ),
            keywords=["Python", "report"],
        )
        results.append(("Sub-agents (task)", status, detail))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        results.append(("Sub-agents (task)", ERROR, str(exc)))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(results)

finally:
    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    try:
        sandbox.delete()
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        Sandbox.delete_template(template_name)
    except Exception:  # noqa: BLE001, S110
        pass

# Exit with non-zero if any test did not pass.
if any(s != PASS for _, s, _ in results):
    sys.exit(1)

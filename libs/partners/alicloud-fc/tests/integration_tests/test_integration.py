from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import os
from dotenv import load_dotenv

from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_alicloud_fc import AlicloudFCSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol

# 加载同目录下的 .env 文件
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)

class TestAlicloudFCSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        import time

        template_name = f"ci-test-{time.strftime('%Y%m%d%H%M%S')}"
        Sandbox.create_template(
            input=TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
            )
        )
        sandbox = Sandbox.create(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name=template_name,
        )
        try:
            with sandbox:
                backend = AlicloudFCSandbox(sandbox=sandbox)
                yield backend
        finally:
            try:
                sandbox.delete()
            except Exception:  # noqa: BLE001, S110 — context manager may already delete the sandbox
                pass
            try:
                Sandbox.delete_template(template_name)
            except Exception:  # noqa: BLE001, S110 — best-effort cleanup
                pass

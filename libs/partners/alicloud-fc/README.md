# langchain-alicloud-fc

Alibaba Cloud Function Compute sandbox integration for [Deep Agents](https://github.com/langchain-ai/deepagents).

## Installation

```bash
pip install langchain-alicloud-fc
```

## Configuration

Set the following environment variables:

| Variable | Description | Fallback |
|---|---|---|
| `AGENTRUN_ACCESS_KEY_ID` | Access Key ID | `ALIBABA_CLOUD_ACCESS_KEY_ID` |
| `AGENTRUN_ACCESS_KEY_SECRET` | Access Key Secret | `ALIBABA_CLOUD_ACCESS_KEY_SECRET` |
| `AGENTRUN_ACCOUNT_ID` | Alibaba Cloud Account ID | `FC_ACCOUNT_ID` |
| `AGENTRUN_REGION` | Region (e.g. `cn-hangzhou`) | `FC_REGION` |

## Usage

```python
from agentrun.sandbox import Sandbox, TemplateType
from langchain_alicloud_fc import AlicloudFCSandbox

sandbox = Sandbox.create(
    template_type=TemplateType.CODE_INTERPRETER,
    template_name="your-template-name",
)

backend = AlicloudFCSandbox(sandbox=sandbox)
result = backend.execute("echo hello")
print(result.output)

# Cleanup
sandbox.delete()
```

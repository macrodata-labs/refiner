---
title: "Secrets and Environment"
description: "Pass credentials and configuration to cloud jobs"
---

# Secrets And Environment

Cloud jobs often need credentials for storage, model providers, or private
datasets. Do not hard-code these values in pipeline code.

## Submit Local Environment Values

```python
pipeline.launch_cloud(
    name="private-hf-read",
    secrets={"HF_TOKEN": None},
)
```

`None` means Refiner reads `HF_TOKEN` from your local environment at submission
time and passes it as a secret value.

## Use Stored Workspace Secrets

```python
pipeline.launch_cloud(
    name="private-hf-read",
    secrets=mdr.Secrets.env(name="production", keys=["HF_TOKEN"]),
)
```

Use stored secrets for shared production workflows.

## Non-Secret Environment

Use `env` for configuration that is not sensitive:

```python
pipeline.launch_cloud(
    name="configured-job",
    env={"RUN_LABEL": "experiment-17"},
)
```

## Related Pages

- [CLI Secrets](../cli/secrets.md)
- [Cloud Launcher](../running-pipelines/cloud-launcher.md)

---
title: "CLI Secrets"
description: "Manage workspace secrets from the Macrodata CLI"
---

# CLI Secrets

Use CLI secrets for values that cloud jobs need but source code should not
contain.

## List Secrets

```bash
macrodata secrets list --env production
```

## Set A Secret

```bash
printf '%s' "$HF_TOKEN" | macrodata secrets set HF_TOKEN --env production
```

The CLI reads the secret value from stdin.

## Remove A Secret

```bash
macrodata secrets remove HF_TOKEN --env production
```

## Use In A Pipeline

```python
pipeline.launch_cloud(
    name="private-dataset",
    secrets=mdr.Secrets.env(name="production", keys=["HF_TOKEN"]),
)
```

See [Secrets and Environment](../platform/secrets-and-environment.md).

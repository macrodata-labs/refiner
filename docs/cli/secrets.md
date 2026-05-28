---
title: "CLI Secrets"
description: "Manage workspace secrets from the Macrodata CLI"
---

# CLI Secrets

Use CLI secrets for values that cloud jobs need but source code should not
contain. Secret values are stored in a workspace environment and are injected
into launched jobs only when the pipeline requests them.

## List Secrets

```bash
macrodata secrets list --env production
macrodata secrets list --env production --json
```

`secrets list` prints secret names, not secret values. Use it to confirm that a
workspace environment contains the keys a cloud job expects.

Example output:

```text
Env         Name      Updated
production  HF_TOKEN  2026-05-28T12:30:00.000Z
production  WANDB_KEY 2026-05-28T12:31:18.000Z
```

If the environment has no stored secrets, the command prints:

```text
No secrets found.
```

| Option | Use |
| --- | --- |
| `--env <name>` | Secret environment to list |
| `--json` | Print raw JSON |

## Set A Secret

```bash
printf '%s' "$HF_TOKEN" | macrodata secrets set HF_TOKEN --env production --value-stdin
printf '%s' "$HF_TOKEN" | macrodata secrets set HF_TOKEN --env production --value-stdin --json
```

`secrets set` adds or replaces one secret in a workspace environment. The CLI
reads the secret value from stdin so the value does not appear in shell history
or process arguments.

Example output:

```text
Saved secret production/HF_TOKEN.
```

If you omit `--env`, the secret is stored in the `default` environment:

```bash
printf '%s' "$OPENAI_API_KEY" | macrodata secrets set OPENAI_API_KEY --value-stdin
```

| Option | Use |
| --- | --- |
| `--env <name>` | Secret environment; defaults to `default` |
| `--value-stdin` | Read the secret value from stdin |
| `--json` | Print raw JSON |

## Remove A Secret

```bash
macrodata secrets remove HF_TOKEN --env production
macrodata secrets remove HF_TOKEN --env production --json
```

`secrets remove` deletes one stored secret name from an environment. The alias
`secrets delete` does the same thing.

Example output:

```text
Removed secret production/HF_TOKEN.
```

| Option | Use |
| --- | --- |
| `--env <name>` | Secret environment; defaults to `default` |
| `--json` | Print raw JSON |

## Use In A Pipeline

After storing a secret, request it from the cloud launcher by environment name
and key. Refiner makes the value available to the job without writing it into
the manifest or source code.

```python
pipeline.launch_cloud(
    name="private-dataset",
    secrets=mdr.Secrets.env(name="production", keys=["HF_TOKEN"]),
)
```

See [Secrets and Environment](../platform/secrets-and-environment.md).

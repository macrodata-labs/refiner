---
title: "Secrets and Environment"
description: "Pass credentials and configuration to cloud jobs and the viewer"
---

# Secrets And Environment

Cloud jobs and the viewer often need credentials for storage, model providers,
or private datasets. Store those values as workspace secrets instead of
hard-coding them in pipeline code.

Open [Settings > Secrets](/settings/secrets).

## Secret Environments

Secrets are grouped by environment. Use environments to keep separate versions
of the same key:

| Environment | Example use |
| --- | --- |
| `default` | Personal experiments and first runs. |
| `staging` | Test buckets, staging provider keys, smaller datasets. |
| `production` | Shared production jobs and durable dataset refreshes. |

The secrets page has an **Environment** selector. Choose an existing
environment or choose **Add environment** to create a new environment name.

## Add A Secret In The UI

1. Open [Settings > Secrets](/settings/secrets).
2. Select the environment, such as `production`.
3. Click **Add Secret**.
4. Enter the environment, name, and value.
5. Click **Save Secret**.

Secret values are encrypted before storage. The table shows name, created time,
updated time, and actions. It does not show values.

Use **Overwrite** to replace a value and **Delete** to remove it permanently.

## Add A Secret From The CLI

```bash
printf '%s' "$HF_TOKEN" | macrodata secrets set HF_TOKEN --env production --value-stdin
macrodata secrets list --env production
```

See [CLI Secrets](../cli/secrets.md).

## Pass Secrets In Code

Use `mdr.Secrets.dict(...)` when the value is available in Python at submission
time:

```python
pipeline.launch_cloud(
    name="private-hf-read",
    secrets=mdr.Secrets.dict({"HF_TOKEN": "---"}),
)
```

The value is sent as a cloud job secret and redacted from logs. Replace `"---"`
with the real token before submitting. You can also pass a plain mapping, but
`mdr.Secrets.dict(...)` makes the intent explicit:

```python
pipeline.launch_cloud(
    name="private-hf-read",
    secrets={"HF_TOKEN": "---"},
)
```

## Submit Local Environment Values

Use this when the secret value exists on your submitting machine and should be
sent with this job:

```python
pipeline.launch_cloud(
    name="private-hf-read",
    secrets=mdr.Secrets.dict({"HF_TOKEN": None}),
)
```

`None` means Refiner reads `HF_TOKEN` from your local environment at submission
time and passes the value as a secret for the job.

## Load A Dotenv File

Use `mdr.Secrets.dotenv(...)` to load job secrets from a local dotenv file at
submission time:

```python
pipeline.launch_cloud(
    name="private-hf-read",
    secrets=mdr.Secrets.dotenv(".env"),
)
```

For example, `.env` can contain:

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

The dotenv file is read locally when you submit the job. The file itself is not
uploaded.

## Use Stored Workspace Secrets

Use stored secrets for shared workflows:

```python
pipeline.launch_cloud(
    name="private-hf-read",
    secrets=mdr.Secrets.env(name="production", keys=["HF_TOKEN"]),
)
```

The job references the `production` environment and the `HF_TOKEN` name. The
manifest records the reference, not the value.

## Non-Secret Environment

Use `env` for configuration that is not sensitive:

```python
pipeline.launch_cloud(
    name="configured-job",
    env={"RUN_LABEL": "experiment-17"},
)
```

Use secrets for tokens, credentials, private keys, and passwords. Use `env` for
labels, feature flags, batch sizes, or other non-sensitive values.

## Viewer Secrets

The [Viewer](viewer.md) uses the selected workspace secret environment to open
private S3, GCS, and Hugging Face files.

Use these names:

| Storage | Secret names |
| --- | --- |
| S3-compatible storage | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, `AWS_REGION` or `AWS_DEFAULT_REGION`, optional `AWS_ENDPOINT_URL` |
| Google Cloud Storage | `GOOGLE_APPLICATION_CREDENTIALS_JSON` |
| Hugging Face | `HF_TOKEN` |

Open [Viewer](/viewer), choose the same environment, paste the file path, and
click **Load**.

## Related Pages

- [Submitting to the Platform](submitting-to-the-platform.md)
- [Viewer](viewer.md)
- [CLI Secrets](../cli/secrets.md)
- [Cloud Launcher](../running-pipelines/cloud-launcher.md)

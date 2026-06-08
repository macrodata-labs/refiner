---
title: "Environment variables"
description: "Predefined variables that affect Refiner, the CLI, and local workers"
---

# Environment variables

This page lists predefined environment variables that Refiner reads directly.
For arbitrary non-secret variables passed to cloud workers, use
[`launch_cloud(env=...)`](secrets-and-environment.md#non-secret-environment).
For tokens and credentials, use
[Secrets and environment](secrets-and-environment.md) when submitting cloud
jobs.

## Platform and authentication

| Variable | Effect |
| --- | --- |
| `MACRODATA_API_KEY` | Authenticates the CLI and Python cloud launcher. Refiner checks this before the local credentials file written by `macrodata login`. |
| `MACRODATA_BASE_URL` | Overrides the Macrodata API base URL. The default is `https://macrodata.co`. |
| `XDG_CONFIG_HOME` | Changes where `macrodata login` stores and reads the local API key file. |

Use `MACRODATA_API_KEY` for CI, agent runs, or any environment where writing a
local credentials file is not desirable:

```bash
export MACRODATA_API_KEY="md_..."
macrodata whoami
```

## CLI run behavior

| Variable | Values | Effect |
| --- | --- | --- |
| `REFINER_ATTACH` | `auto`, `attach`, `detach` | Controls whether cloud launches attach to live output or return after submission. |
| `REFINER_LOGS` | `all`, `none`, `one`, `errors` | Controls worker log output for run/attach flows. |

Command-line flags can override these values for a single run.

## Local execution

| Variable | Effect |
| --- | --- |
| `REFINER_WORKDIR` | Sets the local worker run directory. The path must be absolute. |
| `XDG_CACHE_HOME` | Changes the default local work directory when `REFINER_WORKDIR` is not set. |
| `CUDA_VISIBLE_DEVICES` | Restricts or defines GPU IDs visible to local workers. Refiner also sets this for worker processes after assigning GPUs. |

Without `REFINER_WORKDIR`, local worker files go under
`$XDG_CACHE_HOME/macrodata/refiner` or `~/.cache/macrodata/refiner`.

## Cloud launch dependency fallback

| Variable | Values | Effect |
| --- | --- | --- |
| `MACRODATA_FALLBACK_TO_LATEST_PYPI` | `1`, `true`, `yes`, `on` | Allows non-interactive cloud launch to fall back to the latest PyPI package when the captured local Refiner ref is not available remotely. |

Leave this unset unless you intentionally want that fallback.

## Provider and data credentials

These variables are read by specific Refiner readers or inference providers
when an API key is not passed directly:

| Variable | Used by |
| --- | --- |
| `HF_TOKEN` | Hugging Face HTTP paths. |
| `OPENAI_API_KEY` | OpenAI and OpenAI-compatible inference providers. |
| `ANTHROPIC_API_KEY` | Anthropic inference provider. |
| `GOOGLE_GENERATIVE_AI_API_KEY` | Google inference provider. |

For cloud jobs, pass these as secrets instead of plain environment variables so
their values are redacted from logs and manifests.

## Related pages

- [Submitting to the platform](submitting-to-the-platform.md)
- [Secrets and environment](secrets-and-environment.md)
- [Cloud launcher](../running-pipelines/cloud-launcher.md)
- [Auth and run](../cli/auth-and-run.md)

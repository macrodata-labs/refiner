---
title: "Workspaces and API Keys"
description: "Authenticate Refiner with Macrodata workspaces"
---

# Workspaces And API Keys

A workspace owns cloud jobs, secrets, logs, metrics, and files. Refiner uses an
API key to authenticate CLI and cloud-run requests.

## Login

```bash
macrodata login
```

This stores credentials locally for the CLI and Python launchers.

## Environment Variable

For CI or scripted environments:

```bash
export MACRODATA_API_KEY="md_..."
```

Environment credentials take precedence over saved local credentials.

## Check Auth

```bash
macrodata whoami
```

## Related Pages

- [Cloud Launcher](../running-pipelines/cloud-launcher.md)
- [CLI Auth and Run](../cli/auth-and-run.md)

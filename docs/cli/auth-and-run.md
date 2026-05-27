---
title: "Auth and Run"
description: "Authenticate and run Refiner scripts from the CLI"
---

# Auth And Run

## Login

```bash
macrodata login
```

## Check Identity

```bash
macrodata whoami
```

## Logout

```bash
macrodata logout
```

## Run A Script

```bash
macrodata run train_data_pipeline.py
```

The script should build and launch a Refiner pipeline. Use `macrodata run` when
you want CLI-managed execution behavior around a script.

## Related Pages

- [Workspaces and API Keys](../platform/workspaces-and-api-keys.md)
- [Cloud Launcher](../running-pipelines/cloud-launcher.md)

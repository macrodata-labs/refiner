---
title: "CLI"
description: "Macrodata CLI commands used with Refiner"
---

Refiner installs the `macrodata` CLI.

## Auth

Create a Macrodata API key:

- https://macrodata.co/settings/api-keys

### Login

```bash
macrodata login
```

Non-interactive options:

```bash
macrodata login --token md_xxx
printf '%s' 'md_xxx' | macrodata login --token-stdin
```

### Check Current Auth

```bash
macrodata whoami
```

### Logout

```bash
macrodata logout
```

## Credential Lookup

See [Auth](auth.md) for the shared credential lookup order and credential file location.

## Notes

- `launch_local(...)` does not require Macrodata auth
- `launch_cloud(...)` requires Macrodata auth
